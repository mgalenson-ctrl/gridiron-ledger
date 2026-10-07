"""Fit final models on all played games, predict the current week, and write the app data bundle."""
import numpy as np, pandas as pd, json, hashlib, os
from datetime import datetime, timezone
from train import *

MODEL_VERSION = 'v1.1.0'
EARLY_GROUPS = ['baseline','qb','special_discipline','matchup','weather']
UPDATED_GROUPS = EARLY_GROUPS + ['injuries']
C_FINAL = 0.05
meta = json.load(open(f'{OUT}/data_meta.json'))

LABELS = {
 'home_adv':'Home-field advantage','d_adj_off_off_epa':'Opponent-adjusted offensive EPA/play','d_adj_def_off_epa':'Opponent-adjusted defensive EPA/play',
 'd_qb_epa':'QB EPA per dropback (trailing ~400 dropbacks)','d_qb_cpoe':'QB completion % over expected','d_qb_int_rate':'QB interception rate',
 'd_qb_sack_rate':'QB sack rate','d_qb_scr_epa_per_db':'QB scramble value','d_qb_change':'QB change from last game','d_qb_db_hist':'QB experience (dropbacks in window)',
 'd_fg_rate':'Field-goal accuracy','d_fg_long_rate':'Field-goal accuracy 40+ yds','d_st_epa_per_game':'Special-teams EPA/game',
 'd_to_per_game':'Giveaways per game','d_takeaways_per_game':'Takeaways per game','d_pen_per_game':'Penalties per game',
 'd_pp_vs_rush':'Pass protection vs pass rush','d_rush_vs_rundef':'Rushing offense vs run defense','d_pass_vs_passdef':'Passing offense vs pass defense',
 'd_explosive_edge':'Explosive plays gained vs allowed',
 'wx_wind':'Wind at kickoff','wx_cold':'Cold at kickoff (below 45°F)','wx_wind_x_passlean':'Wind × pass-leaning offense','wx_cold_x_awaydome':'Cold vs visiting dome team','wx_wind_x_fg':'Wind × long field-goal edge',
 'd_inj_starters_out':'Starters out/doubtful','d_inj_starters_q':'Starters questionable','d_inj_qb_out':'QB listed out','d_inj_ol_out':'OL starters out',
}

def fit_final(groups):
    cols = cols_for(groups)
    tr = F.played
    model, med = fit(X_all.loc[tr, cols], y_all[tr], C_FINAL)
    return model, med, cols

def contributions(model, med, cols, Xrow):
    sc, lr = model.named_steps['standardscaler'], model.named_steps['logisticregression']
    x = Xrow[cols].fillna(med).values.astype(float)
    z = (x - sc.mean_) / sc.scale_
    contrib = z * lr.coef_[0]
    return dict(zip(cols, contrib)), float(lr.intercept_[0])

# ---------------------------------------------------------------- apply live forecasts to the current week's games
cur_season = int(F.season.max())
cur_week = int(F[(F.season==cur_season) & (~F.played)].week.min())
FC = {}
if os.path.exists(f'{OUT}/forecasts.json'):
    fj = json.load(open(f'{OUT}/forecasts.json'))
    if fj['season']==cur_season and fj['week']==cur_week: FC = fj['forecasts']
wx_note = {}
for idx, row in F[(F.season==cur_season)&(F.week==cur_week)].iterrows():
    roof = row.roof if isinstance(row.roof,str) else None
    if roof in ('dome','closed'):
        wx_note[row.game_id] = {'status':'roofed','summary':'Roofed stadium; weather not a factor.'}; continue
    fc = FC.get(row.game_id)
    if fc and fc.get('status')=='ok':
        F.loc[idx,'wx_wind'] = fc['wind_mph']; F.loc[idx,'wx_temp'] = fc['temp_f']; F.loc[idx,'wx_source'] = 'forecast'
        wx_note[row.game_id] = {'status':'forecast', **{k:fc.get(k) for k in ('temp_f','wind_mph','gust_mph','pop_pct','valid_for','forecast_created','fetched_at','source')},
                                'retractable': roof is None}
    else:
        F.loc[idx,'wx_wind'] = np.nan; F.loc[idx,'wx_temp'] = np.nan; F.loc[idx,'wx_source'] = 'missing'
        wx_note[row.game_id] = {'status': (fc or {}).get('status','unavailable'), 'summary':'Forecast unavailable; weather inputs imputed with training medians.'}
X_all = build_X(F)

early_model, early_med, early_cols = fit_final(EARLY_GROUPS)
upd_model, upd_med, upd_cols = fit_final(UPDATED_GROUPS)

# ---------------------------------------------------------------- score models (ridge; margin on EARLY cols, total on team sums)
SCORE_ALPHA = 100.0
T_all = build_T(F)
margin_model, margin_med = fit_reg(X_all.loc[F.played, early_cols], F.result[F.played], SCORE_ALPHA)
total_model, total_med = fit_reg(T_all[F.played], F.total[F.played], SCORE_ALPHA)
PS = walk_forward_scores(early_cols, SCORE_ALPHA)
PSd = PS.dropna(subset=['result','total'])
def mae(a,b): return float(np.mean(np.abs(a-b)))
score_metrics = {'n': int(len(PSd)), 'margin_mae': mae(PSd.pred_margin, PSd.result), 'spread_mae': mae(PSd.spread_line, PSd.result),
                 'total_mae': mae(PSd.pred_total, PSd.total), 'market_total_mae': mae(PSd.total_line, PSd.total),
                 'margin_within_7': float((np.abs(PSd.pred_margin-PSd.result)<=7).mean()),
                 'holdout_2025_on': {'margin_mae': mae(PSd[PSd.season>=2025].pred_margin, PSd[PSd.season>=2025].result), 'spread_mae': mae(PSd[PSd.season>=2025].spread_line, PSd[PSd.season>=2025].result),
                                     'total_mae': mae(PSd[PSd.season>=2025].pred_total, PSd[PSd.season>=2025].total), 'market_total_mae': mae(PSd[PSd.season>=2025].total_line, PSd[PSd.season>=2025].total)}}
def score_pred(idx):
    m = float(margin_model.predict(X_all.loc[[idx], early_cols].fillna(margin_med))[0]); t = float(total_model.predict(T_all.loc[[idx]].fillna(total_med))[0])
    h = (t+m)/2; a = (t-m)/2
    return {'margin': round(m,1), 'total': round(t,1), 'home': int(round(h)), 'away': int(round(a)), 'adjusted': False}
def reconcile(score, pick, home, away):
    # the pick comes from the win-probability model; if rounding or a near-zero margin leaves the score tied or pointing the other way,
    # bump the picked team to a one-point lead so the displayed score never contradicts the pick (flagged as adjusted)
    hs, as_ = score['home'], score['away']
    if (pick==home and hs<=as_): score['home'] = as_+1; score['adjusted']=True
    if (pick==away and as_<=hs): score['away'] = hs+1; score['adjusted']=True
    return score

# ---------------------------------------------------------------- walk-forward history for dashboard (final config)
P_early = walk_forward(EARLY_GROUPS, C=C_FINAL); res_early, P_early = evaluate(P_early)
P_upd = walk_forward(UPDATED_GROUPS, C=C_FINAL); res_upd, P_upd = evaluate(P_upd)
P_base = walk_forward(['baseline'], C=C_FINAL); res_base, P_base = evaluate(P_base)
ablation = json.load(open(f'{OUT}/ablation_results.json'))

# ---------------------------------------------------------------- current week
wk = F[(F.season==cur_season)&(F.week==cur_week)].copy()
# prior predictions log (keeps the first EARLY prediction per game across reruns)
LOG_PATH = f'{OUT}/predictions_log.json'
LOG = json.load(open(LOG_PATH)) if os.path.exists(LOG_PATH) else {}
Xwk = X_all.loc[wk.index]
input_hash = hashlib.sha256(pd.util.hash_pandas_object(Xwk[upd_cols].round(6)).values.tobytes()).hexdigest()[:16]
now = datetime.now(timezone.utc).isoformat(timespec='seconds')
injuries_available = bool((wk.h_inj_available==1).all())

def team_profile(row, s):
    g = lambda k: (None if pd.isna(row.get(f'{s}_{k}')) else float(row.get(f'{s}_{k}')))
    return {'team': row[f'{"home" if s=="h" else "away"}_team'],
            'games_played': g('games_played'), 'adj_off_epa': g('adj_off_off_epa'), 'adj_def_epa': g('adj_def_off_epa'),
            'adj_off_sr': g('adj_off_off_sr'), 'adj_pass_epa': g('adj_off_pass_epa'), 'adj_rush_epa': g('adj_off_rush_epa'),
            'adj_def_pass_epa': g('adj_def_pass_epa'), 'adj_def_rush_epa': g('adj_def_rush_epa'), 'adj_ppd': g('adj_off_ppd'),
            'l5_off_epa': g('l5_off_epa'), 'l5_def_epa': g('l5_def_epa'),
            'qb_name': row.get(f'{s}_qb_name') if isinstance(row.get(f'{s}_qb_name'), str) else None,
            'qb_epa': g('qb_epa'), 'qb_cpoe': g('qb_cpoe'), 'qb_int_rate': g('qb_int_rate'), 'qb_sack_rate': g('qb_sack_rate'),
            'qb_db_hist': g('qb_db_hist'), 'qb_change': g('qb_change'), 'qb_blitz_epa': g('qb_blitz_epa'),
            'third_rate': g('third_rate'), 'rz_td_rate': g('rz_td_rate'), 'fg_rate': g('fg_rate'), 'fg_long_rate': g('fg_long_rate'),
            'to_per_game': g('to_per_game'), 'takeaways_per_game': g('takeaways_per_game'), 'pen_per_game': g('pen_per_game'),
            'sack_rate_off': g('sack_rate_off'), 'sack_rate_def': g('sack_rate_def'), 'blitz_rate_def': g('blitz_rate_def'),
            'pa_rate': g('pa_rate'), 'motion_rate': g('motion_rate'), 'pace': g('pace_secs_per_play'), 'pass_oe': g('pass_oe'),
            'fourth_go_rate': g('fourth_go_rate'),
            'rest': g('rest'), 'travel_miles': g('travel_miles'), 'tz_shift': g('tz_shift'), 'prev_ot': g('prev_ot'),
            'consec_road_prior': g('consec_road_prior'), 'coach': row.get(f'{s}_coach'), 'coach_tenure': g('coach_tenure'),
            'inj_available': g('inj_available'), 'inj_starters_out': g('inj_starters_out'), 'inj_starters_q': g('inj_starters_q'),
            'inj_names_out': row.get(f'{s}_inj_names_out') if isinstance(row.get(f'{s}_inj_names_out'), str) else None,
            'av_off_out': g('av_off_out'), 'av_def_out': g('av_def_out'), 'av_off_q': g('av_off_q'), 'av_def_q': g('av_def_q'),
            'av_names': row.get(f'{s}_av_names') if isinstance(row.get(f'{s}_av_names'), str) and row.get(f'{s}_av_names') else None}

# depth-chart cross-check for QB (flag disagreement between nflverse projected starter and latest depth chart)
d26 = pd.read_csv('/tmp/nfl/data/depth_2026.csv'); d26 = d26[d26.dt==d26.dt.max()]
dc_qb = d26[(d26.pos_abb=='QB')&(d26.pos_rank==1)].set_index('team').player_name.to_dict()

games_out = []
for idx, row in wk.iterrows():
    xr = Xwk.loc[[idx]]
    p_early = float(early_model.predict_proba(xr[early_cols].fillna(early_med))[:,1][0])
    c_early, b0 = contributions(early_model, early_med, early_cols, xr.iloc[0])
    missing = [LABELS.get(c,c) for c in early_cols if pd.isna(xr.iloc[0][c])]
    rec = {'game_id': row.game_id, 'season': int(row.season), 'week': int(row.week), 'gameday': str(row.gameday.date()),
           'gametime_et': row.gametime, 'home': row.home_team, 'away': row.away_team, 'neutral': int(row.neutral),
           'stadium': row.stadium, 'roof': row.roof if isinstance(row.roof,str) else None, 'surface': row.surface if isinstance(row.surface,str) else None,
           'div_game': int(row.div_game) if pd.notna(row.div_game) else 0,
           'early': {'p_home': round(p_early,4), 'p_away': round(1-p_early,4),
                     'pick': row.home_team if p_early>=0.5 else row.away_team,
                     'predicted_at': now, 'model_version': MODEL_VERSION, 'input_hash': input_hash,
                     'cutoff': 'Tuesday of game week; uses games through '+meta['pbp_last_game_date'],
                     'factors': sorted([{'feature': LABELS.get(k,k), 'key': k, 'logit': round(float(v),4)} for k,v in c_early.items()],
                                       key=lambda d: -abs(d['logit'])),
                     'intercept_logit': round(b0,4), 'missing_inputs': missing},
           'updated': None, 'weather': wx_note.get(row.game_id), 'score': reconcile(score_pred(idx), row.home_team if p_early>=0.5 else row.away_team, row.home_team, row.away_team),
           'market': {'home_moneyline': None if pd.isna(row.home_moneyline) else float(row.home_moneyline),
                      'away_moneyline': None if pd.isna(row.away_moneyline) else float(row.away_moneyline),
                      'p_home_devig': None if pd.isna(row.home_moneyline) else round(devig(row.home_moneyline,row.away_moneyline),4),
                      'spread_line': None if pd.isna(row.spread_line) else float(row.spread_line),
                      'source': 'nflverse games.csv moneyline at data pull (benchmark only, not a model input)'},
           'home_profile': team_profile(row,'h'), 'away_profile': team_profile(row,'a'),
           'qb_flags': []}
    for s, team in (('h',row.home_team),('a',row.away_team)):
        proj = row.get(f'{s}_qb_name'); dc = dc_qb.get(team)
        if isinstance(proj,str) and dc and proj != dc:
            rec['qb_flags'].append(f'{team}: nflverse projected starter {proj} differs from latest depth chart ({dc}); treat QB as uncertain')
        if not isinstance(proj,str):
            rec['qb_flags'].append(f'{team}: expected starter unknown; QB features imputed with training medians')
    if injuries_available:
        p_u = float(upd_model.predict_proba(xr[upd_cols].fillna(upd_med))[:,1][0])
        c_u, b0u = contributions(upd_model, upd_med, upd_cols, xr.iloc[0])
        rec['updated'] = {'p_home': round(p_u,4), 'p_away': round(1-p_u,4), 'pick': row.home_team if p_u>=0.5 else row.away_team,
                          'predicted_at': now, 'model_version': MODEL_VERSION+'-updated', 'input_hash': input_hash, 'intercept_logit': round(b0u,4),
                          'factors': sorted([{'feature': LABELS.get(k,k), 'key': k, 'logit': round(float(v),4)} for k,v in c_u.items()], key=lambda d:-abs(d['logit']))}
    # keep the first early prediction ever made for this game; log every run
    prior = LOG.get(row.game_id, {})
    if prior.get('early'):
        rec['early_first'] = prior['early']
    else:
        rec['early_first'] = rec['early']
    runs = prior.get('runs', [])
    runs.append({'at': now, 'type': 'updated' if injuries_available else 'early', 'p_home': rec['updated']['p_home'] if rec['updated'] else rec['early']['p_home'],
                 'model_version': MODEL_VERSION, 'weather_status': (rec['weather'] or {}).get('status')})
    LOG[row.game_id] = {'early': rec['early_first'], 'runs': runs}
    rec['latest'] = rec['updated'] or rec['early']
    rec['score'] = reconcile(rec['score'], rec['latest']['pick'], row.home_team, row.away_team)
    rec['latest_type'] = 'updated' if rec['updated'] else 'early'
    rec['runs'] = runs
    games_out.append(rec)
json.dump(LOG, open(LOG_PATH,'w'), indent=0, default=float)

# ---------------------------------------------------------------- history for dashboard: walk-forward preds of final early model
hist = P_early.merge(F[['game_id','gameday','home_score','away_score']], on='game_id').merge(PS[['game_id','pred_margin','pred_total']], on='game_id', how='left')
hist['mkt'] = [devig(h,a) if pd.notna(h) and pd.notna(a) else None for h,a in zip(hist.home_moneyline,hist.away_moneyline)]
hist_out = [{'game_id':r.game_id,'season':int(r.season),'week':int(r.week),'gameday':str(r.gameday.date()),'home':r.home_team,'away':r.away_team,
             'p_home':round(float(r.p_home),4),'mkt':None if r.mkt is None or pd.isna(r.mkt) else round(float(r.mkt),4),
             'home_score':int(r.home_score),'away_score':int(r.away_score),'home_win':float(r.home_win),
             'pred_home': None if pd.isna(r.pred_margin) else int(round((r.pred_total+r.pred_margin)/2)), 'pred_away': None if pd.isna(r.pred_margin) else int(round((r.pred_total-r.pred_margin)/2))} for r in hist.itertuples()]

# confidence tiers (from walk-forward history of the final early model)
hist_c = P_early.copy(); hist_c['conf'] = np.maximum(hist_c.p_home, 1-hist_c.p_home); hist_c['hit'] = ((hist_c.p_home>0.5)==(hist_c.home_win==1)) & hist_c.home_win.isin([0,1])
TIERS = [('Lean',0.5,0.58),('Moderate',0.58,0.68),('Strong',0.68,1.01)]
tiers_out = []
for name,lo,hi in TIERS:
    t = hist_c[(hist_c.conf>=lo)&(hist_c.conf<hi)&hist_c.home_win.isin([0,1])]
    tiers_out.append({'tier':name,'lo':lo,'hi':min(hi,1.0),'n':int(len(t)),'hit_rate': float(t.hit.mean()) if len(t) else None})

coef_table = {'early': dict(zip(early_cols, [round(float(c),4) for c in early_model.named_steps['logisticregression'].coef_[0]])),
              'updated': dict(zip(upd_cols, [round(float(c),4) for c in upd_model.named_steps['logisticregression'].coef_[0]]))}

bundle = {
 'generated_at': now, 'model_version': MODEL_VERSION, 'season': cur_season, 'week': cur_week,
 'data': {'pbp_through': meta['pbp_last_game_date'], 'features_built_at': meta['built_at'],
          'injury_report_weeks_2026': meta['injury_weeks_2026'], 'injuries_available_for_week': injuries_available,
          'depth_chart_latest': meta['depth_chart_2026_latest'], 'ftn_latest_pull': meta['ftn_2026_latest_pull'],
          'sources': ['nflverse play-by-play (nflverse-data releases)', 'nflverse schedules/games.csv (nfldata)',
                      'nflverse injuries (official NFL reports)', 'nflverse depth charts', 'FTN charting via nflverse (2022+)',
                      'Open-Meteo hourly forecasts (open-meteo.com)']},
 'definitions': {
   'target': 'P(home team wins). Ties count 0.5 in Brier score; excluded from winner accuracy; a pick is only correct if that team wins outright.',
   'early_cutoff': 'Tuesday of game week. Uses only games completed before the week starts plus the kickoff-hour weather forecast available at run time. No injury reports.',
   'updated_cutoff': 'Re-run daily. Once the official injury report for the week exists (Wed-Fri practice reports, Friday game status), the updated model adds injury features; forecasts and expected starters are refreshed on every run. The first early prediction is kept unchanged alongside.',
   'scores': 'Projected score = two ridge regressions (alpha 100) fit on the same validated inputs: expected margin (home minus away) and expected total points (team offense/defense sums, pace, roof, wind, cold). Home = (total + margin)/2, away = (total - margin)/2, rounded. The winner pick always comes from the win-probability model; in near coin-flip games the rounded score can tie or point the other way, in which case the picked team is shown with a one-point lead and the score is marked adjusted.',
   'training': 'Logistic regression (L2, C=0.05) on home-minus-away feature differences. Walk-forward: each week predicted by a model fit only on games before that week; evaluation window 2022 wk1 onward. Model selection used 2022-2024; 2025-2026 reported as untouched holdout.',
   'not_in_model': ['Precipitation (no historical data to learn from; shown per game for context). Wind and cold ARE model inputs, learned from actual kickoff conditions and fed by the Open-Meteo forecast',
                    'Crowd/attendance/home-fan share (no verified feed)', 'Playoff stakes / announced resting of starters (no verified feed)',
                    'Snap-weighted player availability (sum of season snap share of players out/doubtful/questionable, by position group): tested against simple starter counts; starter counts scored better on walk-forward Brier (0.2197 vs 0.2206 on 2022-24, 0.2260 vs 0.2272 on 2025-26), so the updated model keeps the counts and the snap-weighted absences are shown on each game for context',
                    'Non-QB player production stats (receiving, rushing, pass-rush): already captured by team efficiency metrics; not added separately',
                    'Rest, travel, coaching tenure, rivalry: tested, did not improve walk-forward Brier, excluded from final model'],
 },
 'score_metrics': score_metrics,
 'metrics': {'early': res_early, 'updated': res_upd, 'baseline': res_base,
             'holdout_2025_on': {}},
 'ablation': ablation, 'coefficients': coef_table, 'feature_labels': LABELS, 'confidence_tiers': tiers_out,
 'weather_meta': {'fetched_at': (json.load(open(f'{OUT}/forecasts.json'))['fetched_at'] if FC else None),
                  'sources': 'Open-Meteo hourly forecasts at the venue; kickoff-hour values averaged over the first 3 hours.',
                  'model_note': 'Weather weights were learned from actual kickoff conditions in 2020-2026 games (as the proxy for a forecast). Walk-forward test: adding weather changed Brier by +0.0008 and accuracy by 0.0, i.e. within noise. Its weights are small and data-derived.'},
 'games': games_out, 'history': hist_out,
}
for name, P in (('early',P_early),('updated',P_upd),('baseline',P_base)):
    h = P[P.season>=2025]; h = h.copy(); h['mkt'] = [devig(a,b) if pd.notna(a) and pd.notna(b) else np.nan for a,b in zip(h.home_moneyline,h.away_moneyline)]
    d = h[h.home_win.isin([0,1])]
    bundle['metrics']['holdout_2025_on'][name] = {'n':int(len(h)),'accuracy':float(((d.p_home>0.5)==(d.home_win==1)).mean()),'brier':brier(h.p_home,h.home_win),
        'market_accuracy': float(((d.mkt>0.5)==(d.home_win==1))[d.mkt.notna()].mean()), 'market_brier': brier(h[h.mkt.notna()].mkt,h[h.mkt.notna()].home_win),
        'home_accuracy': float((d.home_win==1).mean())}

json.dump(bundle, open(f'{OUT}/bundle.json','w'), indent=0, default=float)
# immutable snapshot
os.makedirs(f'{OUT}/snapshots', exist_ok=True)
snap = {'generated_at': now, 'model_version': MODEL_VERSION, 'season': cur_season, 'week': cur_week, 'input_hash': input_hash,
        'inputs': Xwk[upd_cols].round(6).assign(game_id=wk.game_id.values).to_dict(orient='records'),
        'predictions': [{'game_id':g['game_id'],'early':g['early']['p_home'],'updated': g['updated']['p_home'] if g['updated'] else None} for g in games_out],
        'coefficients': coef_table}
json.dump(snap, open(f'{OUT}/snapshots/{cur_season}_wk{cur_week:02d}_{MODEL_VERSION}_{now[:10]}.json','w'), indent=1, default=float)
print('week', cur_week, 'games', len(games_out), 'injuries_available', injuries_available, 'hash', input_hash)
for g in games_out: print(f"{g['away']}@{g['home']}: early home {g['early']['p_home']:.3f} pick {g['early']['pick']} | mkt {g['market']['p_home_devig']} | flags {g['qb_flags']}")
print(json.dumps(bundle['metrics']['holdout_2025_on'], indent=1))
print(coef_table['early'])
