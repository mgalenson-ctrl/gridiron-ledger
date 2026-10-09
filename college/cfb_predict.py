"""Fit final models on all played games, predict the current week, and write the app data bundle."""
import numpy as np, pandas as pd, json, hashlib, os
from datetime import datetime, timezone
from cfb_train import *
OUT = '/home/claude/cfbpredict/out'

MODEL_VERSION = 'cfb-v2.0.0'
# Stats-only model (no betting data), kept for the hunch-vs-model experiment
EARLY_GROUPS = ['baseline','efficiency','elo','form_l3','conf_game']
UPDATED_GROUPS = EARLY_GROUPS
# Final model (owner's choice, Oct 2026): the same stats plus the betting line. Chosen on 2024 Brier; 2025-26 reported as holdout.
CRUNCH_GROUPS = EARLY_GROUPS + ['market']
C_FINAL = 0.02
meta = json.load(open(f'{OUT}/data_meta.json'))

LABELS = {
 'd_elo':'Pregame Elo rating (CFBD)','d_l3_off_epa':'Last 3 games: offensive EPA/play','d_l3_def_epa':'Last 3 games: EPA/play allowed','d_l3_ppd':'Last 3 games: points per drive','conf_game':'Conference game',
 'd_adj_off_off_sr':'Adj. offensive success rate','d_adj_def_off_sr':'Adj. defensive success rate','d_adj_off_pass_epa':'Adj. passing EPA/dropback','d_adj_def_pass_epa':'Adj. pass defense','d_adj_off_rush_epa':'Adj. rushing EPA/play','d_adj_def_rush_epa':'Adj. run defense','d_adj_off_ppd':'Adj. points per drive','d_adj_def_ppd':'Adj. points per drive allowed','d_adj_off_explosive_rate':'Adj. explosive-play rate','d_adj_def_explosive_rate':'Adj. explosive plays allowed','d_qb_cmp':'QB completion %','d_fbs':'FBS vs non-FBS',
 'home_adv':'Home-field advantage','d_adj_off_off_epa':'Opponent-adjusted offensive EPA/play','d_adj_def_off_epa':'Opponent-adjusted defensive EPA/play',
 'd_qb_epa':'QB EPA per dropback (trailing ~400 dropbacks)','d_qb_cpoe':'QB completion % over expected','d_qb_int_rate':'QB interception rate',
 'd_qb_sack_rate':'QB sack rate','d_qb_scr_epa_per_db':'QB scramble value','d_qb_change':'QB change from last game','d_qb_db_hist':'QB experience (dropbacks in window)',
 'd_fg_rate':'Field-goal accuracy','d_fg_long_rate':'Field-goal accuracy 40+ yds','d_st_epa_per_game':'Special-teams EPA/game',
 'd_to_per_game':'Giveaways per game','d_takeaways_per_game':'Takeaways per game','d_pen_per_game':'Penalties per game',
 'd_pp_vs_rush':'Pass protection vs pass rush','d_rush_vs_rundef':'Rushing offense vs run defense','d_pass_vs_passdef':'Passing offense vs pass defense',
 'd_explosive_edge':'Explosive plays gained vs allowed',
 'wx_wind':'Wind at kickoff','wx_cold':'Cold at kickoff (below 45°F)','wx_wind_x_passlean':'Wind × pass-leaning offense','wx_cold_x_awaydome':'Cold vs visiting dome team','wx_wind_x_fg':'Wind × long field-goal edge',
 'mkt_logit':'Betting market win probability (from the spread)','mkt_spread':'Betting market point spread',
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
cur_week = int(F[(F.season==cur_season) & (~F.played) & (F.season_type=='regular')].week.min())
FC = {}
if os.path.exists(f'{OUT}/forecasts.json'):
    fj = json.load(open(f'{OUT}/forecasts.json'))
    if fj['season']==cur_season and fj['week']==cur_week: FC = fj['forecasts']
wx_note = {}
for idx, row in F[(F.season==cur_season)&(F.week==cur_week)].iterrows():
    roof = row.roof if isinstance(row.roof,str) else None
    if roof in ('dome','closed'):
        wx_note[row.game_id] = {'status':'roofed','summary':'Roofed stadium; weather not a factor.'}; continue
    fc = FC.get(str(row.game_id)) or FC.get(row.game_id)
    if fc and fc.get('status')=='ok':
        F.loc[idx,'wx_wind'] = fc['wind_mph']; F.loc[idx,'wx_temp'] = fc['temp_f']; F.loc[idx,'wx_source'] = 'forecast'
        wx_note[row.game_id] = {'status':'forecast', **{k:fc.get(k) for k in ('temp_f','wind_mph','gust_mph','pop_pct','valid_for','forecast_created','fetched_at','source')},
                                'retractable': roof is None}
    else:
        F.loc[idx,'wx_wind'] = np.nan; F.loc[idx,'wx_temp'] = np.nan; F.loc[idx,'wx_source'] = 'missing'
        wx_note[row.game_id] = {'status': (fc or {}).get('status','unavailable'), 'summary':'Forecast unavailable; weather inputs imputed with training medians.'}
# ---------------------------------------------------------------- live lines (ESPN scoreboard): current spread, opening spread, moneyline
LIVE = {}; live_meta = {'fetched_at': None, 'n_with_odds': 0, 'errors': []}
if os.path.exists(f'{OUT}/live.json'):
    lj = json.load(open(f'{OUT}/live.json'))
    if (datetime.now(timezone.utc) - datetime.fromisoformat(lj['fetched_at'])).total_seconds()/3600 <= 3:
        LIVE = lj.get('games', {}); live_meta = {k: lj.get(k) for k in ('fetched_at','n_with_odds','errors','source')}
LINE_HIST = json.load(open(f'{OUT}/line_history.json')) if os.path.exists(f'{OUT}/line_history.json') else {}
line_src = {}
for idx, row in F[(F.season==cur_season)&(F.week==cur_week)].iterrows():
    o = (LIVE.get(str(row.game_id)) or {}).get('odds')
    if o and o.get('spread_home') is not None:
        F.loc[idx,'spread'] = o['spread_home']
        if o.get('total') is not None: F.loc[idx,'over_under'] = o['total']
        line_src[row.game_id] = f"ESPN ({o.get('provider') or 'consensus'}) at {live_meta['fetched_at']}"
    elif pd.notna(row.spread):
        line_src[row.game_id] = 'CFBD via cfbfastR-data'
X_all = build_X(F)

early_model, early_med, early_cols = fit_final(EARLY_GROUPS)
crunch_model, crunch_med, crunch_cols = fit_final(CRUNCH_GROUPS)
upd_model, upd_med, upd_cols = fit_final(UPDATED_GROUPS)

# ---------------------------------------------------------------- score models (ridge; margin on EARLY cols, total on team sums)
SCORE_ALPHA = 100.0
T_all = build_T(F)
margin_model, margin_med = fit_reg(X_all.loc[F.played, crunch_cols], F.result[F.played], SCORE_ALPHA)
total_model, total_med = fit_reg(T_all[F.played], F.total[F.played], SCORE_ALPHA)
PS = walk_forward_scores(crunch_cols, SCORE_ALPHA)
PSd = PS.dropna(subset=['result','total','spread','over_under'])
def mae(a,b): return float(np.mean(np.abs(a-b)))
score_metrics = {'n': int(len(PSd)), 'margin_mae': mae(PSd.pred_margin, PSd.result), 'spread_mae': mae(-PSd.spread, PSd.result),
                 'total_mae': mae(PSd.pred_total, PSd.total), 'market_total_mae': mae(PSd.over_under, PSd.total),
                 'margin_within_7': float((np.abs(PSd.pred_margin-PSd.result)<=7).mean()),
                 'holdout_2025_on': {'margin_mae': mae(PSd[PSd.season>=2025].pred_margin, PSd[PSd.season>=2025].result), 'spread_mae': mae(-PSd[PSd.season>=2025].spread, PSd[PSd.season>=2025].result),
                                     'total_mae': mae(PSd[PSd.season>=2025].pred_total, PSd[PSd.season>=2025].total), 'market_total_mae': mae(PSd[PSd.season>=2025].over_under, PSd[PSd.season>=2025].total)}}
def score_pred(idx):
    m = float(margin_model.predict(X_all.loc[[idx], crunch_cols].fillna(margin_med))[0]); t = float(total_model.predict(T_all.loc[[idx]].fillna(total_med))[0])
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
P_crunch = walk_forward(CRUNCH_GROUPS, C=C_FINAL); res_crunch, P_crunch = evaluate(P_crunch)
ablation = json.load(open(f'{OUT}/ablation_results.json'))

# ---------------------------------------------------------------- current week
wk = F[(F.season==cur_season)&(F.week==cur_week)&(F.season_type=='regular')].copy()
# prior predictions log (keeps the first EARLY prediction per game across reruns)
LOG_PATH = f'{OUT}/predictions_log.json'
LOG = json.load(open(LOG_PATH)) if os.path.exists(LOG_PATH) else {}
Xwk = X_all.loc[wk.index]
input_hash = hashlib.sha256(pd.util.hash_pandas_object(Xwk[upd_cols].round(6)).values.tobytes()).hexdigest()[:16]
now = datetime.now(timezone.utc).isoformat(timespec='seconds')
injuries_available = False

def team_profile(row, s):
    g = lambda k: (None if pd.isna(row.get(f'{s}_{k}')) else float(row.get(f'{s}_{k}')))
    team = row[f'{"home" if s=="h" else "away"}_team']
    return {'team': team, 'conf': row[f'{"home" if s=="h" else "away"}_conf'], 'div': row[f'{"home" if s=="h" else "away"}_div'],
            'games_played': g('games_played'), 'adj_off_epa': g('adj_off_off_epa'), 'adj_def_epa': g('adj_def_off_epa'),
            'adj_off_sr': g('adj_off_off_sr'), 'adj_pass_epa': g('adj_off_pass_epa'), 'adj_rush_epa': g('adj_off_rush_epa'),
            'adj_def_pass_epa': g('adj_def_pass_epa'), 'adj_def_rush_epa': g('adj_def_rush_epa'), 'adj_ppd': g('adj_off_ppd'), 'adj_def_ppd': g('adj_def_ppd'),
            'adj_explosive': g('adj_off_explosive_rate'), 'adj_def_explosive': g('adj_def_explosive_rate'),
            'l3_off_epa': g('l3_off_epa'), 'l3_def_epa': g('l3_def_epa'), 'l3_ppd': g('l3_ppd'), 'elo': g('elo'),
            'qb_name': row.get(f'{s}_qb_name') if isinstance(row.get(f'{s}_qb_name'), str) else None,
            'qb_epa': g('qb_epa'), 'qb_cmp': g('qb_cmp'), 'qb_int_rate': g('qb_int_rate'), 'qb_sack_rate': g('qb_sack_rate'), 'qb_db_hist': g('qb_db_hist'),
            'third_rate': g('third_rate'), 'rz_td_rate': g('rz_td_rate'), 'fg_rate': g('fg_rate'), 'to_per_game': g('to_per_game'), 'takeaways_per_game': g('takeaways_per_game'),
            'pen_per_game': g('pen_per_game'), 'sack_rate_off': g('sack_rate_off'), 'sack_rate_def': g('sack_rate_def'), 'pass_rate': g('pass_rate'), 'plays_per_game': g('plays_per_game'),
            'fourth_go_rate': g('fourth_go_rate'), 'rest': g('rest'), 'travel_miles': g('travel_miles'), 'fbs': g('fbs'), 'inj_available': 0}

# depth-chart cross-check for QB (flag disagreement between nflverse projected starter and latest depth chart)
dc_qb = {}

def line_movement(row):
    rows = LINE_HIST.get(str(row.game_id)) or []
    o = (LIVE.get(str(row.game_id)) or {}).get('odds') or {}
    if not rows and not o: return None
    first = rows[0] if rows else {}
    open_sp = o.get('spread_home_open') if o.get('spread_home_open') is not None else first.get('spread_home')
    cur_sp = o.get('spread_home', rows[-1].get('spread_home') if rows else None)
    p = lambda sp: None if sp is None else round(float(spread_prob(sp)),4)
    flags = []
    if open_sp is not None and cur_sp is not None:
        mv = cur_sp - open_sp
        if abs(mv) >= 2: flags.append(f"Spread moved {abs(mv):g} pts toward {row.home_team if mv<0 else row.away_team}")
        if open_sp != 0 and cur_sp != 0 and (open_sp < 0) != (cur_sp < 0): flags.append('Favorite flipped')
    return {'open_spread_home': open_sp, 'cur_spread_home': cur_sp, 'open_p_home': p(open_sp), 'cur_p_home': p(cur_sp),
            'open_ml': [o.get('ml_home_open'), o.get('ml_away_open')], 'cur_ml': [o.get('ml_home'), o.get('ml_away')],
            'total_open': o.get('total_open'), 'total': o.get('total'), 'provider': o.get('provider'), 'flags': flags,
            'moves': [r for r in rows if r.get('t')][-12:],
            'note': 'Movement is open-to-current from the sportsbook ESPN shows. It shows where the line went, not who bet; betting-split ("sharp money") data is not available from a free source.'}

games_out = []
for idx, row in wk.iterrows():
    xr = Xwk.loc[[idx]]
    p_early = float(early_model.predict_proba(xr[early_cols].fillna(early_med))[:,1][0])
    c_early, b0 = contributions(early_model, early_med, early_cols, xr.iloc[0])
    missing = [LABELS.get(c,c) for c in early_cols if pd.isna(xr.iloc[0][c])]
    et = pd.Timestamp(row.start_utc).tz_convert('America/New_York')
    rec = {'game_id': int(row.game_id), 'season': int(row.season), 'week': int(row.week), 'gameday': str(et.date()),
           'gametime_et': et.strftime('%H:%M'), 'home': row.home_team, 'away': row.away_team, 'neutral': int(row.neutral),
           'stadium': row.venue if isinstance(row.venue,str) else None, 'roof': row.roof if isinstance(row.roof,str) else None, 'surface': None,
           'div_game': int(row.conf_game), 'home_conf': row.home_conf, 'away_conf': row.away_conf, 'home_div': row.home_div, 'away_div': row.away_div,
           'elo': {'home': None if pd.isna(row.home_elo) else float(row.home_elo), 'away': None if pd.isna(row.away_elo) else float(row.away_elo), 'p_home': None if pd.isna(row.home_elo) else round(float(elo_prob(row.home_elo,row.away_elo)),4)},
           'early': {'p_home': round(p_early,4), 'p_away': round(1-p_early,4),
                     'pick': row.home_team if p_early>=0.5 else row.away_team,
                     'predicted_at': now, 'model_version': MODEL_VERSION, 'input_hash': input_hash,
                     'cutoff': 'Tuesday of game week; uses games through '+meta['pbp_last_game_date'],
                     'factors': sorted([{'feature': LABELS.get(k,k), 'key': k, 'logit': round(float(v),4)} for k,v in c_early.items()],
                                       key=lambda d: -abs(d['logit'])),
                     'intercept_logit': round(b0,4), 'missing_inputs': missing},
           'updated': None, 'weather': wx_note.get(row.game_id), 'score': reconcile(score_pred(idx), row.home_team if p_early>=0.5 else row.away_team, row.home_team, row.away_team),
           'market': {'p_home_devig': None if pd.isna(row.spread) else round(float(spread_prob(row.spread)),4),
                      'spread_line': None if pd.isna(row.spread) else float(-row.spread),
                      'total_line': None if pd.isna(row.over_under) else float(row.over_under),
                      'source': line_src.get(row.game_id, 'unavailable')},
           'line': line_movement(row), 'injury_news': None,
           'home_profile': team_profile(row,'h'), 'away_profile': team_profile(row,'a'),
           'qb_flags': []}
    for s, team in (('h',row.home_team),('a',row.away_team)):
        if not isinstance(row.get(f'{s}_qb_name'),str):
            rec['qb_flags'].append(f'{team}: no play-by-play on file this season; team rated as a pooled non-FBS opponent')
    if injuries_available:
        p_u = float(upd_model.predict_proba(xr[upd_cols].fillna(upd_med))[:,1][0])
        c_u, b0u = contributions(upd_model, upd_med, upd_cols, xr.iloc[0])
        rec['updated'] = {'p_home': round(p_u,4), 'p_away': round(1-p_u,4), 'pick': row.home_team if p_u>=0.5 else row.away_team,
                          'predicted_at': now, 'model_version': MODEL_VERSION+'-updated', 'input_hash': input_hash, 'intercept_logit': round(b0u,4),
                          'factors': sorted([{'feature': LABELS.get(k,k), 'key': k, 'logit': round(float(v),4)} for k,v in c_u.items()], key=lambda d:-abs(d['logit']))}
    # keep the first early prediction ever made for this game; log every run
    prior = LOG.get(str(row.game_id), {})
    if prior.get('early'):
        rec['early_first'] = prior['early']
    else:
        rec['early_first'] = rec['early']
    runs = prior.get('runs', [])
    runs.append({'at': now, 'type': 'updated' if injuries_available else 'early', 'p_home': rec['updated']['p_home'] if rec['updated'] else rec['early']['p_home'],
                 'model_version': MODEL_VERSION, 'weather_status': (rec['weather'] or {}).get('status')})
    LOG[str(row.game_id)] = {'early': rec['early_first'], 'runs': runs}
    rec['stats'] = rec['updated'] or rec['early']; rec['stats_type'] = 'updated' if rec['updated'] else 'early'
    if pd.notna(row.spread):
        p_c = float(crunch_model.predict_proba(xr[crunch_cols].fillna(crunch_med))[:,1][0])
        c_c, b0c = contributions(crunch_model, crunch_med, crunch_cols, xr.iloc[0])
        rec['crunch'] = {'p_home': round(p_c,4), 'p_away': round(1-p_c,4), 'pick': row.home_team if p_c>=0.5 else row.away_team,
                         'predicted_at': now, 'model_version': MODEL_VERSION+'-crunch', 'input_hash': input_hash, 'intercept_logit': round(b0c,4),
                         'factors': sorted([{'feature': LABELS.get(k,k), 'key': k, 'logit': round(float(v),4)} for k,v in c_c.items()], key=lambda d:-abs(d['logit']))}
        rec['latest'] = rec['crunch']; rec['latest_type'] = 'crunch'
    else:
        rec['crunch'] = None; rec['latest'] = rec['stats']; rec['latest_type'] = rec['stats_type']
    rec['score'] = reconcile(rec['score'], rec['latest']['pick'], row.home_team, row.away_team)
    runs[-1].update({'p_crunch': rec['crunch']['p_home'] if rec['crunch'] else None, 'p_stats': rec['stats']['p_home'], 'spread_line': rec['market']['spread_line']})
    rec['runs'] = runs
    games_out.append(rec)
json.dump(LOG, open(LOG_PATH,'w'), indent=0, default=float)

# ---------------------------------------------------------------- history for dashboard: walk-forward preds of final early model
hist = P_crunch.merge(F[['game_id','gameday','home_score','away_score']], on='game_id').merge(PS[['game_id','pred_margin','pred_total']], on='game_id', how='left')
hist['mkt'] = [spread_prob(x) if pd.notna(x) else None for x in hist.spread]
hist = hist[hist.season>=2025]  # keep the app bundle small: last two seasons of results
hist_out = [{'game_id':int(r.game_id),'season':int(r.season),'week':int(r.week),'gameday':str(pd.Timestamp(r.gameday).date()),'home':r.home_team,'away':r.away_team,
             'p_home':round(float(r.p_home),4),'mkt':None if r.mkt is None or pd.isna(r.mkt) else round(float(r.mkt),4),
             'home_score':int(r.home_score),'away_score':int(r.away_score),'home_win':float(r.home_win),
             'pred_home': None if pd.isna(r.pred_margin) else int(round((r.pred_total+r.pred_margin)/2)), 'pred_away': None if pd.isna(r.pred_margin) else int(round((r.pred_total-r.pred_margin)/2))} for r in hist.itertuples()]

# confidence tiers (from walk-forward history of the final early model)
hist_c = P_crunch.copy(); hist_c['conf'] = np.maximum(hist_c.p_home, 1-hist_c.p_home); hist_c['hit'] = ((hist_c.p_home>0.5)==(hist_c.home_win==1)) & hist_c.home_win.isin([0,1])
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
          'injury_report_weeks_2026': [], 'injuries_available_for_week': False, 'depth_chart_latest': None, 'ftn_latest_pull': None,
          'sources': ['cfbfastR-data (sportsdataverse): ESPN play-by-play with EPA, CFBD schedules, team/venue info, pregame Elo', 'ESPN public API: current betting lines (open and current), checked each run',
                      'Open-Meteo hourly forecasts (open-meteo.com) for display']},
 'definitions': {
   'target': 'P(home team wins). Ties count 0.5 in Brier score; excluded from winner accuracy; a pick is only correct if that team wins outright.',
   'early_cutoff': 'Monday of game week. Uses only games completed in earlier weeks (last season as a down-weighted prior), the CFBD pregame Elo, and the expected starting QB (the passer with the most dropbacks in the team\'s last game). College has no official injury report feed, so there is no separate updated model; runs are refreshed daily for forecasts and newly completed games, and the first prediction is kept alongside.',
   'updated_cutoff': 'Not applicable for college (no official injury report feed).',
   'scores': 'Projected score = two ridge regressions (alpha 100) fit on the same validated inputs: expected margin (home minus away) and expected total points (team offense/defense sums, pace, pass rate, dome, FBS level). Home = (total + margin)/2, away = (total - margin)/2, rounded. The winner pick always comes from the win-probability model; in near coin-flip games the rounded score can tie or point the other way, in which case the picked team is shown with a one-point lead and the score is marked adjusted.',
   'crunch': 'Final pick = the stats model plus the betting line (current point spread and its implied win probability), at the owner\'s request. Historical training uses CFBD consensus spreads; live picks use the current ESPN line. Walk-forward Brier score (lower is better), 2024 selection season: 0.1635 vs 0.1664 for the market spread alone; 2025-26 holdout: 0.1417 vs 0.1414, with 79.6% of winners picked. It tracks the market closely and is no longer an independent read.',
   'stats_only': 'Stats-only pick = no betting data (efficiency, Elo, recent form, conference game). Shown alongside so the gut-vs-model experiment keeps a version that never sees the odds.',
   'training': 'Logistic regression (L2, C=0.02) on home-minus-away feature differences. Walk-forward: each week predicted by a model fit only on games before that week; evaluation window 2024 wk1 onward (training data starts 2023). Model selection used 2024 only; 2025-2026 reported as untouched holdout. Non-FBS opponents are pooled into one rating entity.',
   'not_in_model': ['Weather (no leakage-free historical forecast archive accessible; shown live for context only)',
                    'Crowd/attendance/home-fan share (no verified feed)', 'Playoff stakes / announced resting of starters (no verified feed)',
                    'Non-QB player production stats: already captured by team efficiency metrics'],
 },
 'score_metrics': score_metrics,
 'live': {**live_meta, 'line_source_counts': {k: sum(1 for v in line_src.values() if v.startswith(k)) for k in ('ESPN','CFBD')}},
 'metrics': {'crunch': res_crunch, 'early': res_early, 'updated': res_upd, 'baseline': res_base,
             'holdout_2025_on': {}},
 'ablation': ablation, 'coefficients': coef_table, 'feature_labels': LABELS, 'confidence_tiers': tiers_out,
 'weather_meta': {'fetched_at': (json.load(open(f'{OUT}/forecasts.json'))['fetched_at'] if FC else None),
                  'sources': 'Open-Meteo hourly forecasts at the venue; kickoff-hour values averaged over the first 3 hours.',
                  'model_note': 'College weather is display-only: the play-by-play feed carries no historical kickoff conditions, so no weather weight could be learned without inventing one.'},
 'games': games_out, 'history': hist_out,
}
for name, P in (('crunch',P_crunch),('early',P_early),('updated',P_upd),('baseline',P_base)):
    h = P[P.season>=2025]; h = h.copy(); h['mkt'] = spread_prob(h.spread)
    d = h[h.home_win.isin([0,1])]
    bundle['metrics']['holdout_2025_on'][name] = {'n':int(len(h)),'accuracy':float(((d.p_home>0.5)==(d.home_win==1)).mean()),'brier':brier(h.p_home,h.home_win),
        'market_accuracy': float(((d.mkt>0.5)==(d.home_win==1))[d.mkt.notna()].mean()), 'market_brier': brier(h[h.mkt.notna()].mkt,h[h.mkt.notna()].home_win),
        'elo_accuracy': float(((elo_prob(d.home_elo,d.away_elo)>0.5)==(d.home_win==1))[d.home_elo.notna()&d.away_elo.notna()].mean()),
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
