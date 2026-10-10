"""Fit final models on all played games, predict the current week, and write the app data bundle."""
import numpy as np, pandas as pd, json, hashlib, os
from datetime import datetime, timezone, timedelta
from train import *

MODEL_VERSION = 'v2.0.0'
# Stats-only model (no betting data): the "pure" pick, kept for the hunch-vs-model experiment
EARLY_GROUPS = ['baseline','qb','special_discipline','matchup','weather','ngs_off']
UPDATED_GROUPS = EARLY_GROUPS + ['injuries']
# Final model (owner's choice, Oct 2026): team efficiency + the betting line. Chosen on 2022-24 Brier; 2025-26 reported as holdout.
CRUNCH_GROUPS = ['baseline','market']
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
 'mkt_logit':'Betting market win probability (moneyline)','mkt_spread':'Betting market point spread',
 'd_ngs_ttt':'QB time to throw (Next Gen Stats)','d_ngs_ryoe':'Rush yards over expected per carry (Next Gen Stats)','d_ngs_sep':'Receiver separation (Next Gen Stats)','d_ngs_yacoe':'Yards after catch over expected (Next Gen Stats)',
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
# ---------------------------------------------------------------- live feeds (ESPN): current line and injury statuses
LIVE = {}
live_meta = {'fetched_at': None, 'n_with_odds': 0, 'errors': []}
if os.path.exists(f'{OUT}/live.json'):
    lj = json.load(open(f'{OUT}/live.json'))
    age_h = (datetime.now(timezone.utc) - datetime.fromisoformat(lj['fetched_at'])).total_seconds()/3600
    if age_h <= 3:
        LIVE = lj.get('games', {}); live_meta = {k: lj.get(k) for k in ('fetched_at','n_with_odds','errors','source')}
LINE_HIST = json.load(open(f'{OUT}/line_history.json')) if os.path.exists(f'{OUT}/line_history.json') else {}
INJ_HIST = json.load(open(f'{OUT}/injury_history.json')) if os.path.exists(f'{OUT}/injury_history.json') else {}
line_src = {}
for idx, row in F[(F.season==cur_season)&(F.week==cur_week)].iterrows():
    o = (LIVE.get(row.game_id) or {}).get('odds')
    if o and o.get('ml_home') is not None and o.get('ml_away') is not None:
        F.loc[idx,'home_moneyline'] = o['ml_home']; F.loc[idx,'away_moneyline'] = o['ml_away']
        if o.get('spread_home') is not None: F.loc[idx,'spread_line'] = -o['spread_home']
        if o.get('total') is not None: F.loc[idx,'total_line'] = o['total']
        line_src[row.game_id] = f"ESPN ({o.get('provider') or 'consensus'}) at {live_meta['fetched_at']}"
    elif pd.notna(row.home_moneyline):
        line_src[row.game_id] = 'nflverse games.csv at data pull'

# live injury statuses -> the same starter-count features the updated model was trained on.
# Official report rows (nflverse) are combined with ESPN statuses posted in the last 10 days; the more severe status wins.
_dep = pd.read_csv('/tmp/nfl/data/depth_2026.csv'); _dep = _dep[_dep.dt==_dep.dt.max()]
STARTERS = {t: grp for t, grp in _dep[_dep.pos_rank==1].groupby('team')}
GSIS2ESPN = dict(zip(_dep.gsis_id.astype(str), _dep.espn_id.astype(str)))
OL = {'T','G','C','OL','OT','OG','LT','RT','LG','RG'}
SEV = {'Out':3,'Injured Reserve':3,'Doubtful':2,'Questionable':1}
_inj_off = pd.read_csv(f'/tmp/nfl/data/injuries_{cur_season}.csv')
_inj_off = _inj_off[(_inj_off.week==cur_week)]
def live_injury_feats(team, side, gid):
    status = {}   # espn_id or name -> (severity, status, name, pos, source)
    for r in _inj_off[_inj_off.team==team].itertuples():
        st = r.report_status if isinstance(r.report_status,str) else None
        if st in SEV:
            k = GSIS2ESPN.get(str(r.gsis_id), r.full_name); status[k] = (SEV[st], st, r.full_name, r.position, 'official report')
    cutoff = datetime.now(timezone.utc) - timedelta(days=10)
    for p in ((LIVE.get(gid) or {}).get('injuries') or {}).get(side, []):
        st = p.get('status'); d = p.get('date')
        try: recent = d is None or datetime.fromisoformat(d.replace('Z','+00:00')) >= cutoff
        except ValueError: recent = True
        if st in SEV and recent:
            k = p.get('espn_id') or p.get('name')
            if k not in status or SEV[st] > status[k][0]: status[k] = (SEV[st], st, p.get('name'), p.get('pos'), 'ESPN')
    if not status and not len(_inj_off[_inj_off.team==team]) and gid not in LIVE: return None
    st_df = STARTERS.get(team); st_ids = set(st_df.espn_id.astype(str)) if st_df is not None else set()
    st_pos = dict(zip(st_df.espn_id.astype(str), st_df.pos_abb)) if st_df is not None else {}
    out_ = {k:v for k,v in status.items() if v[0]>=2}; q = {k:v for k,v in status.items() if v[0]==1}
    return {'inj_available': 1, 'inj_starters_out': sum(k in st_ids for k in out_), 'inj_starters_q': sum(k in st_ids for k in q),
            'inj_total_out': len(out_), 'inj_qb_out': int(any((v[3]=='QB') and k in st_ids for k,v in out_.items())),
            'inj_ol_out': sum(k in st_ids and st_pos.get(k) in OL for k in out_),
            'inj_names_out': '; '.join(f"{v[2]} ({v[3]}, {v[1].lower()}{', starter' if k in st_ids else ''})" for k,v in sorted(out_.items(), key=lambda kv:-(kv[0] in st_ids))[:8]),
            'inj_names_q': '; '.join(f"{v[2]} ({v[3]}{', starter' if k in st_ids else ''})" for k,v in q.items() if k in st_ids)}
inj_live_ok = {}
for idx, row in F[(F.season==cur_season)&(F.week==cur_week)].iterrows():
    ok = True
    for s_, team, side in (('h',row.home_team,'home'),('a',row.away_team,'away')):
        f_ = live_injury_feats(team, side, row.game_id)
        if f_ is None: ok = False; continue
        for k,v in f_.items(): F.loc[idx, f'{s_}_{k}'] = v
    inj_live_ok[row.game_id] = ok
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
PSd = PS.dropna(subset=['result','total'])
def mae(a,b): return float(np.mean(np.abs(a-b)))
score_metrics = {'n': int(len(PSd)), 'margin_mae': mae(PSd.pred_margin, PSd.result), 'spread_mae': mae(PSd.spread_line, PSd.result),
                 'total_mae': mae(PSd.pred_total, PSd.total), 'market_total_mae': mae(PSd.total_line, PSd.total),
                 'margin_within_7': float((np.abs(PSd.pred_margin-PSd.result)<=7).mean()),
                 'holdout_2025_on': {'margin_mae': mae(PSd[PSd.season>=2025].pred_margin, PSd[PSd.season>=2025].result), 'spread_mae': mae(PSd[PSd.season>=2025].spread_line, PSd[PSd.season>=2025].result),
                                     'total_mae': mae(PSd[PSd.season>=2025].pred_total, PSd[PSd.season>=2025].total), 'market_total_mae': mae(PSd[PSd.season>=2025].total_line, PSd[PSd.season>=2025].total)}}
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
wk = F[(F.season==cur_season)&(F.week==cur_week)].copy()
# prior predictions log (keeps the first EARLY prediction per game across reruns)
LOG_PATH = f'{OUT}/predictions_log.json'
LOG = json.load(open(LOG_PATH)) if os.path.exists(LOG_PATH) else {}
Xwk = X_all.loc[wk.index]
input_hash = hashlib.sha256(pd.util.hash_pandas_object(Xwk[upd_cols].round(6)).values.tobytes()).hexdigest()[:16]
now = datetime.now(timezone.utc).isoformat(timespec='seconds')
injuries_available = bool(all(inj_live_ok.get(g, False) for g in wk.game_id))

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
            'ngs_ttt': g('ngs_ttt'), 'ngs_ryoe': g('ngs_ryoe'), 'ngs_sep': g('ngs_sep'), 'ngs_yacoe': g('ngs_yacoe'),
            'inj_available': g('inj_available'), 'inj_starters_out': g('inj_starters_out'), 'inj_starters_q': g('inj_starters_q'),
            'inj_names_out': row.get(f'{s}_inj_names_out') if isinstance(row.get(f'{s}_inj_names_out'), str) else None,
            'av_off_out': g('av_off_out'), 'av_def_out': g('av_def_out'), 'av_off_q': g('av_off_q'), 'av_def_q': g('av_def_q'),
            'av_names': row.get(f'{s}_av_names') if isinstance(row.get(f'{s}_av_names'), str) and row.get(f'{s}_av_names') else None}

# depth-chart cross-check for QB (flag disagreement between nflverse projected starter and latest depth chart)
d26 = pd.read_csv('/tmp/nfl/data/depth_2026.csv'); d26 = d26[d26.dt==d26.dt.max()]
dc_qb = d26[(d26.pos_abb=='QB')&(d26.pos_rank==1)].set_index('team').player_name.to_dict()

def _ml_p(h, a):
    try: return round(devig(float(h), float(a)), 4)
    except Exception: return None
def line_movement(row):
    rows = LINE_HIST.get(row.game_id) or []
    o = (LIVE.get(row.game_id) or {}).get('odds') or {}
    if not rows and not o: return None
    first = rows[0] if rows else {}
    open_sp = o.get('spread_home_open') if o.get('spread_home_open') is not None else first.get('spread_home')
    open_mlh = o.get('ml_home_open') if o.get('ml_home_open') is not None else first.get('ml_home')
    open_mla = o.get('ml_away_open') if o.get('ml_away_open') is not None else first.get('ml_away')
    cur_sp = o.get('spread_home', rows[-1].get('spread_home') if rows else None)
    cur_mlh = o.get('ml_home', rows[-1].get('ml_home') if rows else None); cur_mla = o.get('ml_away', rows[-1].get('ml_away') if rows else None)
    p_open, p_cur = _ml_p(open_mlh, open_mla), _ml_p(cur_mlh, cur_mla)
    flags = []
    if open_sp is not None and cur_sp is not None:
        mv = cur_sp - open_sp   # negative = moved toward home
        if abs(mv) >= 1.5: flags.append(f"Spread moved {abs(mv):g} pts toward {row.home_team if mv<0 else row.away_team}")
        for key in (3, 7):
            if (abs(open_sp) < key) != (abs(cur_sp) < key) and abs(mv) >= 0.5: flags.append(f"Crossed the key number {key}")
        if open_sp != 0 and cur_sp != 0 and (open_sp < 0) != (cur_sp < 0): flags.append('Favorite flipped')
    if p_open is not None and p_cur is not None and abs(p_cur-p_open) >= 0.05:
        flags.append(f"Win odds shifted {abs(p_cur-p_open)*100:.0f} pts toward {row.home_team if p_cur>p_open else row.away_team}")
    return {'open_spread_home': open_sp, 'cur_spread_home': cur_sp, 'open_p_home': p_open, 'cur_p_home': p_cur,
            'open_ml': [open_mlh, open_mla], 'cur_ml': [cur_mlh, cur_mla], 'total_open': o.get('total_open'), 'total': o.get('total'),
            'provider': o.get('provider'), 'flags': flags, 'moves': [r for r in rows if r.get('t')][-12:],
            'note': 'Movement is open-to-current from the sportsbook ESPN shows. It shows where the line went, not who bet; betting-split ("sharp money") data is not available from a free source.'}
def injury_news(row):
    out = []
    cutoff = datetime.now(timezone.utc) - timedelta(hours=72)
    for team in (row.away_team, row.home_team):
        st_df = STARTERS.get(team); st_ids = set(st_df.espn_id.astype(str)) if st_df is not None else set()
        for key, ph in (INJ_HIST.get(team) or {}).items():
            for i, ch in enumerate(ph.get('changes', [])):
                seen = datetime.fromisoformat(ch['seen'])
                if seen < cutoff: continue
                prev = ph['changes'][i-1]['status'] if i else None
                if prev is None and ch['status'] == 'Cleared': continue
                out.append({'team': team, 'name': ph.get('name'), 'pos': ph.get('pos'), 'status': ch['status'], 'prev': prev, 'seen': ch['seen'], 'starter': key in st_ids})
    first_run = not any(len(ph.get('changes', [])) > 1 for t in (row.away_team, row.home_team) for ph in (INJ_HIST.get(t) or {}).values())
    out = [n for n in out if n['starter'] or n['status'] in ('Out','Doubtful')]
    out.sort(key=lambda n: n['seen'], reverse=True)
    return {'items': out[:12], 'baseline_only': first_run}

# ---------------------------------------------------------------- value check: odds -> break-even -> model probability -> estimated edge
# Cover probabilities use the model's projected margin plus its own out-of-sample errors (walk-forward residuals), so key numbers
# like 3 and 7 come from real outcomes rather than an assumed bell curve. Validated below on seasons the residuals never saw.
RESID = (PS.result - PS.pred_margin).dropna().values
def ml_be(ml): return None if ml is None else (100/(ml+100) if ml > 0 else -ml/(-ml+100))
def ml_profit(ml): return ml/100 if ml > 0 else 100/(-ml)
def p_to_ml(p):
    if p is None or p <= 0 or p >= 1: return None
    return int(round(-100*p/(1-p))) if p >= 0.5 else int(round(100*(1-p)/p))
def home_cover(margin, L, resid=None):
    m = np.round(margin + (RESID if resid is None else resid)); w = (m > L).mean(); push = (m == L).mean()
    return None if push >= 1 else float(w/(1-push))
def value_check(row, rec, odds):
    out = {'moneyline': None, 'spread': None, 'alt': []}
    L = rec['latest']; pick = L['pick']; home = row.home_team
    mlh, mla = rec['market'].get('home_moneyline'), rec['market'].get('away_moneyline')
    if mlh is not None and mla is not None:
        ml = mlh if pick == home else mla; p = L['p_home'] if pick == home else 1 - L['p_home']; be = ml_be(ml)
        out['moneyline'] = {'pick': pick, 'odds': ml, 'p': round(p, 4), 'break_even': round(be, 4), 'edge': round(p - be, 4),
                            'ev_per_100': round(100*(p*ml_profit(ml) - (1-p)), 1)}
    sl = rec['market'].get('spread_line'); mg = rec['score'].get('margin')
    if sl is not None and mg is not None and rec.get('ats'):
        side = rec['ats']['team']; is_home = side == home
        ph = home_cover(mg, sl); p = ph if is_home else (None if ph is None else 1 - ph)
        so = (odds or {}).get('sp_odds_home' if is_home else 'sp_odds_away'); assumed = so is None; so = -110.0 if so is None else so
        be = ml_be(so)
        if p is not None:
            out['spread'] = {'pick': side, 'line': rec['ats']['line'], 'odds': so, 'odds_assumed': assumed, 'p': round(p, 4), 'break_even': round(be, 4),
                             'edge': round(p - be, 4), 'ev_per_100': round(100*(p*ml_profit(so) - (1-p)), 1)}
        base = rec['ats']['line']   # side team's line (negative = laying points)
        for d in (-10, -7, -3.5, -3, 3, 3.5, 7, 10):
            tl = base + d                     # team gets tl points
            hl = -tl if is_home else tl       # equivalent home-margin threshold
            ph2 = home_cover(mg, hl); pc = ph2 if is_home else (None if ph2 is None else 1 - ph2)
            if pc is not None and 0.1 <= pc <= 0.9:
                out['alt'].append({'team': side, 'line': tl, 'p': round(pc, 4), 'fair_odds': p_to_ml(pc)})
        out['alt'].sort(key=lambda a: a['line'])
    return out

def edge_validation(PSv, P_main, line_col, sign, ml_cols=None):
    """Walk-forward check: cover probabilities from earlier seasons' errors only, scored on later seasons."""
    D = PSv[['game_id','season','pred_margin','result']].merge(F[['game_id', line_col]], on='game_id').dropna()
    D['line'] = sign*D[line_col]; seasons = sorted(D.season.unique()); rows = []
    for s in seasons[1:]:
        res = (D[D.season < s].result - D[D.season < s].pred_margin).values
        for r in D[D.season == s].itertuples():
            for d in (-7, -3.5, -3, 0, 3, 3.5, 7):
                Lh = r.line + d
                if r.result == Lh: continue
                p = home_cover(r.pred_margin, Lh, res)
                if p is not None: rows.append((d, p, float(r.result > Lh)))
    R = pd.DataFrame(rows, columns=['d','p','y'])
    alt = {'n': int(len(R)), 'brier': float(((R.p-R.y)**2).mean()), 'brier_coinflip': float(((0.5-R.y)**2).mean()),
           'calibration': [{'lo': float(b.left), 'hi': float(b.right), 'n': int(len(g)), 'pred': float(g.p.mean()), 'actual': float(g.y.mean())}
                           for b, g in R.groupby(pd.cut(R.p, [0,.2,.3,.4,.5,.6,.7,.8,1]), observed=True)]}
    M = R[R.d == 0].copy(); M['sp'] = np.maximum(M.p, 1-M.p); M['hit'] = np.where(M.p >= .5, M.y, 1-M.y)
    main = [{'min_p': t, 'n': int((M.sp >= t).sum()), 'cover_rate': float(M[M.sp >= t].hit.mean()) if (M.sp >= t).sum() else None} for t in (.5, .55, .58, .6)]
    out = {'seasons_scored': [int(x) for x in seasons[1:]], 'alt_lines': alt, 'main_line': main, 'break_even_110': 0.5238}
    if ml_cols:
        Q = P_main[P_main.home_win.isin([0,1])].dropna(subset=list(ml_cols)); rr = []
        for r in Q.itertuples():
            for side in ('h','a'):
                p = r.p_home if side == 'h' else 1-r.p_home; ml = getattr(r, ml_cols[0] if side == 'h' else ml_cols[1])
                won = (r.home_win == 1) if side == 'h' else (r.home_win == 0)
                rr.append((p-ml_be(ml), ml_profit(ml) if won else -1.0))
        Rm = pd.DataFrame(rr, columns=['edge','ret'])
        out['moneyline'] = [{'min_edge': t, 'bets': int((Rm.edge > t).sum()), 'win_rate': float((Rm[Rm.edge > t].ret > 0).mean()), 'roi': float(Rm[Rm.edge > t].ret.mean())}
                            for t in (0, .02, .03, .05)]
    return out

def KICKOFF(row):
    try: return pd.Timestamp(f"{pd.Timestamp(row.gameday).date()} {row.gametime}").tz_localize('America/New_York').tz_convert('UTC').to_pydatetime()
    except Exception: return None
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
                      'total_line': None if pd.isna(row.total_line) else float(row.total_line),
                      'source': line_src.get(row.game_id, 'unavailable')},
           'line': line_movement(row), 'injury_news': injury_news(row),
           'home_profile': team_profile(row,'h'), 'away_profile': team_profile(row,'a'),
           'qb_flags': []}
    for s, team in (('h',row.home_team),('a',row.away_team)):
        proj = row.get(f'{s}_qb_name'); dc = dc_qb.get(team)
        if isinstance(proj,str) and dc and proj != dc:
            rec['qb_flags'].append(f'{team}: nflverse projected starter {proj} differs from latest depth chart ({dc}); treat QB as uncertain')
        if not isinstance(proj,str):
            rec['qb_flags'].append(f'{team}: expected starter unknown; QB features imputed with training medians')
    if inj_live_ok.get(row.game_id):
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
    runs.append({'at': now, 'type': 'updated' if inj_live_ok.get(row.game_id) else 'early', 'p_home': rec['updated']['p_home'] if rec['updated'] else rec['early']['p_home'],
                 'model_version': MODEL_VERSION, 'weather_status': (rec['weather'] or {}).get('status')})
    LOG[row.game_id] = {**prior, 'early': rec['early_first'], 'runs': runs}
    rec['stats'] = rec['updated'] or rec['early']
    rec['stats_type'] = 'updated' if rec['updated'] else 'early'
    if pd.notna(row.home_moneyline) and pd.notna(row.away_moneyline):
        p_c = float(crunch_model.predict_proba(xr[crunch_cols].fillna(crunch_med))[:,1][0])
        c_c, b0c = contributions(crunch_model, crunch_med, crunch_cols, xr.iloc[0])
        rec['crunch'] = {'p_home': round(p_c,4), 'p_away': round(1-p_c,4), 'pick': row.home_team if p_c>=0.5 else row.away_team,
                         'predicted_at': now, 'model_version': MODEL_VERSION+'-crunch', 'input_hash': input_hash, 'intercept_logit': round(b0c,4),
                         'factors': sorted([{'feature': LABELS.get(k,k), 'key': k, 'logit': round(float(v),4)} for k,v in c_c.items()], key=lambda d:-abs(d['logit']))}
        rec['latest'] = rec['crunch']; rec['latest_type'] = 'crunch'
    else:
        rec['crunch'] = None; rec['latest'] = rec['stats']; rec['latest_type'] = rec['stats_type']
    rec['score'] = reconcile(rec['score'], rec['latest']['pick'], row.home_team, row.away_team)
    LOG[row.game_id]['runs'][-1].update({'p_crunch': rec['crunch']['p_home'] if rec['crunch'] else None, 'p_stats': rec['stats']['p_home'],
                                         'spread_line': rec['market']['spread_line'], 'ml_home': rec['market']['home_moneyline']})
    rec['runs'] = runs
    # ---- against-the-spread side: projected margin vs the current line (home expected margin)
    _sl = rec['market'].get('spread_line'); _m = rec['score'].get('margin')
    if _sl is not None and _m is not None and round(_m - _sl, 1) != 0:
        _home = _m > _sl
        rec['ats'] = {'team': row.home_team if _home else row.away_team, 'line': (-_sl if _home else _sl), 'edge': round(abs(_m - _sl), 1)}
    else:
        rec['ats'] = None
    rec['value'] = value_check(row, rec, (LIVE.get(row.game_id) or {}).get('odds'))
    # ---- lock at kickoff: after a game starts, show the last pre-kickoff prediction and stop logging runs
    _ko = KICKOFF(row)
    _entry = LOG[row.game_id]
    if _ko is not None and datetime.now(timezone.utc) >= _ko:
        if _entry['runs'] and _entry['runs'][-1].get('at') == now: _entry['runs'].pop()
        _late = [r for r in _entry['runs'] if datetime.fromisoformat(r['at']) >= _ko]
        if _late:
            _entry.setdefault('runs_after_kickoff', []).extend(_late)
            _entry['runs'] = [r for r in _entry['runs'] if datetime.fromisoformat(r['at']) < _ko]
        if _entry.get('frozen'):
            rec = {**_entry['frozen'], 'early_first': rec['early_first'], 'runs': _entry['runs'], 'locked': True, 'locked_note': None}
        else:
            _last = _entry['runs'][-1] if _entry['runs'] else None
            rec['value'] = None; rec['ats'] = None
            rec['crunch'] = None; rec['line'] = None; rec['latest'] = dict(rec['stats']); rec['latest_type'] = rec['stats_type']
            if _last:   # show the probability that was actually on record before kickoff
                rec['latest'].update({'p_home': _last['p_home'], 'p_away': round(1-_last['p_home'],4), 'pick': row.home_team if _last['p_home']>=0.5 else row.away_team, 'predicted_at': _last['at']})
            rec['score'] = reconcile(rec['score'], rec['latest']['pick'], row.home_team, row.away_team)
            rec['runs'] = _entry['runs']; rec['locked'] = True
            rec['locked_note'] = ('No full pre-kickoff snapshot was saved for this game (it started before kickoff locking was added), so the last stats-only call before kickoff is shown; the reasons listed come from the current stats model. '
                                  + (f"Last call before kickoff: {row.home_team if _last['p_home']>=0.5 else row.away_team} {max(_last['p_home'],1-_last['p_home'])*100:.0f}% ({_last['at'][:16].replace('T',' ')} UTC)." if _last else ''))
    else:
        _entry['frozen'] = {k: v for k, v in rec.items() if k not in ('runs', 'early_first')}
        rec['locked'] = False; rec['locked_note'] = None
    games_out.append(rec)
json.dump(LOG, open(LOG_PATH,'w'), indent=0, default=float)

# ---------------------------------------------------------------- history for dashboard: walk-forward preds of final early model
def ats_table(A):
    """A: game_id, season, p_home, home_team, pred_margin, result, line (home expected margin). Pushes and pick'ems excluded."""
    A = A.dropna(subset=['pred_margin','result','line']); A = A[A.line != 0]
    r = A.result - A.line; A = A[r != 0]; r = r[r != 0]
    fav = np.where(A.line > 0, r > 0, r < 0); side = np.where(A.pred_margin > A.line, r > 0, r < 0); pick = np.where(A.p_home >= 0.5, r > 0, r < 0)
    def row(m): return {'n': int(m.sum()), 'model_side': float(side[m].mean()) if m.sum() else None, 'winner_pick': float(pick[m].mean()) if m.sum() else None, 'favorite': float(fav[m].mean()) if m.sum() else None}
    out = {'all': row(np.ones(len(A), bool)), 'by_season': {int(s): row((A.season == s).values) for s in sorted(A.season.unique())}, 'break_even': 0.5238}
    return out
_A = P_crunch[['game_id','season','p_home']].merge(PS[['game_id','pred_margin','result']], on='game_id').merge(F[['game_id','spread_line']], on='game_id')
_A['line'] = _A['spread_line']
ats_metrics = ats_table(_A)
edge_val = edge_validation(PS, P_crunch, 'spread_line', 1, ('home_moneyline','away_moneyline'))
hist = P_crunch.merge(F[['game_id','gameday','home_score','away_score','spread_line']], on='game_id').merge(PS[['game_id','pred_margin','pred_total']], on='game_id', how='left')
hist['mkt'] = [devig(h,a) if pd.notna(h) and pd.notna(a) else None for h,a in zip(hist.home_moneyline,hist.away_moneyline)]
hist_out = [{'game_id':r.game_id,'season':int(r.season),'week':int(r.week),'gameday':str(r.gameday.date()),'home':r.home_team,'away':r.away_team,
             'p_home':round(float(r.p_home),4),'mkt':None if r.mkt is None or pd.isna(r.mkt) else round(float(r.mkt),4),
             'home_score':int(r.home_score),'away_score':int(r.away_score),'home_win':float(r.home_win),
             'pred_home': None if pd.isna(r.pred_margin) else int(round((r.pred_total+r.pred_margin)/2)), 'pred_away': None if pd.isna(r.pred_margin) else int(round((r.pred_total-r.pred_margin)/2)), 'pred_margin': None if pd.isna(r.pred_margin) else round(float(r.pred_margin),1), 'spread': None if pd.isna(r.spread_line) else float(r.spread_line)} for r in hist.itertuples()]

# confidence tiers (from walk-forward history of the final early model)
hist_c = P_crunch.copy(); hist_c['conf'] = np.maximum(hist_c.p_home, 1-hist_c.p_home); hist_c['hit'] = ((hist_c.p_home>0.5)==(hist_c.home_win==1)) & hist_c.home_win.isin([0,1])
TIERS = [('Lean',0.5,0.58),('Moderate',0.58,0.68),('Strong',0.68,1.01)]
tiers_out = []
for name,lo,hi in TIERS:
    t = hist_c[(hist_c.conf>=lo)&(hist_c.conf<hi)&hist_c.home_win.isin([0,1])]
    tiers_out.append({'tier':name,'lo':lo,'hi':min(hi,1.0),'n':int(len(t)),'hit_rate': float(t.hit.mean()) if len(t) else None})

coef_table = {'early': dict(zip(early_cols, [round(float(c),4) for c in early_model.named_steps['logisticregression'].coef_[0]])),
              'updated': dict(zip(upd_cols, [round(float(c),4) for c in upd_model.named_steps['logisticregression'].coef_[0]])),
              'crunch': dict(zip(crunch_cols, [round(float(c),4) for c in crunch_model.named_steps['logisticregression'].coef_[0]]))}

bundle = {
 'generated_at': now, 'model_version': MODEL_VERSION, 'season': cur_season, 'week': cur_week,
 'data': {'pbp_through': meta['pbp_last_game_date'], 'features_built_at': meta['built_at'],
          'injury_report_weeks_2026': meta['injury_weeks_2026'], 'injuries_available_for_week': injuries_available,
          'depth_chart_latest': meta['depth_chart_2026_latest'], 'ftn_latest_pull': meta['ftn_2026_latest_pull'],
          'sources': ['nflverse play-by-play (nflverse-data releases)', 'nflverse schedules/games.csv (nfldata)',
                      'nflverse injuries (official NFL reports)', 'ESPN public API: current betting lines (open and current) and injury statuses, checked each run',
                      'nflverse Next Gen Stats (player tracking)', 'nflverse depth charts', 'FTN charting via nflverse (2022+)',
                      'Open-Meteo hourly forecasts (open-meteo.com)']},
 'definitions': {
   'target': 'P(home team wins). Ties count 0.5 in Brier score; excluded from winner accuracy; a pick is only correct if that team wins outright.',
   'early_cutoff': 'Tuesday of game week. Uses only games completed before the week starts plus the kickoff-hour weather forecast available at run time. No injury reports.',
   'updated_cutoff': 'Re-run daily. Once the official injury report for the week exists (Wed-Fri practice reports, Friday game status), the updated model adds injury features; forecasts and expected starters are refreshed on every run. The first early prediction is kept unchanged alongside.',
   'scores': 'Projected score = two ridge regressions (alpha 100) fit on the same validated inputs: expected margin (home minus away) and expected total points (team offense/defense sums, pace, roof, wind, cold). Home = (total + margin)/2, away = (total - margin)/2, rounded. The winner pick always comes from the win-probability model; in near coin-flip games the rounded score can tie or point the other way, in which case the picked team is shown with a one-point lead and the score is marked adjusted.',
   'crunch': 'Final pick = team efficiency (opponent-adjusted EPA) plus the betting line (de-vigged moneyline and point spread), at the owner\'s request. Historical training uses nflverse closing lines; live picks use the current ESPN line. Walk-forward 2022-24: 69.6% / Brier 0.2080 vs market alone 68.1% / 0.2092. Holdout 2025-26: 65.5% / 0.2185 vs market 65.8% / 0.2139. In other words it tracks the market closely; it is no longer an independent read.',
   'stats_only': 'Stats-only pick = no betting data: efficiency, QB form, special teams, matchups, weather, Next Gen Stats offense (time to throw, rush yards over expected, receiver separation, YAC over expected), plus injury starter counts once statuses are known. Shown alongside so the gut-vs-model experiment keeps a version that never sees the odds.',
   'training': 'Logistic regression (L2, C=0.05) on home-minus-away feature differences. Walk-forward: each week predicted by a model fit only on games before that week; evaluation window 2022 wk1 onward. Model selection used 2022-2024; 2025-2026 reported as untouched holdout.',
   'not_in_model': ['Precipitation (no historical data to learn from; shown per game for context). Wind and cold ARE model inputs, learned from actual kickoff conditions and fed by the Open-Meteo forecast',
                    'Crowd/attendance/home-fan share (no verified feed)', 'Playoff stakes / announced resting of starters (no verified feed)',
                    'Snap-weighted player availability (sum of season snap share of players out/doubtful/questionable, by position group): tested against simple starter counts; starter counts scored better on walk-forward Brier (0.2197 vs 0.2206 on 2022-24, 0.2260 vs 0.2272 on 2025-26), so the updated model keeps the counts and the snap-weighted absences are shown on each game for context',
                    'Next Gen Stats defense-allowed metrics, CPOE and aggressiveness: tested; did not improve the 2022-24 Brier beyond the four offense metrics kept',
                    'Next Gen Stats and injuries in the final (betting-line) model: tested; did not improve it (the line already moves on injuries), so they stay in the stats-only model',
                    'Line movement and betting splits ("sharp money"): no free historical source of opening lines or bet percentages to learn from; movement is shown per game for context, the current line itself is the model input',
                    'Rest, travel, coaching tenure, rivalry: tested, did not improve walk-forward Brier, excluded from final model'],
 },
 'score_metrics': score_metrics, 'ats_metrics': ats_metrics, 'edge_validation': edge_val,
 'live': {**live_meta, 'injuries_from_live': sum(inj_live_ok.values()), 'line_source_counts': {k: sum(1 for v in line_src.values() if v.startswith(k)) for k in ('ESPN','nflverse')}},
 'metrics': {'crunch': res_crunch, 'early': res_early, 'updated': res_upd, 'baseline': res_base,
             'holdout_2025_on': {}},
 'ablation': ablation, 'coefficients': coef_table, 'feature_labels': LABELS, 'confidence_tiers': tiers_out,
 'weather_meta': {'fetched_at': (json.load(open(f'{OUT}/forecasts.json'))['fetched_at'] if FC else None),
                  'sources': 'Open-Meteo hourly forecasts at the venue; kickoff-hour values averaged over the first 3 hours.',
                  'model_note': 'Weather weights were learned from actual kickoff conditions in 2020-2026 games (as the proxy for a forecast). Walk-forward test: adding weather changed Brier by +0.0008 and accuracy by 0.0, i.e. within noise. Its weights are small and data-derived.'},
 'games': games_out, 'history': hist_out,
}
for name, P in (('crunch',P_crunch),('early',P_early),('updated',P_upd),('baseline',P_base)):
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
