"""
Build leakage-free, pre-cutoff features for every NFL game 2020-2026.

Prediction cutoff definition
----------------------------
EARLY cutoff  : the Tuesday of game week (after the previous week's Monday game).
                Only games with gameday < this game's week start are used.
UPDATED cutoff: Saturday of game week, additionally using the official injury
                report (Wed-Fri practice reports + Friday game status) for that week.

All team-level metrics for a game in (season S, week W) use only plays from games
whose week < W in season S, plus prior-season plays as a down-weighted prior.
"""
import numpy as np, pandas as pd, glob, json, os, math, hashlib, sys
from datetime import datetime, timezone

DATA = '/tmp/nfl/data'
NFLDATA = '/home/claude/nflverse/nfldata/data'
OUT = '/home/claude/nflpredict/out'
os.makedirs(OUT, exist_ok=True)
SEASONS = list(range(2020, 2027))

# ---------------------------------------------------------------- schedules
games = pd.read_csv(f'{NFLDATA}/games.csv')
games = games[games.season.isin(SEASONS)].copy()
games['gameday'] = pd.to_datetime(games.gameday)
games['home_win'] = np.where(games.result.isna(), np.nan,
                             np.where(games.result > 0, 1.0, np.where(games.result < 0, 0.0, 0.5)))
games['played'] = games.result.notna()

# ---------------------------------------------------------------- play-by-play
cols = ['game_id','game_date','season','week','posteam','defteam','home_team','away_team','epa','success',
        'pass','rush','qb_dropback','cpoe','interception','sack','qb_scramble','yards_gained','fixed_drive',
        'fixed_drive_result','third_down_converted','third_down_failed','fourth_down_converted',
        'fourth_down_failed','field_goal_result','kick_distance','penalty','penalty_team','fumble_lost',
        'play_type','passer_player_id','passer_player_name','down','ydstogo','yardline_100',
        'special_teams_play','touchdown','td_team','punt_attempt','field_goal_attempt','game_seconds_remaining',
        'play_id','xpass','pass_oe','score_differential','wp','half_seconds_remaining','qtr','no_huddle','shotgun']
pbp = pd.concat([pd.read_csv(f'{DATA}/play_by_play_{s}.csv.gz', usecols=cols, low_memory=False) for s in SEASONS])
pbp['game_date'] = pd.to_datetime(pbp.game_date)
pbp = pbp[pbp.game_id.isin(games.game_id)]
print('plays', len(pbp))

# FTN charting (2022+): motion, play action, blitz
ftn = pd.concat([pd.read_csv(f'{DATA}/ftn_{s}.csv') for s in [2022,2023,2024,2025,2026]])
ftn = ftn.rename(columns={'nflverse_game_id':'game_id','nflverse_play_id':'play_id'})
pbp = pbp.merge(ftn[['game_id','play_id','is_motion','is_play_action','n_blitzers','n_pass_rushers']],
                on=['game_id','play_id'], how='left')

# ---------------------------------------------------------------- per team-game box
off = pbp[(pbp.posteam.notna()) & ((pbp['pass']==1)|(pbp.rush==1)) & pbp.epa.notna()].copy()
off['explosive'] = ((off['pass']==1)&(off.yards_gained>=20)) | ((off.rush==1)&(off.yards_gained>=10))
off['is_db'] = off.qb_dropback==1
off['pass_epa'] = np.where(off.is_db, off.epa, np.nan)
off['rush_epa'] = np.where(off.rush==1, off.epa, np.nan)
off['third'] = off.down==3
off['third_conv'] = off.third_down_converted==1
off['fourth_att'] = (off.down==4)
off['fourth_conv'] = off.fourth_down_converted==1
off['short_yd'] = (off.down.isin([3,4])) & (off.ydstogo<=2)
off['short_conv'] = off.short_yd & ((off.third_down_converted==1)|(off.fourth_down_converted==1))
off['rz'] = off.yardline_100<=20
off['to'] = (off.interception==1)|(off.fumble_lost==1)
off['twomin'] = (off.half_seconds_remaining<=120)
off['blitzed'] = off.is_db & (off.n_blitzers>=1)   # FTN: n_blitzers = rushers beyond the standard 4
off['blitz_epa'] = np.where(off.blitzed, off.epa, np.nan)
off['pa'] = off.is_db & (off.is_play_action==True)
off['motion'] = off.is_motion==True
off['late_close'] = ~((off.qtr==4) & (off.score_differential.abs()>16))  # drop garbage-time

g = off.groupby(['game_id','posteam','defteam'])
tg = pd.DataFrame({
    'plays': g.size(),
    'off_epa': g.epa.mean(), 'off_sr': g.success.mean(),
    'pass_epa': g.pass_epa.mean(), 'rush_epa': g.rush_epa.mean(),
    'explosive_rate': g.explosive.mean(),
    'pass_rate': g.is_db.mean(),
    'pass_oe': g.pass_oe.mean(),
    'third_att': g.third.sum(), 'third_conv': g.third_conv.sum(),
    'fourth_att': g.fourth_att.sum(), 'fourth_conv': g.fourth_conv.sum(),
    'short_att': g.short_yd.sum(), 'short_conv': g.short_conv.sum(),
    'turnovers': g['to'].sum(),
    'sacks_taken': g.sack.sum(), 'dropbacks': g.is_db.sum(), 'ints': g.interception.sum(),
    'cpoe': g.cpoe.mean(),
    'blitz_db': g.blitzed.sum(), 'blitz_epa_sum': g.blitz_epa.sum(min_count=1),
    'pa_db': g.pa.sum(), 'motion_plays': g.motion.sum(),
    'ftn_cov': g.is_motion.apply(lambda s: s.notna().mean()),
}).reset_index()

# drives: points per drive, red zone TD rate
dr = pbp[pbp.posteam.notna() & pbp.fixed_drive.notna()].groupby(['game_id','posteam','fixed_drive']).agg(
    res=('fixed_drive_result','first'), min_yl=('yardline_100','min'), nplays=('play_id','size'),
    secs=('game_seconds_remaining', lambda s: s.max()-s.min())).reset_index()
dr['pts'] = dr.res.map({'Touchdown':7,'Field goal':3}).fillna(0)
dr['rz_trip'] = dr.min_yl<=20
dr['rz_td'] = dr.rz_trip & (dr.res=='Touchdown')
dg = dr.groupby(['game_id','posteam']).agg(drives=('fixed_drive','size'), pts=('pts','sum'),
                                            rz_trips=('rz_trip','sum'), rz_tds=('rz_td','sum'),
                                            secs=('secs','sum'), nplays=('nplays','sum')).reset_index()
tg = tg.merge(dg, on=['game_id','posteam'], how='left')

# special teams & penalties
st = pbp[(pbp.special_teams_play==1)|(pbp.field_goal_attempt==1)]
fg = pbp[pbp.field_goal_attempt==1].groupby(['game_id','posteam']).agg(
    fga=('field_goal_result','size'), fgm=('field_goal_result', lambda s:(s=='made').sum()),
    fg_long_att=('kick_distance', lambda s:(s>=40).sum()),
    fg_long_made=('field_goal_result', lambda s: 0)).reset_index()
lf = pbp[(pbp.field_goal_attempt==1)&(pbp.kick_distance>=40)].groupby(['game_id','posteam']).field_goal_result.apply(lambda s:(s=='made').sum()).rename('fg_long_made').reset_index()
fg = fg.drop(columns='fg_long_made').merge(lf, on=['game_id','posteam'], how='left').fillna({'fg_long_made':0})
stepa = st[st.posteam.notna()&st.epa.notna()].groupby(['game_id','posteam']).agg(st_epa=('epa','sum'), st_plays=('epa','size')).reset_index()
pen = pbp[(pbp.penalty==1)&pbp.penalty_team.notna()].groupby(['game_id','penalty_team']).size().rename('penalties').reset_index().rename(columns={'penalty_team':'posteam'})
tg = tg.merge(fg, on=['game_id','posteam'], how='left').merge(stepa, on=['game_id','posteam'], how='left').merge(pen, on=['game_id','posteam'], how='left')
tg = tg.fillna({'fga':0,'fgm':0,'fg_long_att':0,'fg_long_made':0,'st_epa':0,'st_plays':0,'penalties':0})
tg = tg.merge(games[['game_id','season','week','gameday','home_team']], on='game_id')
tg['is_home'] = tg.posteam==tg.home_team
tg = tg.rename(columns={'posteam':'team','defteam':'opp'})
tg.to_pickle(f'{OUT}/team_game_box.pkl')
print('team-games', len(tg))

# ---------------------------------------------------------------- QB per game
qb = off[off.is_db].groupby(['game_id','passer_player_id']).agg(
    db=('epa','size'), epa_sum=('epa','sum'), cpoe_sum=('cpoe','sum'), cpoe_n=('cpoe','count'),
    ints=('interception','sum'), sacks=('sack','sum'), scr=('qb_scramble','sum'),
    scr_epa=('epa', lambda s: 0), blitz_db=('blitzed','sum'), blitz_epa=('blitz_epa','sum')).reset_index()
scr = off[off.is_db & (off.qb_scramble==1)].groupby(['game_id','passer_player_id']).epa.sum().rename('scr_epa').reset_index()
qb = qb.drop(columns='scr_epa').merge(scr, on=['game_id','passer_player_id'], how='left').fillna({'scr_epa':0})
qb = qb.merge(games[['game_id','gameday','season','week']], on='game_id')
qb = qb.sort_values(['passer_player_id','gameday'])

# ---------------------------------------------------------------- opponent-adjusted ratings (ridge)
METRICS = ['off_epa','off_sr','pass_epa','rush_epa','explosive_rate']
tg['ppd'] = tg.pts/tg.drives
METRICS.append('ppd')
teams = sorted(tg.team.unique())
tix = {t:i for i,t in enumerate(teams)}
NT = len(teams)

def ridge_ratings(rows, metric, lam=2.0):
    """rows: team-game rows with weight. Solve y = off[team] - def[opp] + hfa*is_home, ridge toward 0.
    Returns (off ratings, def ratings) where positive def = good defense (suppresses EPA)."""
    r = rows[rows[metric].notna()]
    n = len(r)
    X = np.zeros((n, 2*NT+1))
    X[np.arange(n), r.team.map(tix).values] = 1
    X[np.arange(n), NT + r.opp.map(tix).values] = -1
    X[:, -1] = r.is_home.values.astype(float)
    w = r.w.values
    y = r[metric].values
    Xw = X * w[:,None]
    A = X.T @ Xw + lam*np.eye(2*NT+1); A[-1,-1] = X[:, -1] @ (X[:, -1]*w) + 0.1
    b = Xw.T @ y
    beta = np.linalg.solve(A, b)
    # center off/def so that league avg = mean metric
    mu = np.average(y, weights=w)
    o = beta[:NT]; d = beta[NT:2*NT]
    o = o - o.mean() + mu; d = d - d.mean()
    return dict(zip(teams,o)), dict(zip(teams,d))

rating_rows = []
weeks = games[['season','week']].drop_duplicates().sort_values(['season','week'])
PRIOR_W = 0.3
for s, w in weeks.itertuples(index=False):
    cur = tg[(tg.season==s)&(tg.week<w)].assign(w=1.0)
    prev = tg[(tg.season==s-1)].assign(w=PRIOR_W)
    rows = pd.concat([cur, prev])
    if len(rows)==0: continue
    rec = {'season':s,'week':w}
    for m in METRICS:
        o,d = ridge_ratings(rows, m)
        for t in teams:
            rating_rows.append({'season':s,'week':w,'team':t,'metric':m,'off':o.get(t,np.nan),'def':d.get(t,np.nan)})
R = pd.DataFrame(rating_rows)
R = R.pivot_table(index=['season','week','team'], columns='metric', values=['off','def']).reset_index()
R.columns = ['season','week','team'] + [f'adj_{a}_{b}' for a,b in R.columns[3:]]
print('ratings', R.shape)

# ---------------------------------------------------------------- season-to-date raw + last-5 (strictly prior weeks, same season)
def rate(num, den, prior_rate, prior_n):
    return (num + prior_rate*prior_n)/(den+prior_n)

lg = {  # league priors (computed from 2020-2024 pooled; used only for shrinkage)
    'third': tg.third_conv.sum()/tg.third_att.sum(), 'fourth': tg.fourth_conv.sum()/max(tg.fourth_att.sum(),1),
    'short': tg.short_conv.sum()/max(tg.short_att.sum(),1), 'rz': tg.rz_tds.sum()/tg.rz_trips.sum(),
    'fg': tg.fgm.sum()/tg.fga.sum(), 'fg_long': tg.fg_long_made.sum()/tg.fg_long_att.sum(),
    'sack': tg.sacks_taken.sum()/tg.dropbacks.sum(), 'int': tg.ints.sum()/tg.dropbacks.sum(),
}
std_rows = []
for (s,t), grp in tg.sort_values('week').groupby(['season','team']):
    grp = grp.sort_values('week')
    for w in range(1, 23):
        prior = grp[grp.week < w]
        n = len(prior)
        rec = {'season':s,'week':w,'team':t,'games_played':n}
        if n==0:
            std_rows.append(rec); continue
        rec['third_rate'] = rate(prior.third_conv.sum(), prior.third_att.sum(), lg['third'], 30)
        rec['fourth_go_rate'] = prior.fourth_att.sum()/n           # coaching aggressiveness (attempts/game)
        rec['fourth_rate'] = rate(prior.fourth_conv.sum(), prior.fourth_att.sum(), lg['fourth'], 10)
        rec['short_rate'] = rate(prior.short_conv.sum(), prior.short_att.sum(), lg['short'], 15)
        rec['rz_td_rate'] = rate(prior.rz_tds.sum(), prior.rz_trips.sum(), lg['rz'], 10)
        rec['fg_rate'] = rate(prior.fgm.sum(), prior.fga.sum(), lg['fg'], 10)
        rec['fg_long_rate'] = rate(prior.fg_long_made.sum(), prior.fg_long_att.sum(), lg['fg_long'], 8)
        rec['sack_rate_off'] = rate(prior.sacks_taken.sum(), prior.dropbacks.sum(), lg['sack'], 60)
        rec['int_rate_off'] = rate(prior.ints.sum(), prior.dropbacks.sum(), lg['int'], 60)
        rec['to_per_game'] = prior.turnovers.mean()
        rec['pen_per_game'] = prior.penalties.mean()
        rec['st_epa_per_game'] = prior.st_epa.mean()
        rec['pass_rate'] = prior.pass_rate.mean(); rec['pass_oe'] = prior.pass_oe.mean()
        rec['pace_secs_per_play'] = prior.secs.sum()/max(prior.nplays.sum(),1)
        ftn_cov = prior.ftn_cov.mean()
        if ftn_cov > 0.5:
            rec['pa_rate'] = prior.pa_db.sum()/max(prior.dropbacks.sum(),1)
            rec['motion_rate'] = prior.motion_plays.sum()/max(prior.plays.sum(),1)
            rec['blitz_epa_off'] = prior.blitz_epa_sum.sum()/max(prior.blitz_db.sum(),1)
        l5 = prior.tail(5)
        rec['l5_off_epa'] = l5.off_epa.mean(); rec['l5_off_sr'] = l5.off_sr.mean(); rec['l5_ppd'] = l5.ppd.mean()
        std_rows.append(rec)
STD = pd.DataFrame(std_rows)

# defensive side: what opponents did against this team (raw, season-to-date + last5)
dtg = tg.rename(columns={'team':'opp','opp':'team'})
def_rows = []
for (s,t), grp in dtg.sort_values('week').groupby(['season','team']):
    for w in range(1,23):
        prior = grp[grp.week<w]
        if len(prior)==0: continue
        rec = {'season':s,'week':w,'team':t}
        rec['def_third_rate'] = rate(prior.third_conv.sum(), prior.third_att.sum(), lg['third'], 30)
        rec['def_rz_td_rate'] = rate(prior.rz_tds.sum(), prior.rz_trips.sum(), lg['rz'], 10)
        rec['sack_rate_def'] = rate(prior.sacks_taken.sum(), prior.dropbacks.sum(), lg['sack'], 60)
        rec['takeaways_per_game'] = prior.turnovers.mean()
        rec['def_explosive_allowed'] = prior.explosive_rate.mean()
        ftn_cov = prior.ftn_cov.mean()
        if ftn_cov > 0.5:
            rec['blitz_rate_def'] = prior.blitz_db.sum()/max(prior.dropbacks.sum(),1)
        l5 = prior.tail(5)
        rec['l5_def_epa'] = l5.off_epa.mean(); rec['l5_def_sr'] = l5.off_sr.mean()
        def_rows.append(rec)
DEF = pd.DataFrame(def_rows)

# ---------------------------------------------------------------- QB trailing form (strictly before gameday, across seasons)
LG_EPA = qb.epa_sum.sum()/qb.db.sum(); LG_CPOE = qb.cpoe_sum.sum()/qb.cpoe_n.sum()
LG_INT = qb.ints.sum()/qb.db.sum(); LG_SACK = qb.sacks.sum()/qb.db.sum()
QB_PRIOR = 150  # dropbacks of league-average prior
def qb_form(pid, before_date, window_db=400):
    h = qb[(qb.passer_player_id==pid)&(qb.gameday<before_date)].sort_values('gameday', ascending=False)
    # take recent games until window_db dropbacks
    cum = h.db.cumsum(); h = h[cum.shift(fill_value=0) < window_db]
    db = h.db.sum()
    out = {'qb_db_hist': db}
    out['qb_epa'] = (h.epa_sum.sum() + LG_EPA*QB_PRIOR)/(db+QB_PRIOR)
    out['qb_cpoe'] = (h.cpoe_sum.sum() + LG_CPOE*QB_PRIOR)/(h.cpoe_n.sum()+QB_PRIOR)
    out['qb_int_rate'] = (h.ints.sum() + LG_INT*QB_PRIOR)/(db+QB_PRIOR)
    out['qb_sack_rate'] = (h.sacks.sum() + LG_SACK*QB_PRIOR)/(db+QB_PRIOR)
    out['qb_scr_epa_per_db'] = h.scr_epa.sum()/(db+QB_PRIOR)
    bdb = h.blitz_db.sum()
    out['qb_blitz_epa'] = (h.blitz_epa.sum() + LG_EPA*50)/(bdb+50) if bdb>0 else np.nan
    return out

# ---------------------------------------------------------------- injuries (UPDATED cutoff only)
inj = pd.concat([pd.read_csv(f'{DATA}/injuries_{s}.csv') for s in SEASONS])
inj = inj[inj.game_type.eq('REG') | inj.season_type.eq('REG') | True]
# starters: depth charts. 2020-2025 format: depth_team==1 per week; 2026 format: pos_rank==1 by date
starters = {}
for s_ in range(2020, 2025):
    d = pd.read_csv(f'{DATA}/depth_{s_}.csv', usecols=['season','club_code','week','depth_team','gsis_id','position','game_type'])
    d = d[(d.depth_team==1)]
    for (sea, wk, team), grp in d.groupby(['season','week','club_code']):
        starters[(sea,wk,team)] = set(grp.gsis_id.dropna())
for s_ in (2025, 2026):
    d26 = pd.read_csv(f'{DATA}/depth_{s_}.csv')
    d26['dt'] = pd.to_datetime(d26.dt)
    for wk in range(1, 23):
        wk_games = games[(games.season==s_)&(games.week==wk)]
        if len(wk_games)==0: continue
        first = wk_games.gameday.min().tz_localize('UTC')
        snap = d26[d26.dt < first + pd.Timedelta(days=1)]
        if len(snap)==0: continue
        snap = snap[snap.dt==snap.dt.max()]
        for team, grp in snap[snap.pos_rank==1].groupby('team'):
            starters[(s_,wk,team)] = set(grp.gsis_id.dropna())

# ---------------------------------------------------------------- snap-weighted availability (UPDATED cutoff only)
snaps = pd.concat([pd.read_csv(f'{DATA}/snaps_{s_}.csv') for s_ in SEASONS])
snaps = snaps[snaps.game_type=='REG']
import unicodedata
def norm_name(n):
    n = unicodedata.normalize('NFKD', str(n)).encode('ascii','ignore').decode().lower()
    n = re.sub(r"[^a-z ]", "", n); n = re.sub(r"\b(jr|sr|ii|iii|iv)\b", "", n)
    return ' '.join(n.split())
import re
snaps['key'] = snaps.player.map(norm_name)
snaps = snaps.merge(games[['game_id','gameday']], on='game_id', how='left')
PG = {'OL':{'T','G','C','OL','OT','OG'}, 'SKILL':{'WR','TE','RB','FB'}, 'DL':{'DE','DT','NT','DL','EDGE'}, 'LB':{'LB','OLB','ILB','MLB'}, 'DB':{'CB','S','DB','FS','SS'}}
def pos_group(p):
    for g_, ps in PG.items():
        if p in ps: return g_
    return None
snaps['pg'] = snaps.position.map(pos_group)
snap_idx = {k: v for k, v in snaps.groupby(['season','team'])}
def availability_feats(season, week, team, gameday):
    """Sum of season-to-date snap share (prior games only) of players listed Out/Doubtful (weight 1) or Questionable (0.5)."""
    r = inj[(inj.season==season)&(inj.week==week)&(inj.team==team)]
    out = {'av_off_out':0.0,'av_def_out':0.0,'av_ol_out':0.0,'av_skill_out':0.0,'av_dl_out':0.0,'av_lb_out':0.0,'av_db_out':0.0,'av_off_q':0.0,'av_def_q':0.0,'av_names':''}
    if len(r)==0 or (season,team) not in snap_idx: return out
    sp = snap_idx[(season,team)]; sp = sp[sp.gameday < gameday]
    if len(sp)==0: return out
    ng = sp.game_id.nunique()
    per = sp.groupby('key').agg(off=('offense_pct','sum'), de=('defense_pct','sum'), pg=('pg','first'), pos=('position','first'))
    per['off'] /= ng; per['de'] /= ng
    names=[]
    for p in r.itertuples():
        if p.position=='QB' or not isinstance(p.report_status,str): continue
        w = 1.0 if p.report_status in ('Out','Doubtful') else (0.5 if p.report_status=='Questionable' else 0)
        if w==0: continue
        k = norm_name(p.full_name)
        if k not in per.index: continue
        o, d, pg = per.loc[k,'off'], per.loc[k,'de'], per.loc[k,'pg']
        if w==1.0:
            out['av_off_out'] += o; out['av_def_out'] += d
            if pg=='OL': out['av_ol_out'] += o
            if pg=='SKILL': out['av_skill_out'] += o
            if pg=='DL': out['av_dl_out'] += d
            if pg=='LB': out['av_lb_out'] += d
            if pg=='DB': out['av_db_out'] += d
            if max(o,d) >= 0.4: names.append(f"{p.full_name} ({p.position}, {int(round(max(o,d)*100))}% snaps)")
        else:
            out['av_off_q'] += o*w; out['av_def_q'] += d*w
    out['av_names'] = '; '.join(names[:8])
    return out

def injury_feats(season, week, team):
    r = inj[(inj.season==season)&(inj.week==week)&(inj.team==team)]
    if len(r)==0:
        return {'inj_available': 0}
    st = starters.get((season,week,team), set())
    out_ = r[r.report_status.isin(['Out','Doubtful'])]
    q = r[r.report_status.eq('Questionable')]
    return {'inj_available': 1,
            'inj_starters_out': int(out_.gsis_id.isin(st).sum()),
            'inj_starters_q': int(q.gsis_id.isin(st).sum()),
            'inj_total_out': int(len(out_)),
            'inj_qb_out': int((out_.position=='QB').sum()>0),
            'inj_ol_out': int(((out_.position.isin(['T','G','C','OL','OT','OG']))&(out_.gsis_id.isin(st))).sum()),
            'inj_names_out': '; '.join(out_.full_name.head(8).tolist())}

# ---------------------------------------------------------------- rest / travel / coaches / rivalry
air = pd.read_csv(f'{NFLDATA}/airports.csv').set_index('team')
INTL = {'Tottenham Hotspur Stadium':(51.604,-0.066,0), 'Wembley Stadium':(51.556,-0.280,0),
        'Allianz Arena':(48.219,11.625,1),'Deutsche Bank Park':(50.069,8.645,1),'Frankfurt Stadium':(50.069,8.645,1),
        'Neo Química Arena':(-23.545,-46.474,-3),'Corinthians Arena':(-23.545,-46.474,-3),
        'Croke Park':(53.361,-6.251,0),'Santiago Bernabéu':(40.453,-3.688,1),'Bernabeu Stadium':(40.453,-3.688,1),
        'Olympiastadion':(52.515,13.239,1),'Olympic Stadium':(52.515,13.239,1),'Melbourne Cricket Ground':(-37.820,144.983,10),
        'Estadio Azteca':(19.303,-99.150,-6)}
def hav(a,b,c,d):
    R=3959; p1,p2=math.radians(a),math.radians(c); dp=p2-p1; dl=math.radians(d-b)
    return 2*R*math.asin(math.sqrt(math.sin(dp/2)**2+math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2))
def venue(row):
    if row.location=='Neutral' or row.stadium in INTL:
        if row.stadium in INTL:
            la,lo,tz = INTL[row.stadium]; return la,lo,tz,1
        return air.loc[row.home_team,'latitude'],air.loc[row.home_team,'longitude'],air.loc[row.home_team,'time_zone'],0
    return air.loc[row.home_team,'latitude'],air.loc[row.home_team,'longitude'],air.loc[row.home_team,'time_zone'],0

coach_games = {}
all_games_hist = pd.read_csv(f'{NFLDATA}/games.csv')[['season','week','gameday','home_team','away_team','home_coach','away_coach','result']]
all_games_hist['gameday']=pd.to_datetime(all_games_hist.gameday)
all_games_hist = all_games_hist.sort_values('gameday')
coach_hist = pd.concat([all_games_hist[['gameday','home_team','home_coach']].rename(columns={'home_team':'team','home_coach':'coach'}),
                        all_games_hist[['gameday','away_team','away_coach']].rename(columns={'away_team':'team','away_coach':'coach'})]).sort_values('gameday')
def coach_tenure(team, coach, before):
    h = coach_hist[(coach_hist.team==team)&(coach_hist.gameday<before)]
    if len(h)==0: return 0, 1
    tenure = int((h.coach==coach).sum())
    changed = int(h.iloc[-1].coach != coach)
    return tenure, changed

games_sorted = games.sort_values('gameday')
def team_sched(team):
    h = games_sorted[(games_sorted.home_team==team)|(games_sorted.away_team==team)]
    return h

sched_cache = {t: team_sched(t) for t in teams}
def rest_travel(row, team, is_home):
    h = sched_cache[team]; prior = h[h.gameday<row.gameday]
    rec = {}
    last = prior.tail(1)
    rec['prev_ot'] = int(last.overtime.fillna(0).iloc[0]) if len(last) else 0
    # consecutive road games prior
    c=0
    for _,p in prior.iloc[::-1].iterrows():
        if p.away_team==team and p.location!='Neutral': c+=1
        else: break
    rec['consec_road_prior'] = c
    la,lo,tz,intl = venue(row)
    home_lat,home_lon,home_tz = air.loc[team,'latitude'],air.loc[team,'longitude'],air.loc[team,'time_zone']
    rec['travel_miles'] = 0.0 if (is_home and not intl) else hav(home_lat,home_lon,la,lo)
    rec['tz_shift'] = 0.0 if (is_home and not intl) else abs(tz-home_tz)
    rec['intl'] = intl
    # previous meeting margin (last 400 days) from this team's view
    m = prior[((prior.home_team==team)&(prior.away_team==(row.away_team if is_home else row.home_team)))|
              ((prior.away_team==team)&(prior.home_team==(row.away_team if is_home else row.home_team)))]
    m = m[m.gameday > row.gameday - pd.Timedelta(days=400)]
    if len(m):
        mm = m.iloc[-1]; margin = mm.result if mm.home_team==team else -mm.result
        rec['prev_meeting_margin'] = margin; rec['prev_meeting_n'] = len(m)
    else:
        rec['prev_meeting_margin'] = 0.0; rec['prev_meeting_n'] = 0
    return rec

TEAM_ROOF = (games[games.location!='Neutral'].assign(r=lambda d: d.roof.isin(['dome','closed']).astype(int)).groupby('home_team').r.mean() > 0.5).astype(int).to_dict()
# venue coordinates for forecasts
VENUES = {}
for t in teams:
    VENUES[t] = {'lat': float(air.loc[t,'latitude']), 'lon': float(air.loc[t,'longitude']), 'tz_vs_et': float(air.loc[t,'time_zone'])}
json.dump({'team_venues': VENUES, 'intl': {k:{'lat':v[0],'lon':v[1],'utc_offset':v[2]} for k,v in INTL.items()}, 'team_roof': TEAM_ROOF}, open(f'{OUT}/venues.json','w'), indent=1)
# ---------------------------------------------------------------- assemble per game
feat_rows = []
R_idx = R.set_index(['season','week','team'])
STD_idx = STD.set_index(['season','week','team'])
DEF_idx = DEF.set_index(['season','week','team'])
def side_feats(row, team, is_home):
    s,w = row.season, row.week
    f = {}
    for src in (R_idx, STD_idx, DEF_idx):
        if (s,w,team) in src.index:
            f.update(src.loc[(s,w,team)].to_dict())
    qbid = row.home_qb_id if is_home else row.away_qb_id
    qbname = row.home_qb_name if is_home else row.away_qb_name
    f['qb_id'] = qbid; f['qb_name'] = qbname
    if isinstance(qbid, str):
        f.update(qb_form(qbid, row.gameday))
        # qb change vs previous game's starter for this team
        h = sched_cache[team]; prior = h[h.gameday<row.gameday].tail(1)
        if len(prior):
            p = prior.iloc[0]; prev_qb = p.home_qb_id if p.home_team==team else p.away_qb_id
            f['qb_change'] = int(prev_qb != qbid)
        else: f['qb_change'] = 0
    else:
        f['qb_db_hist'] = np.nan
    f['rest'] = row.home_rest if is_home else row.away_rest
    coach = row.home_coach if is_home else row.away_coach
    f['coach'] = coach
    f['coach_tenure'], f['coach_changed'] = coach_tenure(team, coach, row.gameday)
    f.update(rest_travel(row, team, is_home))
    f.update(injury_feats(s, w, team))
    f.update(availability_feats(s, w, team, row.gameday))
    return f

for row in games.itertuples(index=False):
    row = pd.Series(row._asdict())
    rec = {'game_id':row.game_id,'season':row.season,'week':row.week,'gameday':row.gameday,'gametime':row.gametime,
           'home_team':row.home_team,'away_team':row.away_team,'home_win':row.home_win,'played':row.played,
           'result':row.result,'home_score':row.home_score,'away_score':row.away_score,
           'neutral':int(row.location=='Neutral'),'div_game':row.div_game,'roof':row.roof,'surface':row.surface,
           'stadium':row.stadium,'game_type':row.game_type,
           'spread_line':row.spread_line,'home_moneyline':row.home_moneyline,'away_moneyline':row.away_moneyline,
           'total':row.total,'total_line':row.total_line}
    roofed = row.roof in ('dome','closed')
    rec['wx_wind'] = 0.0 if roofed else (row.wind if pd.notna(row.wind) else np.nan)
    rec['wx_temp'] = 70.0 if roofed else (row.temp if pd.notna(row.temp) else np.nan)
    rec['wx_source'] = 'roofed' if roofed else ('actual' if pd.notna(row.wind) else 'missing')
    hf = side_feats(row, row.home_team, True); af = side_feats(row, row.away_team, False)
    hf['home_dome'] = int(TEAM_ROOF.get(row.home_team,0)); af['home_dome'] = int(TEAM_ROOF.get(row.away_team,0))
    for k,v in hf.items(): rec['h_'+k]=v
    for k,v in af.items(): rec['a_'+k]=v
    feat_rows.append(rec)
F = pd.DataFrame(feat_rows)
F.to_pickle(f'{OUT}/game_features.pkl')
print('games', len(F), 'played', F.played.sum())
meta = {'built_at': datetime.now(timezone.utc).isoformat(), 'seasons': SEASONS,
        'pbp_last_game_date': str(pbp.game_date.max().date()), 'plays': int(len(pbp)),
        'injury_weeks_2026': sorted(inj[inj.season==2026].week.unique().tolist()),
        'depth_chart_2026_latest': str(d26.dt.max()),
        'ftn_2026_latest_pull': str(ftn[ftn.season==2026].date_pulled.max())}
json.dump(meta, open(f'{OUT}/data_meta.json','w'), indent=1)
print(meta)
