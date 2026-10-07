"""College football (FBS) feature builder. Mirrors the NFL pipeline: strictly pre-cutoff features per game.
Cutoff: Monday of game week — only games from earlier weeks of the season (plus last season as a down-weighted prior).
Data: cfbfastR-data (ESPN play-by-play with EPA, CFBD schedules, team info)."""
import numpy as np, pandas as pd, json, os, math, re
from datetime import datetime, timezone
DATA='/tmp/cfb/data'; SRC='/home/claude/sportsdataverse/cfbfastr-data'; OUT='/home/claude/cfbpredict/out'
os.makedirs(OUT, exist_ok=True)
SEASONS=[2023,2024,2025,2026]

# ---------------------------------------------------------------- schedules
sch = pd.concat([pd.read_csv(f'{SRC}/schedules/csv/cfb_schedules_{s}.csv') for s in SEASONS])
sch = sch[(sch.home_division=='fbs')|(sch.away_division=='fbs')].copy()
sch['start'] = pd.to_datetime(sch.start_date, utc=True)
sch['gameday'] = sch.start.dt.tz_convert('America/New_York').dt.normalize().dt.tz_localize(None)
sch['wk'] = np.where(sch.season_type=='postseason', sch.week+20, sch.week)   # ordering key
sch['played'] = sch.home_points.notna() & sch.away_points.notna() & sch.completed.fillna(False).astype(bool)
sch['result'] = sch.home_points - sch.away_points
sch['total'] = sch.home_points + sch.away_points
sch['home_win'] = np.where(sch.played, np.where(sch.result>0,1.0,np.where(sch.result<0,0.0,0.5)), np.nan)
sch['game_id'] = sch.game_id.astype(int)
games = sch
print('games', len(games), 'played', games.played.sum())

# team info (venues)
ti = pd.concat([pd.read_pickle(f'{DATA}/team_info_{s}.pkl').assign(season=s) for s in (2025,2026)]).drop_duplicates('school', keep='last').set_index('school')
FBS = set(ti[ti.classification=='fbs'].index)
def ent(team):   # rating entity: FBS teams individually, everyone else pooled
    return team if team in FBS else 'NON-FBS'

# ---------------------------------------------------------------- play-by-play
pbp = pd.concat([pd.read_pickle(f'{DATA}/pbp_{s}.pkl') for s in SEASONS])
pbp = pbp.rename(columns={'year':'season'})
pbp['game_id'] = pbp.game_id.astype(int)
pbp = pbp[pbp.game_id.isin(games.game_id)]
pbp = pbp.merge(games[['game_id','gameday','wk','season']].rename(columns={'season':'season_g'}), on='game_id')
pbp['season'] = pbp.season_g
print('plays', len(pbp))
# market/elo per game (pregame, from pbp rows)
gm = pbp.groupby('game_id').agg(spread=('spread','first'), over_under=('over_under','first'), home_elo=('home_team_pregame_elo','first'), away_elo=('away_team_pregame_elo','first')).reset_index()
games = games.merge(gm, on='game_id', how='left')
# elo for upcoming games from schedule columns if present
games['home_elo'] = games.home_elo.fillna(games.home_pregame_elo); games['away_elo'] = games.away_elo.fillna(games.away_pregame_elo)

off = pbp[pbp.pos_team.notna() & ((pbp['pass']==1)|(pbp.rush==1)) & pbp.EPA.notna()].copy()
off['is_db'] = (off.pass_attempt==1)|(off.sack==1)
off['explosive'] = ((off['pass']==1)&(off.yards_gained>=20))|((off.rush==1)&(off.yards_gained>=12))
off['pass_epa'] = np.where(off.is_db, off.EPA, np.nan); off['rush_epa'] = np.where(off.rush==1, off.EPA, np.nan)
off['third'] = off.down==3; off['third_conv'] = off.third & (off.yards_gained>=off.distance)
off['fourth'] = off.down==4; off['fourth_conv'] = off.fourth & (off.yards_gained>=off.distance)
off['to'] = (off['int']==1)|(off.fumble_vec==1)&(off.turnover==1)
g = off.groupby(['game_id','pos_team','def_pos_team'])
tg = pd.DataFrame({'plays':g.size(),'off_epa':g.EPA.mean(),'off_sr':g.success.mean(),'pass_epa':g.pass_epa.mean(),'rush_epa':g.rush_epa.mean(),
    'explosive_rate':g.explosive.mean(),'pass_rate':g.is_db.mean(),'third_att':g.third.sum(),'third_conv':g.third_conv.sum(),
    'fourth_att':g.fourth.sum(),'fourth_conv':g.fourth_conv.sum(),'turnovers':g['to'].sum(),'sacks_taken':g.sack.sum(),'dropbacks':g.is_db.sum(),'ints':g['int'].sum()}).reset_index()
# drives
dr = pbp[pbp.pos_team.notna() & pbp.drive_id.notna()].groupby(['game_id','pos_team','drive_id']).agg(pts=('drive_pts','max'), min_ytg=('yards_to_goal','min'), res=('drive_result','first')).reset_index()
dr['pts'] = dr.pts.fillna(0).clip(lower=0); dr['rz'] = dr.min_ytg<=20; dr['rz_td'] = dr.rz & (dr.pts>=6)
dg = dr.groupby(['game_id','pos_team']).agg(drives=('drive_id','size'), pts=('pts','sum'), rz_trips=('rz','sum'), rz_tds=('rz_td','sum')).reset_index()
tg = tg.merge(dg, on=['game_id','pos_team'], how='left')
fg = pbp[pbp.fg_inds==1].groupby(['game_id','pos_team']).agg(fga=('fg_made','size'), fgm=('fg_made','sum')).reset_index()
pen = pbp[(pbp.penalty_flag==1)].groupby(['game_id','pos_team']).size().rename('penalties').reset_index()
tg = tg.merge(fg, on=['game_id','pos_team'], how='left').merge(pen, on=['game_id','pos_team'], how='left').fillna({'fga':0,'fgm':0,'penalties':0})
tg = tg.merge(games[['game_id','season','wk','gameday','home_team','neutral_site']], on='game_id')
tg['is_home'] = (tg.pos_team==tg.home_team) & (~tg.neutral_site.fillna(False).astype(bool))
tg = tg.rename(columns={'pos_team':'team','def_pos_team':'opp'})
tg['ppd'] = tg.pts/tg.drives
tg.to_pickle(f'{OUT}/team_game_box.pkl'); print('team-games', len(tg))

# ---------------------------------------------------------------- QB per game
qbp = off[off.is_db].copy()
qbp['qb'] = qbp.passer_player_name.where(qbp.passer_player_name.notna(), qbp.sack_taken_player)
qb = qbp[qbp.qb.notna()].groupby(['game_id','pos_team','qb']).agg(db=('EPA','size'), epa_sum=('EPA','sum'), comp=('completion','sum'), att=('pass_attempt','sum'), ints=('int','sum'), sacks=('sack','sum')).reset_index()
qb = qb.merge(games[['game_id','gameday','season','wk']], on='game_id').sort_values(['pos_team','gameday'])
LG_EPA = qb.epa_sum.sum()/qb.db.sum(); LG_CMP = qb.comp.sum()/qb.att.sum(); LG_INT = qb.ints.sum()/qb.db.sum(); LG_SACK = qb.sacks.sum()/qb.db.sum()
QB_PRIOR = 120
qb_by_team = {k: v for k, v in qb.groupby('pos_team')}
def expected_starter(team, before):
    h = qb_by_team.get(team)
    if h is None: return None
    h = h[h.gameday < before]
    if len(h)==0: return None
    last = h[h.game_id==h.iloc[-1].game_id]
    return last.sort_values('db').iloc[-1].qb
def qb_form(team, name, before, window=350):
    h = qb_by_team.get(team); h = h[(h.qb==name)&(h.gameday<before)].sort_values('gameday', ascending=False)
    cum = h.db.cumsum(); h = h[cum.shift(fill_value=0) < window]; db = h.db.sum()
    return {'qb_db_hist': db, 'qb_epa': (h.epa_sum.sum()+LG_EPA*QB_PRIOR)/(db+QB_PRIOR), 'qb_cmp': (h.comp.sum()+LG_CMP*QB_PRIOR)/(h.att.sum()+QB_PRIOR),
            'qb_int_rate': (h.ints.sum()+LG_INT*QB_PRIOR)/(db+QB_PRIOR), 'qb_sack_rate': (h.sacks.sum()+LG_SACK*QB_PRIOR)/(db+QB_PRIOR)}

# ---------------------------------------------------------------- opponent-adjusted ridge ratings
METRICS=['off_epa','off_sr','pass_epa','rush_epa','explosive_rate','ppd']
tg['team_e'] = tg.team.map(ent); tg['opp_e'] = tg.opp.map(ent)
ents = sorted(set(tg.team_e)|set(tg.opp_e)); eix={e:i for i,e in enumerate(ents)}; NE=len(ents)
def ridge_ratings(rows, metric, lam=3.0):
    r = rows[rows[metric].notna()]; n=len(r)
    X = np.zeros((n, 2*NE+1)); X[np.arange(n), r.team_e.map(eix).values]=1; X[np.arange(n), NE+r.opp_e.map(eix).values]=-1; X[:,-1]=r.is_home.values.astype(float)
    w = r.w.values; y = r[metric].values; Xw = X*w[:,None]
    A = X.T@Xw + lam*np.eye(2*NE+1); A[-1,-1] = X[:,-1]@(X[:,-1]*w)+0.1; beta = np.linalg.solve(A, Xw.T@y)
    mu = np.average(y, weights=w); o = beta[:NE]; d = beta[NE:2*NE]; o = o-o.mean()+mu; d = d-d.mean()
    return dict(zip(ents,o)), dict(zip(ents,d))
PRIOR_W=0.25
rating_rows=[]
for s,w in games[['season','wk']].drop_duplicates().sort_values(['season','wk']).itertuples(index=False):
    cur = tg[(tg.season==s)&(tg.wk<w)].assign(w=1.0); prev = tg[tg.season==s-1].assign(w=PRIOR_W); rows = pd.concat([cur,prev])
    if len(rows)==0: continue
    for m in METRICS:
        o,d = ridge_ratings(rows, m)
        for e in ents: rating_rows.append({'season':s,'wk':w,'team':e,'metric':m,'off':o[e],'def':d[e]})
R = pd.DataFrame(rating_rows).pivot_table(index=['season','wk','team'], columns='metric', values=['off','def']).reset_index()
R.columns = ['season','wk','team']+[f'adj_{a}_{b}' for a,b in R.columns[3:]]
R_idx = R.set_index(['season','wk','team']); print('ratings', R.shape)

# ---------------------------------------------------------------- season-to-date raw rates (prior weeks, same season)
def rate(num, den, pr, pn): return (num+pr*pn)/(den+pn)
lg = {'third': tg.third_conv.sum()/tg.third_att.sum(), 'fourth': tg.fourth_conv.sum()/max(tg.fourth_att.sum(),1), 'rz': tg.rz_tds.sum()/tg.rz_trips.sum(),
      'fg': tg.fgm.sum()/tg.fga.sum(), 'sack': tg.sacks_taken.sum()/tg.dropbacks.sum(), 'int': tg.ints.sum()/tg.dropbacks.sum()}
tg_by = {k: v.sort_values('wk') for k, v in tg.groupby(['season','team'])}
dtg_by = {k: v.sort_values('wk') for k, v in tg.rename(columns={'team':'opp','opp':'team'}).groupby(['season','team'])}
def std_feats(s, w, team):
    grp = tg_by.get((s,team)); rec = {'games_played': 0}
    if grp is None: return rec
    prior = grp[grp.wk<w]; n=len(prior); rec['games_played']=n
    if n==0: return rec
    rec.update({'third_rate': rate(prior.third_conv.sum(), prior.third_att.sum(), lg['third'], 25), 'fourth_rate': rate(prior.fourth_conv.sum(), prior.fourth_att.sum(), lg['fourth'], 8),
        'fourth_go_rate': prior.fourth_att.sum()/n, 'rz_td_rate': rate(prior.rz_tds.sum(), prior.rz_trips.sum(), lg['rz'], 8), 'fg_rate': rate(prior.fgm.sum(), prior.fga.sum(), lg['fg'], 8),
        'sack_rate_off': rate(prior.sacks_taken.sum(), prior.dropbacks.sum(), lg['sack'], 50), 'int_rate_off': rate(prior.ints.sum(), prior.dropbacks.sum(), lg['int'], 50),
        'to_per_game': prior.turnovers.mean(), 'pen_per_game': prior.penalties.mean(), 'pass_rate': prior.pass_rate.mean(), 'plays_per_game': prior.plays.mean(),
        'l3_off_epa': prior.tail(3).off_epa.mean(), 'l3_ppd': prior.tail(3).ppd.mean()})
    d = dtg_by.get((s,team))
    if d is not None:
        dp = d[d.wk<w]
        if len(dp):
            rec.update({'def_third_rate': rate(dp.third_conv.sum(), dp.third_att.sum(), lg['third'], 25), 'def_rz_td_rate': rate(dp.rz_tds.sum(), dp.rz_trips.sum(), lg['rz'], 8),
                        'sack_rate_def': rate(dp.sacks_taken.sum(), dp.dropbacks.sum(), lg['sack'], 50), 'takeaways_per_game': dp.turnovers.mean(), 'l3_def_epa': dp.tail(3).off_epa.mean()})
    return rec

# ---------------------------------------------------------------- rest / travel / venue
def hav(a,b,c,d):
    R_=3959; p1,p2=math.radians(a),math.radians(c); dp=p2-p1; dl=math.radians(d-b)
    return 2*R_*math.asin(math.sqrt(math.sin(dp/2)**2+math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2))
gs = games.sort_values('start')
sched_by = {}
for t in set(games.home_team)|set(games.away_team):
    sched_by[t] = gs[(gs.home_team==t)|(gs.away_team==t)]
def rest_travel(row, team, is_home):
    h = sched_by[team]; prior = h[h.start<row.start]; rec={}
    rec['rest'] = (row.start - prior.iloc[-1].start).days if len(prior) else 14
    rec['bye'] = int(rec['rest']>=12)
    vlat = ti.latitude.get(row.home_team, np.nan); vlon = ti.longitude.get(row.home_team, np.nan)
    tlat = ti.latitude.get(team, np.nan); tlon = ti.longitude.get(team, np.nan)
    rec['travel_miles'] = 0.0 if (is_home and not row.neutral_site) else (hav(tlat,tlon,vlat,vlon) if pd.notna(vlat) and pd.notna(tlat) else np.nan)
    rec['home_dome'] = int(ti.dome.get(team, 0) == 1)
    return rec

# ---------------------------------------------------------------- assemble
rows=[]
for row in games.itertuples(index=False):
    row = pd.Series(row._asdict())
    s,w = row.season, row.wk
    rec = {'game_id':row.game_id,'season':s,'week':int(row.week),'wk':w,'season_type':row.season_type,'start_utc':row.start.isoformat(),'gameday':row.gameday,
           'home_team':row.home_team,'away_team':row.away_team,'home_conf':row.home_conference,'away_conf':row.away_conference,'home_div':row.home_division,'away_div':row.away_division,
           'home_win':row.home_win,'played':bool(row.played),'result':row.result,'total':row.total,'home_score':row.home_points,'away_score':row.away_points,
           'neutral':int(bool(row.neutral_site)),'conf_game':int(bool(row.conference_game)),'venue':row.venue,'spread':row.spread,'over_under':row.over_under,
           'home_elo':row.home_elo,'away_elo':row.away_elo,'completed':row.completed}
    vdome = ti.dome.get(row.home_team, np.nan); rec['roof'] = 'dome' if vdome==1 else ('outdoors' if vdome==0 else None)
    rec['venue_lat'] = ti.latitude.get(row.home_team, np.nan); rec['venue_lon'] = ti.longitude.get(row.home_team, np.nan); rec['venue_tz'] = ti.timezone.get(row.home_team, None)
    rec['wx_wind']=np.nan; rec['wx_temp']=np.nan; rec['wx_source']='none'   # no historical kickoff weather in this feed
    for side, team, is_home in (('h',row.home_team,True),('a',row.away_team,False)):
        f={}
        e = ent(team)
        if (s,w,e) in R_idx.index: f.update(R_idx.loc[(s,w,e)].to_dict())
        f.update(std_feats(s,w,team))
        qbn = expected_starter(team, row.gameday); f['qb_name']=qbn
        if qbn: f.update(qb_form(team, qbn, row.gameday))
        f.update(rest_travel(row, team, is_home)); f['fbs'] = int(team in FBS); f['elo'] = row.home_elo if is_home else row.away_elo
        for k,v in f.items(): rec[f'{side}_{k}']=v
    rows.append(rec)
F = pd.DataFrame(rows); F.to_pickle(f'{OUT}/game_features.pkl')
meta = {'built_at': datetime.now(timezone.utc).isoformat(), 'seasons': SEASONS, 'pbp_last_game_date': str(pbp.gameday.max().date()), 'plays': int(len(pbp)), 'games': int(len(F)), 'played': int(F.played.sum())}
json.dump(meta, open(f'{OUT}/data_meta.json','w'), indent=1); print(meta)
ti.reset_index()[['school','abbreviation','conference','classification','venue_name','city','state','timezone','latitude','longitude','dome','color','logo']].to_json(f'{OUT}/teams.json', orient='records')
