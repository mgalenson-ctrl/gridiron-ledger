"""Next Gen Stats (nflverse) -> pre-game team features: offense time-to-throw, rush yards over expected per carry, receiver separation,
YAC over expected, and the same allowed by each defense. Exponentially weighted (half-life 6 games) over prior games only."""
import numpy as np, pandas as pd
N='/tmp/nfl/data/'
F=pd.read_pickle('/home/claude/nflpredict/out/game_features.pkl')
sched=pd.concat([F[['game_id','season','week','home_team','away_team','gameday']].rename(columns={'home_team':'team','away_team':'opp'}),
                 F[['game_id','season','week','away_team','home_team','gameday']].rename(columns={'away_team':'team','home_team':'opp'})])
def wavg(d,val,w): 
    d=d[d[w]>0]; return d.groupby(['season','week','team_abbr']).apply(lambda g: np.average(g[val],weights=g[w])).rename(val)
p=pd.read_csv(N+'ngs_passing.csv.gz'); p=p[p.week>0]
r=pd.read_csv(N+'ngs_rushing.csv.gz'); r=r[r.week>0]
c=pd.read_csv(N+'ngs_receiving.csv.gz'); c=c[c.week>0]
for _d in (p,r,c): _d['team_abbr']=_d.team_abbr.replace({'LAR':'LA'})
print('abbr not in sched', set(p.team_abbr)-set(sched.team))
tw=pd.concat([wavg(p,'avg_time_to_throw','attempts'),wavg(p,'completion_percentage_above_expectation','attempts'),wavg(p,'aggressiveness','attempts'),
   wavg(r,'rush_yards_over_expected_per_att','rush_attempts'),wavg(c,'avg_separation','targets'),wavg(c,'avg_yac_above_expectation','receptions')],axis=1).reset_index().rename(columns={'team_abbr':'team'})
tw.columns=['season','week','team','ttt','cpoe','aggr','ryoe','sep','yacoe']
tw=tw.merge(sched,on=['season','week','team'],how='inner')
# defense allowed: opponent's offensive numbers
d=tw[['season','week','opp','cpoe','ryoe','sep','yacoe']].rename(columns={'opp':'team','cpoe':'dcpoe','ryoe':'dryoe','sep':'dsep','yacoe':'dyacoe'})
tw=tw.merge(d,on=['season','week','team'],how='left').sort_values(['team','gameday'])
feats=['ttt','cpoe','aggr','ryoe','sep','yacoe','dcpoe','dryoe','dsep','dyacoe']
for f in feats: tw['pre_'+f]=tw.groupby('team')[f].transform(lambda s: s.ewm(halflife=6,min_periods=3).mean().shift(1))
# map pregame values onto every scheduled game (including future): use last available ewm after each team's latest game
last=tw.groupby('team').apply(lambda g: g[feats].ewm(halflife=6,min_periods=3).mean().iloc[-1]).add_prefix('pre_')
rows=sched.merge(tw[['game_id','team']+['pre_'+f for f in feats]],on=['game_id','team'],how='left')
fut=rows['pre_ttt'].isna()&~rows.game_id.isin(F[F.played].game_id)
for f in feats: rows.loc[fut,'pre_'+f]=rows.loc[fut,'team'].map(last['pre_'+f])
out=F[['game_id','home_team','away_team']].copy()
for side,col in [('h','home_team'),('a','away_team')]:
    m=rows.rename(columns={'team':col}).drop(columns=['opp','season','week','gameday'])
    m=m.rename(columns={'pre_'+f:f'{side}_ngs_{f}' for f in feats}); out=out.merge(m,on=['game_id',col],how='left')
out.to_pickle('/home/claude/nflpredict/out/ngs_features.pkl'); print(out.shape, out.filter(like='ngs').notna().mean().round(2).to_dict())
