"""Walk-forward training for college football. Same protocol as the NFL model: L2 logistic regression on home-minus-away
feature differences; each week predicted by a model fit only on games before it. Evaluation from 2024 week 1."""
import numpy as np, pandas as pd, json
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from scipy.stats import norm
OUT='/home/claude/cfbpredict/out'
F = pd.read_pickle(f'{OUT}/game_features.pkl')

GROUPS = {
 'baseline': ['adj_off_off_epa','adj_def_off_epa'],
 'efficiency': ['adj_off_off_sr','adj_def_off_sr','adj_off_pass_epa','adj_def_pass_epa','adj_off_rush_epa','adj_def_rush_epa','adj_off_ppd','adj_def_ppd','adj_off_explosive_rate','adj_def_explosive_rate'],
 'form_l3': ['l3_off_epa','l3_def_epa','l3_ppd'],
 'qb': ['qb_epa','qb_cmp','qb_int_rate','qb_sack_rate','qb_db_hist'],
 'situational': ['third_rate','def_third_rate','rz_td_rate','def_rz_td_rate','fourth_rate'],
 'special_discipline': ['fg_rate','to_per_game','takeaways_per_game','pen_per_game'],
 'matchup': ['pp_vs_rush','rush_vs_rundef','pass_vs_passdef','explosive_edge'],
 'rest_travel': ['rest','bye','travel_miles'],
 'elo': ['elo'],
 'level': ['fbs'],
 'weather': ['wx_wind','wx_cold'],
 'conf_game': ['conf_game_flag'],
}
GAME_LEVEL = {'conf_game_flag','wx_wind','wx_cold'}
def build_X(df):
    X = pd.DataFrame(index=df.index)
    df['h_pp_vs_rush'] = -(df.h_sack_rate_off - df.a_sack_rate_def); df['a_pp_vs_rush'] = -(df.a_sack_rate_off - df.h_sack_rate_def)
    df['h_rush_vs_rundef'] = df.h_adj_off_rush_epa - df.a_adj_def_rush_epa; df['a_rush_vs_rundef'] = df.a_adj_off_rush_epa - df.h_adj_def_rush_epa
    df['h_pass_vs_passdef'] = df.h_adj_off_pass_epa - df.a_adj_def_pass_epa; df['a_pass_vs_passdef'] = df.a_adj_off_pass_epa - df.h_adj_def_pass_epa
    df['h_explosive_edge'] = df.h_adj_off_explosive_rate - df.a_adj_def_explosive_rate; df['a_explosive_edge'] = df.a_adj_off_explosive_rate - df.h_adj_def_explosive_rate
    df['wx_cold'] = (45 - df.wx_temp).clip(lower=0)
    for g, feats in GROUPS.items():
        for f in feats:
            if f=='conf_game_flag': X['conf_game'] = df.conf_game.astype(float)
            elif f in GAME_LEVEL: X[f] = df[f].astype(float)
            else: X[f'd_{f}'] = df[f'h_{f}'] - df[f'a_{f}']
    X['home_adv'] = 1.0 - df.neutral.astype(float)
    return X
def cols_for(groups):
    c=['home_adv']
    for g in groups:
        for f in GROUPS[g]: c.append('conf_game' if f=='conf_game_flag' else (f if f in GAME_LEVEL else f'd_{f}'))
    return c
X_all = build_X(F); y_all = F.home_win

def fit(Xtr, ytr, C):
    m = ytr.isin([0,1]); med = Xtr[m].median()
    model = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000)); model.fit(Xtr[m].fillna(med), ytr[m].astype(int)); return model, med
def brier(p,y): return float(np.mean((p-y)**2))
SPREAD_SD = 16.0   # points; converts a CFB spread to a win probability for the benchmark (normal approximation)
def spread_prob(spread): return norm.cdf(-spread/SPREAD_SD)
def elo_prob(h,a): return 1/(1+10**(-(h-a)/400))

def walk_forward(groups, C=0.1, start=(2024,1)):
    cols = cols_for(groups); preds=[]
    weeks = F[F.played][['season','wk']].drop_duplicates().sort_values(['season','wk'])
    for s,w in weeks.itertuples(index=False):
        if (s,w)<start: continue
        tr = F.played & ((F.season<s)|((F.season==s)&(F.wk<w))); te = F.played & (F.season==s)&(F.wk==w)
        if te.sum()==0 or tr.sum()<200: continue
        model, med = fit(X_all.loc[tr,cols], y_all[tr], C)
        preds.append(pd.DataFrame({'game_id':F.loc[te,'game_id'].values,'p_home':model.predict_proba(X_all.loc[te,cols].fillna(med))[:,1]}))
    return pd.concat(preds).merge(F[['game_id','season','week','wk','home_win','spread','home_elo','away_elo','home_team','away_team','home_div','away_div']], on='game_id')

def evaluate(P):
    P=P.copy(); P['mkt'] = spread_prob(P.spread); P['elo_p'] = elo_prob(P.home_elo, P.away_elo)
    nt = P[P.home_win.isin([0,1])]
    res = {'n':int(len(P)), 'accuracy': float(((nt.p_home>0.5)==(nt.home_win==1)).mean()), 'brier': brier(P.p_home,P.home_win),
           'logloss': float(-np.mean(nt.home_win*np.log(np.clip(nt.p_home,1e-6,1))+(1-nt.home_win)*np.log(np.clip(1-nt.p_home,1e-6,1)))),
           'home_pick_accuracy': float((nt.home_win==1).mean()), 'home_pick_brier': brier(np.full(len(P),(nt.home_win==1).mean()), P.home_win)}
    m = nt[nt.mkt.notna()]; res['market_accuracy'] = float(((m.mkt>0.5)==(m.home_win==1)).mean()); res['market_brier'] = brier(P[P.mkt.notna()].mkt, P[P.mkt.notna()].home_win); res['market_n']=int(len(m))
    e = nt[nt.elo_p.notna()]; res['elo_accuracy'] = float(((e.elo_p>0.5)==(e.home_win==1)).mean()); res['elo_brier'] = brier(P[P.elo_p.notna()].elo_p, P[P.elo_p.notna()].home_win)
    res['agree_with_market'] = float(((m.mkt>0.5)==(m.p_home>0.5)).mean())
    bins=np.linspace(0,1,11); P['bin']=pd.cut(P.p_home,bins,include_lowest=True)
    cal=P.groupby('bin',observed=True).agg(n=('p_home','size'),pred=('p_home','mean'),actual=('home_win','mean')).reset_index()
    res['calibration']=[{'lo':float(b.left),'hi':float(b.right),'n':int(n),'pred':float(p),'actual':float(a)} for b,n,p,a in cal.itertuples(index=False)]
    res['by_season']={int(s):{'n':int(len(g)),'accuracy':float(((g.p_home>0.5)==(g.home_win==1))[g.home_win.isin([0,1])].mean()),'brier':brier(g.p_home,g.home_win),
                              'market_accuracy': float(((g.mkt>0.5)==(g.home_win==1))[g.home_win.isin([0,1])&g.mkt.notna()].mean())} for s,g in P.groupby('season')}
    # FBS-vs-FBS only subset
    ff = nt[(nt.home_div=='fbs')&(nt.away_div=='fbs')]; res['fbs_only'] = {'n':int(len(ff)),'accuracy':float(((ff.p_home>0.5)==(ff.home_win==1)).mean()),'brier':brier(ff.p_home,ff.home_win),
                                                                              'market_accuracy': float(((ff.mkt>0.5)==(ff.home_win==1))[ff.mkt.notna()].mean())}
    return res, P

def build_T(df):
    T = pd.DataFrame(index=df.index)
    for f in ['adj_off_off_epa','adj_def_off_epa','adj_off_ppd','adj_def_ppd','plays_per_game','pass_rate','adj_off_explosive_rate','adj_def_explosive_rate','qb_epa','fg_rate']:
        T[f's_{f}'] = df[f'h_{f}'] + df[f'a_{f}']
    T['roofed'] = (df.roof=='dome').astype(float)
    T['conf_game']=df.conf_game.astype(float); T['d_fbs'] = df.h_fbs - df.a_fbs
    return T
T_all = build_T(F)
def fit_reg(X,y,alpha):
    m=y.notna(); med=X[m].median(); model=make_pipeline(StandardScaler(), Ridge(alpha=alpha)); model.fit(X[m].fillna(med), y[m]); return model, med
def walk_forward_scores(margin_cols, alpha=100.0, start=(2024,1)):
    out=[]; XM=X_all[margin_cols]
    weeks = F[F.played][['season','wk']].drop_duplicates().sort_values(['season','wk'])
    for s,w in weeks.itertuples(index=False):
        if (s,w)<start: continue
        tr = F.played & ((F.season<s)|((F.season==s)&(F.wk<w))); te = F.played & (F.season==s)&(F.wk==w)
        if te.sum()==0 or tr.sum()<200: continue
        mm,mmed=fit_reg(XM[tr],F.result[tr],alpha); tm,tmed=fit_reg(T_all[tr],F.total[tr],alpha)
        out.append(pd.DataFrame({'game_id':F.loc[te,'game_id'].values,'pred_margin':mm.predict(XM[te].fillna(mmed)),'pred_total':tm.predict(T_all[te].fillna(tmed))}))
    return pd.concat(out).merge(F[['game_id','season','wk','result','total','spread','over_under']], on='game_id')

if __name__=='__main__':
    ladder=[('home_only',[]),('baseline_adj_epa',['baseline']),('+efficiency',['baseline','efficiency']),('+qb',['baseline','qb']),('+form_l3',['baseline','form_l3']),
            ('+situational',['baseline','situational']),('+special_discipline',['baseline','special_discipline']),('+matchup',['baseline','matchup']),
            ('+rest_travel',['baseline','rest_travel']),('+level',['baseline','level']),('+conf_game',['baseline','conf_game']),('+elo',['baseline','elo']),
            ('elo_only',['elo']),('base+qb+sd+matchup+level',['baseline','qb','special_discipline','matchup','level']),
            ('base+qb+sd+matchup+level+elo',['baseline','qb','special_discipline','matchup','level','elo']),
            ('all',[g for g in GROUPS if g!='weather'])]
    results={}; allP={}
    for name,groups in ladder:
        P=walk_forward(groups); res,P=evaluate(P); results[name]={'groups':groups,'n_features':len(cols_for(groups)),**res}; allP[name]=P
        print(f"{name:32s} n={res['n']:4d} acc={res['accuracy']:.3f} brier={res['brier']:.4f} | home={res['home_pick_accuracy']:.3f} mkt={res['market_accuracy']:.3f}/{res['market_brier']:.4f} (n={res['market_n']}) elo={res['elo_accuracy']:.3f}/{res['elo_brier']:.4f} | fbs-only acc={res['fbs_only']['accuracy']:.3f} mkt={res['fbs_only']['market_accuracy']:.3f}")
    json.dump(results, open(f'{OUT}/ablation_results.json','w'), indent=1); pd.to_pickle(allP, f'{OUT}/walkforward_preds.pkl')
