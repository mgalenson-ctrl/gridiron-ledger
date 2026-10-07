"""
Walk-forward training / evaluation and current-week prediction.

Target: home_win (1 / 0, ties = 0.5 for Brier; ties excluded from winner accuracy; a predicted pick is
"correct" only if that team won outright).  The model outputs P(home wins | no tie); ties are ~0.2% of games.
Model: L2-regularised logistic regression on (home - away) feature differences + home-field indicator.
Walk-forward: for each (season, week) from 2022 wk1 on, fit on every played game strictly before that week
(all prior seasons + earlier weeks of the same season), predict that week's games.
"""
import numpy as np, pandas as pd, json, hashlib, os, sys
from datetime import datetime, timezone
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

OUT = '/home/claude/nflpredict/out'
F = pd.read_pickle(f'{OUT}/game_features.pkl')
F = F[F.game_type.isin(['REG','WC','DIV','CON','SB'])].copy()

# ---------------------------------------------------------------- feature groups (diff = home - away)
GROUPS = {
 'baseline': ['adj_off_off_epa','adj_def_off_epa'],
 'efficiency': ['adj_off_off_sr','adj_def_off_sr','adj_off_pass_epa','adj_def_pass_epa','adj_off_rush_epa','adj_def_rush_epa',
                'adj_off_ppd','adj_def_ppd','adj_off_explosive_rate','adj_def_explosive_rate'],
 'form_l5': ['l5_off_epa','l5_def_epa','l5_off_sr','l5_def_sr','l5_ppd'],
 'qb': ['qb_epa','qb_cpoe','qb_int_rate','qb_sack_rate','qb_scr_epa_per_db','qb_change','qb_db_hist'],
 'situational': ['third_rate','def_third_rate','rz_td_rate','def_rz_td_rate','fourth_rate','short_rate'],
 'special_discipline': ['fg_rate','fg_long_rate','st_epa_per_game','to_per_game','takeaways_per_game','pen_per_game'],
 'matchup': ['pp_vs_rush','rush_vs_rundef','pass_vs_passdef','explosive_edge'],
 'rest_travel': ['rest','short_week','bye','prev_ot','consec_road_prior','travel_miles','tz_shift'],
 'coaching': ['coach_tenure','coach_changed','fourth_go_rate','pace_secs_per_play','pass_oe'],
 'rivalry': ['prev_meeting_margin','div_game_flag'],
 'injuries': ['inj_starters_out','inj_starters_q','inj_qb_out','inj_ol_out'],
 'weather': ['wx_wind','wx_cold','wx_wind_x_passlean','wx_cold_x_awaydome','wx_wind_x_fg'],
 'availability': ['av_off_out','av_def_out','av_ol_out','av_skill_out','av_dl_out','av_lb_out','av_db_out','av_off_q','av_def_q'],
}
GAME_LEVEL = {'div_game_flag','wx_wind','wx_cold','wx_wind_x_passlean','wx_cold_x_awaydome','wx_wind_x_fg'}
SIDE_ONLY = {'div_game_flag'}  # game-level, not diffed

def build_X(df):
    X = pd.DataFrame(index=df.index)
    # derived
    for s in ('h','a'):
        df[f'{s}_short_week'] = (df[f'{s}_rest'] < 7).astype(float)
        df[f'{s}_bye'] = (df[f'{s}_rest'] >= 13).astype(float)
    # matchup interactions: our pass protection (low sack rate) vs their pass rush (high sack rate), etc.
    df['h_pp_vs_rush'] = -(df.h_sack_rate_off - df.a_sack_rate_def); df['a_pp_vs_rush'] = -(df.a_sack_rate_off - df.h_sack_rate_def)
    df['h_rush_vs_rundef'] = df.h_adj_off_rush_epa + df.a_adj_def_rush_epa*(-1); df['a_rush_vs_rundef'] = df.a_adj_off_rush_epa - df.h_adj_def_rush_epa
    df['h_pass_vs_passdef'] = df.h_adj_off_pass_epa - df.a_adj_def_pass_epa; df['a_pass_vs_passdef'] = df.a_adj_off_pass_epa - df.h_adj_def_pass_epa
    df['h_explosive_edge'] = df.h_adj_off_explosive_rate - df.a_adj_def_explosive_rate; df['a_explosive_edge'] = df.a_adj_off_explosive_rate - df.h_adj_def_explosive_rate
    # weather (game-level; zero under a roof). Historical rows use actual kickoff conditions as the proxy for the forecast.
    df['wx_cold'] = (45 - df.wx_temp).clip(lower=0)
    df['wx_wind_x_passlean'] = df.wx_wind * (df.h_pass_oe - df.a_pass_oe) / 10.0   # wind hurts the more pass-leaning side
    df['wx_cold_x_awaydome'] = df.wx_cold * df.a_home_dome                          # cold vs a visiting dome team
    df['wx_wind_x_fg'] = df.wx_wind * (df.h_fg_long_rate - df.a_fg_long_rate)
    for g, feats in GROUPS.items():
        for f in feats:
            if f == 'div_game_flag':
                X['div_game'] = df.div_game.fillna(0).astype(float)
            elif f in GAME_LEVEL:
                X[f] = df[f].astype(float)
            else:
                X[f'd_{f}'] = df[f'h_{f}'] - df[f'a_{f}']
    X['home_adv'] = 1.0 - df.neutral.astype(float)
    return X

def cols_for(groups):
    c = ['home_adv']
    for g in groups:
        for f in GROUPS[g]:
            c.append('div_game' if f=='div_game_flag' else (f if f in GAME_LEVEL else f'd_{f}'))
    return c

X_all = build_X(F)
y_all = F.home_win

def fit(Xtr, ytr, C):
    # ties -> drop from fitting (binary target); medians for NaN imputation learned on train only
    m = ytr.isin([0,1])
    med = Xtr[m].median()
    model = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000))
    model.fit(Xtr[m].fillna(med), ytr[m].astype(int))
    return model, med

def brier(p, y): return float(np.mean((p - y)**2))
def devig(hml, aml):
    def imp(ml): return 100/(ml+100) if ml>0 else -ml/(-ml+100)
    ph, pa = imp(hml), imp(aml); return ph/(ph+pa)

def walk_forward(groups, C=0.3, start=(2022,1)):
    cols = cols_for(groups)
    preds = []
    weeks = F[F.played][['season','week']].drop_duplicates().sort_values(['season','week'])
    for s, w in weeks.itertuples(index=False):
        if (s, w) < start: continue
        tr = F.played & ((F.season < s) | ((F.season == s) & (F.week < w)))
        te = F.played & (F.season == s) & (F.week == w)
        if te.sum()==0: continue
        model, med = fit(X_all.loc[tr, cols], y_all[tr], C)
        p = model.predict_proba(X_all.loc[te, cols].fillna(med))[:,1]
        preds.append(pd.DataFrame({'game_id': F.loc[te,'game_id'].values, 'p_home': p}))
    P = pd.concat(preds).merge(F[['game_id','season','week','home_win','home_moneyline','away_moneyline','home_team','away_team']], on='game_id')
    return P

def evaluate(P):
    P = P.copy()
    P['mkt'] = [devig(h,a) if pd.notna(h) and pd.notna(a) else np.nan for h,a in zip(P.home_moneyline, P.away_moneyline)]
    nt = P[P.home_win.isin([0,1])]
    res = {
        'n': int(len(P)), 'n_decided': int(len(nt)),
        'accuracy': float(((nt.p_home>0.5)==(nt.home_win==1)).mean()),
        'brier': brier(P.p_home, P.home_win),
        'logloss': float(-np.mean(nt.home_win*np.log(np.clip(nt.p_home,1e-6,1))+(1-nt.home_win)*np.log(np.clip(1-nt.p_home,1e-6,1)))),
        'home_pick_accuracy': float((nt.home_win==1).mean()),
        'home_pick_brier': brier(np.full(len(P), (nt.home_win==1).mean()), P.home_win),
    }
    m = nt[nt.mkt.notna()]
    res['market_accuracy'] = float(((m.mkt>0.5)==(m.home_win==1)).mean())
    res['market_brier'] = brier(P[P.mkt.notna()].mkt, P[P.mkt.notna()].home_win)
    res['agree_with_market'] = float(((m.mkt>0.5)==(m.p_home>0.5)).mean())
    # calibration bins
    bins = np.linspace(0,1,11)
    P['bin'] = pd.cut(P.p_home, bins, include_lowest=True)
    cal = P.groupby('bin', observed=True).agg(n=('p_home','size'), pred=('p_home','mean'), actual=('home_win','mean')).reset_index()
    res['calibration'] = [{'lo':float(b.left),'hi':float(b.right),'n':int(n),'pred':float(p),'actual':float(a)} for b,n,p,a in cal.itertuples(index=False)]
    # by season
    res['by_season'] = {int(s): {'n':int(len(g)), 'accuracy': float(((g.p_home>0.5)==(g.home_win==1))[g.home_win.isin([0,1])].mean()),
                                  'brier': brier(g.p_home, g.home_win),
                                  'market_accuracy': float(((g.mkt>0.5)==(g.home_win==1))[g.home_win.isin([0,1])&g.mkt.notna()].mean())}
                        for s,g in P.groupby('season')}
    return res, P

if __name__ == '__main__':
    results = {}
    # Baseline progression (ablation): start simple, add groups one at a time
    ladder = [
        ('home_only', []),
        ('baseline_adj_epa', ['baseline']),
        ('+efficiency', ['baseline','efficiency']),
        ('+qb', ['baseline','efficiency','qb']),
        ('+form_l5', ['baseline','efficiency','qb','form_l5']),
        ('+situational', ['baseline','efficiency','qb','situational']),
        ('+special_discipline', ['baseline','efficiency','qb','special_discipline']),
        ('+matchup', ['baseline','efficiency','qb','matchup']),
        ('+rest_travel', ['baseline','efficiency','qb','rest_travel']),
        ('+coaching', ['baseline','efficiency','qb','coaching']),
        ('+rivalry', ['baseline','efficiency','qb','rivalry']),
        ('+weather', ['baseline','efficiency','qb','weather']),
        ('final+weather', ['baseline','qb','special_discipline','matchup','weather']),
        ('all_early', [g for g in GROUPS if g!='injuries']),
        ('final+weather+injuries', ['baseline','qb','special_discipline','matchup','weather','injuries']),
        ('final+weather+availability', ['baseline','qb','special_discipline','matchup','weather','availability']),
        ('final+weather+inj+avail', ['baseline','qb','special_discipline','matchup','weather','injuries','availability']),
        ('all_updated(+injuries)', [g for g in GROUPS if g!='availability']),
    ]
    allP = {}
    for name, groups in ladder:
        P = walk_forward(groups)
        res, P = evaluate(P)
        results[name] = {'groups': groups, 'n_features': len(cols_for(groups)), **res}
        allP[name] = P
        print(f"{name:28s} n={res['n']:4d} acc={res['accuracy']:.3f} brier={res['brier']:.4f} ll={res['logloss']:.4f} | home={res['home_pick_accuracy']:.3f} mkt_acc={res['market_accuracy']:.3f} mkt_brier={res['market_brier']:.4f}")
    json.dump(results, open(f'{OUT}/ablation_results.json','w'), indent=1)
    pd.to_pickle(allP, f'{OUT}/walkforward_preds.pkl')


# ---------------------------------------------------------------- score model: ridge on margin (home - away) and total points
from sklearn.linear_model import Ridge
def build_T(df):
    T = pd.DataFrame(index=df.index)
    for f in ['adj_off_off_epa','adj_def_off_epa','adj_off_ppd','adj_def_ppd','pace_secs_per_play','pass_oe','adj_off_explosive_rate','adj_def_explosive_rate','qb_epa','st_epa_per_game','fg_rate']:
        T[f's_{f}'] = df[f'h_{f}'] + df[f'a_{f}']
    T['roofed'] = df.roof.isin(['dome','closed']).astype(float)
    T['wx_wind'] = df.wx_wind.astype(float); T['wx_cold'] = (45-df.wx_temp).clip(lower=0).astype(float)
    T['div_game'] = df.div_game.fillna(0).astype(float)
    return T
T_all = build_T(F)
def fit_reg(X, y, alpha):
    m = y.notna(); med = X[m].median()
    model = make_pipeline(StandardScaler(), Ridge(alpha=alpha)); model.fit(X[m].fillna(med), y[m]); return model, med
def walk_forward_scores(margin_cols, alpha=30.0, start=(2022,1)):
    out=[]
    weeks = F[F.played][['season','week']].drop_duplicates().sort_values(['season','week'])
    XM = X_all[margin_cols]
    for s,w in weeks.itertuples(index=False):
        if (s,w) < start: continue
        tr = F.played & ((F.season<s)|((F.season==s)&(F.week<w))); te = F.played & (F.season==s)&(F.week==w)
        if te.sum()==0: continue
        mm, mmed = fit_reg(XM[tr], F.result[tr], alpha); tm, tmed = fit_reg(T_all[tr], F.total[tr], alpha)
        out.append(pd.DataFrame({'game_id':F.loc[te,'game_id'].values, 'pred_margin': mm.predict(XM[te].fillna(mmed)), 'pred_total': tm.predict(T_all[te].fillna(tmed))}))
    P = pd.concat(out).merge(F[['game_id','season','week','result','total','spread_line','total_line']], on='game_id')
    return P
