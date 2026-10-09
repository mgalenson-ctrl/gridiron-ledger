"""Live feeds from ESPN's public JSON (no key): current betting lines with the opening line, and NFL injury statuses.

    python3 scripts/live_feeds.py nfl|college

Writes <sport>/out/live.json for this run and keeps two running logs that are committed to the repository between runs:
  <sport>/out/line_history.json   every change in spread / moneyline / total, with the time it was first seen
  nfl/out/injury_history.json     every change in a player's listed status, with the time it was first seen
Any network or parsing failure is recorded in live.json['errors']; the pipeline then falls back to the nflverse / CFBD lines.
"""
import json, os, sys, time, urllib.request
from datetime import datetime, timezone, timedelta

UA = {'User-Agent': 'Mozilla/5.0 (Hunch vs. Crunch personal research)', 'Accept': 'application/json'}
NOW = datetime.now(timezone.utc)
NOW_S = NOW.isoformat(timespec='seconds')


def get(url, tries=3):
    err = None
    for i in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=25) as r:
                return json.load(r)
        except Exception as e:  # noqa: BLE001 - recorded, then retried
            err = e; time.sleep(2 * (i + 1))
    raise err


def dig(d, *keys):
    for k in keys:
        if not isinstance(d, dict): return None
        d = d.get(k)
    return d


def num(x):
    """'-3' -> -3.0, '+140' -> 140.0, 'o49.5' -> 49.5, 'EVEN' -> 100.0, 'PK' -> 0.0"""
    if x is None: return None
    if isinstance(x, (int, float)): return float(x)
    s = str(x).strip().lower()
    if s in ('even', 'ev'): return 100.0
    if s in ('pk', 'pick', "pick'em"): return 0.0
    if s[:1] in ('o', 'u'): s = s[1:]
    try: return float(s.replace('+', ''))
    except ValueError: return None


def parse_odds(o):
    """ESPN odds object (scoreboard competitions[0].odds[0] or summary pickcenter[0]) -> home-perspective numbers.
    spread = home team's line (negative = home favored). ESPN's 'close' is the current line until kickoff."""
    if not isinstance(o, dict): return None
    cur_sp = num(dig(o, 'pointSpread', 'home', 'close', 'line'))
    if cur_sp is None: cur_sp = num(o.get('spread'))
    open_sp = num(dig(o, 'pointSpread', 'home', 'open', 'line'))
    ml_h = num(dig(o, 'moneyline', 'home', 'close', 'odds'));  ml_h = ml_h if ml_h is not None else num(dig(o, 'homeTeamOdds', 'moneyLine'))
    ml_a = num(dig(o, 'moneyline', 'away', 'close', 'odds'));  ml_a = ml_a if ml_a is not None else num(dig(o, 'awayTeamOdds', 'moneyLine'))
    out = {'provider': dig(o, 'provider', 'name'), 'details': o.get('details'),
           'spread_home': cur_sp, 'spread_home_open': open_sp,
           'ml_home': ml_h, 'ml_away': ml_a,
           'ml_home_open': num(dig(o, 'moneyline', 'home', 'open', 'odds')), 'ml_away_open': num(dig(o, 'moneyline', 'away', 'open', 'odds')),
           'total': num(dig(o, 'total', 'over', 'close', 'line')) or num(o.get('overUnder')),
           'total_open': num(dig(o, 'total', 'over', 'open', 'line')),
           'home_fav_at_open': dig(o, 'homeTeamOdds', 'favoriteAtOpen')}
    if all(out[k] is None for k in ('spread_home', 'ml_home', 'total')): return None
    return out


def load(path, default):
    try: return json.load(open(path))
    except Exception: return default


def record_line(hist, gid, odds):
    if not odds: return
    snap = {k: odds.get(k) for k in ('spread_home', 'ml_home', 'ml_away', 'total')}
    rows = hist.setdefault(str(gid), [])
    if not rows and odds.get('spread_home_open') is not None:
        rows.append({'t': None, 'opening': True, 'spread_home': odds['spread_home_open'], 'ml_home': odds.get('ml_home_open'),
                     'ml_away': odds.get('ml_away_open'), 'total': odds.get('total_open'), 'provider': odds.get('provider')})
    last = rows[-1] if rows else None
    if last is None or any(last.get(k) != v for k, v in snap.items()):
        rows.append({'t': NOW_S, **snap, 'provider': odds.get('provider')})


def nfl():
    import pandas as pd
    g = pd.read_csv('/home/claude/nflverse/nfldata/data/games.csv')
    g['gameday'] = pd.to_datetime(g.gameday)
    today = pd.Timestamp(NOW.date())
    up = g[g.result.isna() & (g.gameday >= today - pd.Timedelta(days=1)) & (g.gameday <= today + pd.Timedelta(days=9)) & g.espn.notna()]
    games, errors = {}, []
    lh = load('/home/claude/nflpredict/out/line_history.json', {})
    ih = load('/home/claude/nflpredict/out/injury_history.json', {})
    for r in up.itertuples():
        gid = r.game_id
        try:
            s = get(f'https://site.api.espn.com/apis/site/v2/sports/football/nfl/summary?event={int(r.espn)}')
        except Exception as e:  # noqa: BLE001
            errors.append(f'{gid}: {type(e).__name__}: {e}'); continue
        pc = s.get('pickcenter') or s.get('odds') or []
        odds = parse_odds(pc[0]) if pc else None
        side_of = {}
        for c in dig(s, 'header', 'competitions') and s['header']['competitions'][0].get('competitors', []) or []:
            side_of[str(dig(c, 'team', 'id'))] = c.get('homeAway')
        inj = {'home': [], 'away': []}
        for t in s.get('injuries') or []:
            side = side_of.get(str(dig(t, 'team', 'id')))
            if side not in inj: continue
            for e in t.get('injuries') or []:
                a = e.get('athlete') or {}
                inj[side].append({'espn_id': str(a.get('id')) if a.get('id') is not None else None, 'name': a.get('displayName') or a.get('fullName'),
                                  'pos': dig(a, 'position', 'abbreviation'), 'status': e.get('status') or dig(e, 'type', 'description'),
                                  'date': e.get('date'), 'detail': dig(e, 'details', 'type')})
        games[gid] = {'espn_id': int(r.espn), 'odds': odds, 'injuries': inj}
        record_line(lh, gid, odds)
        # injury status changes (first time each status is seen)
        for side, lst in inj.items():
            team = r.home_team if side == 'home' else r.away_team
            th = ih.setdefault(team, {})
            for p in lst:
                key = p['espn_id'] or p['name']
                ph = th.setdefault(key, {'name': p['name'], 'pos': p['pos'], 'changes': []})
                if not ph['changes'] or ph['changes'][-1]['status'] != p['status']:
                    ph['changes'].append({'status': p['status'], 'seen': NOW_S, 'espn_date': p['date']})
            listed = {p['espn_id'] or p['name'] for p in lst}
            for key, ph in th.items():   # dropped off the list = cleared
                if key not in listed and ph['changes'] and ph['changes'][-1]['status'] != 'Cleared' and up[(up.home_team == team) | (up.away_team == team)].shape[0]:
                    ph['changes'].append({'status': 'Cleared', 'seen': NOW_S, 'espn_date': None})
        time.sleep(0.3)
    json.dump(lh, open('/home/claude/nflpredict/out/line_history.json', 'w'), indent=0)
    json.dump(ih, open('/home/claude/nflpredict/out/injury_history.json', 'w'), indent=0)
    return games, errors, '/home/claude/nflpredict/out/live.json'


def college():
    games, errors = {}, []
    lh = load('/home/claude/cfbpredict/out/line_history.json', {})
    d0 = (NOW - timedelta(days=1)).strftime('%Y%m%d'); d1 = (NOW + timedelta(days=9)).strftime('%Y%m%d')
    try:
        sb = get(f'https://site.api.espn.com/apis/site/v2/sports/football/college-football/scoreboard?groups=80&limit=500&dates={d0}-{d1}')
        for ev in sb.get('events') or []:
            comp = (ev.get('competitions') or [{}])[0]
            odds = parse_odds((comp.get('odds') or [None])[0])
            gid = str(ev.get('id'))
            games[gid] = {'espn_id': ev.get('id'), 'odds': odds, 'injuries': None}
            record_line(lh, gid, odds)
    except Exception as e:  # noqa: BLE001
        errors.append(f'scoreboard: {type(e).__name__}: {e}')
    json.dump(lh, open('/home/claude/cfbpredict/out/line_history.json', 'w'), indent=0)
    return games, errors, '/home/claude/cfbpredict/out/live.json'


if __name__ == '__main__':
    sport = sys.argv[1]
    games, errors, path = nfl() if sport == 'nfl' else college()
    n_odds = sum(1 for v in games.values() if v.get('odds'))
    json.dump({'fetched_at': NOW_S, 'source': 'ESPN public site API (site.api.espn.com)', 'games': games, 'errors': errors,
               'n_games': len(games), 'n_with_odds': n_odds}, open(path, 'w'), indent=0)
    print(f'{sport}: {len(games)} games, {n_odds} with odds, {len(errors)} errors')
    for e in errors[:5]: print('  ', e)
