"""Preseason college data from CollegeFootballData.com (needs the CFBD_API_KEY repository secret).

Writes /tmp/cfb/data/cfbd_preseason.csv with one row per (season, team):
  talent   247Sports team talent composite (roster recruiting strength), known before the season
  ret_ppa  share of last season's offensive production (PPA) returning, known before the season
Both are fixed before week 1, so they cannot leak results. Without a key the file is not written and
the model runs exactly as before.
"""
import csv, json, os, sys, time, urllib.request

KEY = os.environ.get('CFBD_API_KEY', '').strip()
OUT = '/tmp/cfb/data/cfbd_preseason.csv'
BASE = 'https://api.collegefootballdata.com'


def get(path):
    req = urllib.request.Request(BASE + path, headers={'Authorization': f'Bearer {KEY}', 'Accept': 'application/json',
                                                       'User-Agent': 'Hunch vs. Crunch personal research'})
    err = None
    for i in range(3):
        try:
            with urllib.request.urlopen(req, timeout=40) as r:
                return json.load(r)
        except Exception as e:  # noqa: BLE001
            err = e; time.sleep(3 * (i + 1))
    raise err


def pick(d, *names):
    low = {k.lower(): v for k, v in d.items()}
    for n in names:
        if n.lower() in low and low[n.lower()] is not None:
            return low[n.lower()]
    return None


if __name__ == '__main__':
    if not KEY:
        print('CFBD_API_KEY not set; skipping preseason data'); sys.exit(0)
    seasons = [int(s) for s in (sys.argv[1:] or ['2023', '2024', '2025', '2026'])]
    rows = {}
    for y in seasons:
        try:
            for t in get(f'/talent?year={y}'):
                team = pick(t, 'team', 'school')
                if team: rows.setdefault((y, team), {})['talent'] = pick(t, 'talent')
        except Exception as e:  # noqa: BLE001
            print(f'talent {y} failed: {type(e).__name__}: {e}')
        try:
            for t in get(f'/player/returning?year={y}'):
                team = pick(t, 'team', 'school')
                if team: rows.setdefault((y, team), {})['ret_ppa'] = pick(t, 'percentPPA', 'percent_ppa')
        except Exception as e:  # noqa: BLE001
            print(f'returning production {y} failed: {type(e).__name__}: {e}')
        time.sleep(0.5)
    if not rows:
        print('no preseason data returned'); sys.exit(0)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['season', 'team', 'talent', 'ret_ppa'])
        for (y, team), v in sorted(rows.items()):
            w.writerow([y, team, v.get('talent'), v.get('ret_ppa')])
    n_t = sum(1 for v in rows.values() if v.get('talent') is not None); n_r = sum(1 for v in rows.values() if v.get('ret_ppa') is not None)
    print(f'preseason rows {len(rows)}: talent {n_t}, returning production {n_r}')
