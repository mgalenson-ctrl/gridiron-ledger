"""Fetch kickoff-hour forecasts from Open-Meteo for the current week's non-roofed games -> <sport>/out/forecasts.json
Usage: python3 scripts/weather_fetch.py nfl|college
Never invents values: a failed or out-of-range fetch is recorded with status 'unavailable'/'out_of_range' and the app labels it."""
import json, sys, time, urllib.request, urllib.parse
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
import pandas as pd

API = 'https://api.open-meteo.com/v1/forecast'
HOURLY = 'temperature_2m,wind_speed_10m,wind_gusts_10m,precipitation_probability'

def parse_open_meteo(payload, kickoff_utc, hours=3):
    """payload: Open-Meteo JSON with hourly.time in GMT ('YYYY-MM-DDTHH:MM'). Returns dict of kickoff-window values or None."""
    h = payload.get('hourly') or {}
    times = h.get('time') or []
    key = kickoff_utc.astimezone(timezone.utc).strftime('%Y-%m-%dT%H:00')
    if key not in times: return None
    i = times.index(key); sl = slice(i, min(i+hours, len(times)))
    def vals(name): return [v for v in (h.get(name) or [])[sl] if v is not None]
    t, w, g, p = vals('temperature_2m'), vals('wind_speed_10m'), vals('wind_gusts_10m'), vals('precipitation_probability')
    if not t or not w: return None
    return {'temp_f': sum(t)/len(t), 'wind_mph': sum(w)/len(w), 'gust_mph': (max(g) if g else None), 'pop_pct': (max(p) if p else None),
            'valid_for': key+'Z', 'window_hours': len(t)}

def fetch(lat, lon, kickoff_utc):
    d0 = kickoff_utc.astimezone(timezone.utc).date(); d1 = (kickoff_utc.astimezone(timezone.utc)+timedelta(hours=4)).date()
    q = urllib.parse.urlencode({'latitude': round(lat,4), 'longitude': round(lon,4), 'hourly': HOURLY, 'temperature_unit': 'fahrenheit',
                                'wind_speed_unit': 'mph', 'timezone': 'GMT', 'start_date': str(d0), 'end_date': str(d1)})
    req = urllib.request.Request(f'{API}?{q}', headers={'User-Agent': 'gridiron-ledger/1.0'})
    with urllib.request.urlopen(req, timeout=30) as r: return json.loads(r.read().decode())

def games(sport):
    if sport == 'nfl':
        F = pd.read_pickle('/home/claude/nflpredict/out/game_features.pkl'); V = json.load(open('/home/claude/nflpredict/out/venues.json'))
        s = int(F.season.max()); wk = int(F[(F.season==s)&(~F.played)].week.min()); g = F[(F.season==s)&(F.week==wk)]
        out = []
        for r in g.itertuples():
            roof = r.roof if isinstance(r.roof, str) else None
            if roof in ('dome','closed'): continue
            ko = datetime.fromisoformat(f'{r.gameday.date()}T{r.gametime}:00').replace(tzinfo=ZoneInfo('America/New_York')).astimezone(timezone.utc)
            intl = V['intl'].get(r.stadium); v = intl or V['team_venues'][r.home_team]
            out.append((str(r.game_id), float(v['lat']), float(v['lon']), ko, roof))
        return s, wk, out, '/home/claude/nflpredict/out/forecasts.json'
    F = pd.read_pickle('/home/claude/cfbpredict/out/game_features.pkl')
    s = int(F.season.max()); m = (F.season==s)&(~F.played)&(F.season_type=='regular'); wk = int(F[m].week.min()); g = F[(F.season==s)&(F.week==wk)&(F.season_type=='regular')]
    out = []
    for r in g.itertuples():
        roof = r.roof if isinstance(r.roof, str) else None
        if roof == 'dome': continue
        ko = datetime.fromisoformat(r.start_utc).astimezone(timezone.utc)
        if pd.isna(r.venue_lat): out.append((str(int(r.game_id)), None, None, ko, roof)); continue
        out.append((str(int(r.game_id)), float(r.venue_lat), float(r.venue_lon), ko, roof))
    return s, wk, out, '/home/claude/cfbpredict/out/forecasts.json'

if __name__ == '__main__':
    sport = sys.argv[1]; season, week, gl, path = games(sport)
    now = datetime.now(timezone.utc); res = {}
    for gid, lat, lon, ko, roof in gl:
        rec = {'game_id': gid, 'source': 'open-meteo', 'roof': roof, 'kickoff_utc': ko.isoformat(), 'fetched_at': now.isoformat(timespec='seconds'), 'status': 'unavailable'}
        if lat is not None:
            try:
                v = parse_open_meteo(fetch(lat, lon, ko), ko)
                if v is None: rec['status'] = 'out_of_range'
                else: rec.update(v); rec['status'] = 'ok'; rec['forecast_created'] = None
            except Exception as e:
                rec['error'] = str(e)[:200]
            time.sleep(0.3)
        res[gid] = rec
    json.dump({'season': season, 'week': week, 'fetched_at': now.isoformat(timespec='seconds'), 'forecasts': res}, open(path, 'w'), indent=1)
    ok = sum(1 for v in res.values() if v['status']=='ok'); print(f'{sport}: week {week}, {ok}/{len(res)} forecasts ok')
