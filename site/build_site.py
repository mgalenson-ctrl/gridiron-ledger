"""Assemble the installable site in site/dist from the two data bundles and page templates.
A sport whose bundle is missing (pipeline failed, or offseason) keeps its previously published page if PREVIOUS_URL is set."""
import json, os, shutil, sys, urllib.request, html
from datetime import datetime, timezone
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIST = os.path.join(ROOT, 'site', 'dist'); STATIC = os.path.join(ROOT, 'site', 'static')
PREV = os.environ.get('PREVIOUS_URL', '').rstrip('/')
SPORTS = [('nfl', 'NFL', 'Hunch vs. Crunch NFL'), ('college', 'College', 'Hunch vs. Crunch College')]
FOOT = '''<footer class="gl-foot"><nav><a href="../">Home</a> · <a href="../nfl/">NFL</a> · <a href="../college/">College</a></nav>
<p>Statistical estimates for entertainment and curiosity. Not betting advice. Market lines appear only as an accuracy benchmark.</p>
<p>Data: nflverse (play-by-play, schedules, injuries, depth charts, snap counts), FTN Data charting via nflverse (CC BY-SA 4.0), cfbfastR / sportsdataverse, CollegeFootballData Elo, Open-Meteo forecasts. Not affiliated with the NFL, NCAA, or any team.</p></footer>
<style>.gl-foot{max-width:980px;margin:0 auto;padding:8px 16px 40px;font:12px/1.5 "IBM Plex Mono",ui-monospace,Menlo,monospace;color:var(--muted)}.gl-foot a{color:var(--accent)}.gl-foot p{margin:6px 0;max-width:80ch}</style>
<script>if('serviceWorker' in navigator){navigator.serviceWorker.register('../sw.js',{scope:'../'}).catch(()=>{})}</script>'''
def head(title, up):
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<title>{html.escape(title)}</title><link rel="manifest" href="{up}manifest.webmanifest"><meta name="theme-color" content="#1f7a4d">
<meta name="apple-mobile-web-app-capable" content="yes"><meta name="apple-mobile-web-app-title" content="Hunch vs. Crunch"><link rel="apple-touch-icon" href="{up}icons/icon-180.png"><link rel="icon" href="{up}icons/icon-192.png">
<style>:root{{padding-top:env(safe-area-inset-top,0px);padding-bottom:env(safe-area-inset-bottom,0px)}}body{{margin:0}}img{{max-width:100%}}[hidden]{{display:none!important}}</style></head><body>
'''
os.makedirs(DIST, exist_ok=True)
for f in os.listdir(STATIC):
    src = os.path.join(STATIC, f); dst = os.path.join(DIST, f)
    (shutil.copytree(src, dst, dirs_exist_ok=True) if os.path.isdir(src) else shutil.copy(src, dst))
cards = []; built = 0
for key, label, title in SPORTS:
    out = os.path.join(DIST, key); os.makedirs(out, exist_ok=True)
    bpath = os.path.join(ROOT, key, 'out', 'bundle.json'); tpl = open(os.path.join(ROOT, key, 'app_template.html')).read()
    if os.path.exists(bpath):
        b = json.load(open(bpath)); raw = json.dumps(b, separators=(',', ':')).replace('</', '<\\/')
        page = head(title, '../') + tpl.replace('__BUNDLE__', raw) + FOOT + '</body></html>'
        open(os.path.join(out, 'index.html'), 'w').write(page); built += 1
        top = sorted(b['games'], key=lambda g: -max(g['latest']['p_home'], 1 - g['latest']['p_home']))[:3]
        cards.append((key, label, f"Week {b['week']} · {len(b['games'])} games", b['generated_at'], [(g['latest']['pick'], max(g['latest']['p_home'], 1 - g['latest']['p_home'])) for g in top]))
    elif PREV:
        try:
            page = urllib.request.urlopen(f'{PREV}/{key}/', timeout=30).read().decode()
            open(os.path.join(out, 'index.html'), 'w').write(page); cards.append((key, label, 'Showing the last published picks', None, []))
            print(key, 'kept previous page')
        except Exception as e:
            print(key, 'no bundle and no previous page:', e)
    else:
        print(key, 'no bundle; skipped')
if not cards: sys.exit('nothing to publish')
def card(c):
    key, label, sub, gen, top = c
    picks = ''.join(f'<li><b>{html.escape(str(t))}</b><span>{p*100:.0f}%</span></li>' for t, p in top)
    when = f'<p class="when">Updated {html.escape(gen[:16].replace("T"," "))} UTC</p>' if gen else ''
    return f'<a class="card" href="{key}/"><h2>{label}</h2><p class="sub">{html.escape(sub)}</p>{"<ul>"+picks+"</ul>" if picks else ""}{when}<span class="go">Open picks</span></a>'
index = head('Hunch vs. Crunch', '') + '''<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wdth,wght@87.5,500;87.5,700;100,400;100,600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
:root{--bg:#f3f3ef;--surface:#fff;--line:#d9dad3;--fg:#17201b;--muted:#5e665f;--accent:#1f7a4d;--accent-fg:#fff;color-scheme:light}
@media (prefers-color-scheme:dark){:root{--bg:#121613;--surface:#1b211c;--line:#2d352f;--fg:#e8ebe6;--muted:#9aa49c;--accent:#4fbf85;--accent-fg:#0f1711;color-scheme:dark}}
*{box-sizing:border-box}body{background:var(--bg);color:var(--fg);font-family:"Archivo",system-ui,sans-serif;font-size:15px;line-height:1.45}
main{max-width:760px;margin:0 auto;padding-inline:16px;padding-block:28px 8px}
h1{font-size:30px;margin:0 0 4px;font-stretch:87.5%;letter-spacing:-.01em;text-wrap:balance}.lede{color:var(--muted);margin:0 0 20px;max-width:60ch}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:12px}
.card{display:grid;gap:6px;background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:16px;color:inherit;text-decoration:none}
.card:hover,.card:focus-visible{border-color:var(--accent);outline:none}.card h2{margin:0;font-size:22px;font-stretch:87.5%}
.sub,.when{margin:0;color:var(--muted);font:12px "IBM Plex Mono",ui-monospace,monospace}
.card ul{list-style:none;margin:6px 0;padding:0;display:grid;gap:4px}.card li{display:flex;justify-content:space-between;gap:12px;border-bottom:1px solid var(--line);padding-bottom:4px}
.card li span{font-family:"IBM Plex Mono",ui-monospace,monospace;font-variant-numeric:tabular-nums}
.go{justify-self:start;margin-top:6px;background:var(--accent);color:var(--accent-fg);border-radius:6px;padding:6px 12px;font-weight:600;font-size:14px}
.install{margin-top:22px;color:var(--muted);font-size:14px;max-width:62ch}
</style><main><h1>Hunch vs. Crunch</h1><p class="lede">Statistical picks for every NFL and FBS college game: a winner, a confidence level, a projected score and the reasons, refreshed every morning.</p>
<div class="grid">''' + ''.join(card(c) for c in cards) + '''</div>
<p class="install"><b>Add it to your phone:</b> on iPhone open this page in Safari, tap Share, then Add to Home Screen. On Android, open the browser menu and choose Install app.</p></main>
''' + FOOT.replace('href="../"', 'href="./"').replace('"../nfl/"', '"nfl/"').replace('"../college/"', '"college/"').replace("'../sw.js',{scope:'../'}", "'sw.js'") + '</body></html>'
open(os.path.join(DIST, 'index.html'), 'w').write(index)
json.dump({'built_at': datetime.now(timezone.utc).isoformat(timespec='seconds'), 'pages': [c[0] for c in cards]}, open(os.path.join(DIST, 'version.json'), 'w'))
print('site built:', [c[0] for c in cards], 'fresh bundles:', built)
