# Gridiron Ledger

Statistical picks for NFL and FBS college football, published as an installable website and refreshed every morning by GitHub Actions.

## What is here
- `nfl/` and `college/` — the two prediction pipelines (features, walk-forward training, prediction) and each page template.
- `scripts/` — `run_nfl.sh`, `run_college.sh` (download data and run a pipeline) and `weather_fetch.py` (Open-Meteo kickoff forecasts).
- `site/` — `build_site.py` turns the two data bundles into `site/dist` (home page, `/nfl/`, `/college/`, install manifest, offline worker, icons).
- `.github/workflows/daily.yml` — runs everything daily at 11:30 UTC and publishes to GitHub Pages. Run it by hand from the Actions tab at any time.

## One-time setup
1. Repository Settings → Pages → Build and deployment → Source: **GitHub Actions**.
2. Actions tab → "Daily refresh and publish" → Run workflow. The first run takes about 10 minutes.
3. The site address appears on the finished run and under Settings → Pages.

## Custom domain
Settings → Pages → Custom domain, enter the domain, then at the domain registrar add a CNAME record for `www` pointing to `<username>.github.io`
(for a bare domain, the four A records GitHub lists). Tick "Enforce HTTPS" once it is available.

## How it behaves
- A sport whose pipeline fails, or that is out of season, keeps its previously published page; the run shows a warning.
- `nfl/out/predictions_log.json` and `college/out/predictions_log.json` are committed daily so the first prediction for each game is preserved.
- Per-run input snapshots are attached to each workflow run for 90 days.
- GitHub pauses scheduled workflows after 60 days without repository activity; the daily log commit keeps it active in season.

## Notes
- Forecasts: Open-Meteo, free for non-commercial use. A commercial product needs their paid plan or another provider.
- Data sources have their own terms (nflverse, FTN Data CC BY-SA 4.0, cfbfastR/sportsdataverse, CollegeFootballData). Review them before any commercial use.
- Estimates only. Not betting advice.
