#!/usr/bin/env bash
# NFL: download data, build features, train, fetch forecasts, predict.  Output: nfl/out/bundle.json
set -euo pipefail
mkdir -p /tmp/nfl/data && cd /tmp/nfl/data
B=https://github.com/nflverse/nflverse-data/releases/download
for y in 2020 2021 2022 2023 2024 2025 2026; do
  curl -fsSL -o play_by_play_$y.csv.gz $B/pbp/play_by_play_$y.csv.gz
  curl -fsSL -o injuries_$y.csv        $B/injuries/injuries_$y.csv
  curl -fsSL -o snaps_$y.csv           $B/snap_counts/snap_counts_$y.csv
  curl -fsSL -o depth_$y.csv           $B/depth_charts/depth_charts_$y.csv
done
for y in 2022 2023 2024 2025 2026; do curl -fsSL -o ftn_$y.csv $B/ftn_charting/ftn_charting_$y.csv; done
rm -rf /home/claude/nflverse/nfldata && git clone -q --depth 1 https://github.com/nflverse/nfldata /home/claude/nflverse/nfldata
cd /home/claude/nflpredict
python3 build_features.py
python3 train.py
python3 "$GITHUB_WORKSPACE/scripts/weather_fetch.py" nfl || echo "weather step failed; continuing without forecasts"
python3 predict.py
