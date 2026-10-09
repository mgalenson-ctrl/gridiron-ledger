#!/usr/bin/env bash
# College: fetch cfbfastR-data files, extract play-by-play, build, train, fetch forecasts, predict.  Output: college/out/bundle.json
set -euo pipefail
D=/home/claude/sportsdataverse/cfbfastr-data
rm -rf $D && git clone -q --depth 1 --filter=blob:none --no-checkout https://github.com/sportsdataverse/cfbfastr-data $D
( cd $D && git checkout -q HEAD -- \
  data/rds/pbp_players_pos_2023.rds data/rds/pbp_players_pos_2024.rds data/rds/pbp_players_pos_2025.rds data/rds/pbp_players_pos_2026.rds \
  schedules/csv/cfb_schedules_2023.csv schedules/csv/cfb_schedules_2024.csv schedules/csv/cfb_schedules_2025.csv schedules/csv/cfb_schedules_2026.csv \
  team_info/rds/cfb_team_info_2025.rds team_info/rds/cfb_team_info_2026.rds )
mkdir -p /tmp/cfb/data && cd /home/claude/cfbpredict
for y in 2023 2024 2025 2026; do python3 extract_pbp.py $y; done
python3 team_info.py
python3 cfb_build.py
python3 cfb_train.py
python3 "$GITHUB_WORKSPACE/scripts/weather_fetch.py" college || echo "weather step failed; continuing without forecasts"
python3 "$GITHUB_WORKSPACE/scripts/live_feeds.py" college || echo "live feeds failed; using CFBD lines only"
python3 cfb_predict.py
