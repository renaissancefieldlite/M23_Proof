#!/bin/zsh
cd "$(dirname "$0")"
export M23_DESCENT_PARTITION_MODE="${M23_DESCENT_PARTITION_MODE:-scale_band}"
export M23_DESCENT_WORKERS="${M23_DESCENT_WORKERS:-8}"
export M23_LIVE_SEARCH_SLEEP_SECONDS="${M23_LIVE_SEARCH_SLEEP_SECONDS:-5}"
export M23_LIVE_SEARCH_MAX_RUNS="${M23_LIVE_SEARCH_MAX_RUNS:-0}"
exec python3 run_live_search_forever.py \
  --workers "$M23_DESCENT_WORKERS" \
  --partition-mode "$M23_DESCENT_PARTITION_MODE" \
  --sleep-seconds "$M23_LIVE_SEARCH_SLEEP_SECONDS" \
  --max-runs "$M23_LIVE_SEARCH_MAX_RUNS"
