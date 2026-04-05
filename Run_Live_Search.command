#!/bin/zsh
cd "$(dirname "$0")"
export M23_DESCENT_PARTITION_MODE="${M23_DESCENT_PARTITION_MODE:-scale_band}"
export M23_DESCENT_WORKERS="${M23_DESCENT_WORKERS:-8}"
exec python3 run_parallel_descent_channels.py --workers "$M23_DESCENT_WORKERS" --partition-mode "$M23_DESCENT_PARTITION_MODE"
