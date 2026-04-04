#!/bin/bash
cd "$(dirname "$0")"

export M23_WORKER_COUNT="${M23_WORKER_COUNT:-32}"
export M23_PARTITION_MODE="${M23_PARTITION_MODE:-chunk}"

python3 m23_monitor_qt.py
