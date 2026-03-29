#!/bin/bash
cd "$(dirname "$0")"

export M23_WORKER_COUNT="${M23_WORKER_COUNT:-2}"

python3 m23_monitor_qt.py
