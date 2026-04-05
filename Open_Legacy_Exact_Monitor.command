#!/bin/bash
cd "$(dirname "$0")"

# Legacy exact-lane monitor surface.

export M23_WORKER_COUNT="${M23_WORKER_COUNT:-32}"
export M23_PARTITION_MODE="${M23_PARTITION_MODE:-chunk}"

if python3 -c "import PyQt6" >/dev/null 2>&1; then
  python3 m23_monitor_qt.py
else
  echo "PyQt6 not available; falling back to legacy tkinter monitor."
  python3 m23_monitor.py
fi
