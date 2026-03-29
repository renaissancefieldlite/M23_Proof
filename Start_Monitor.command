#!/bin/bash
cd "$(dirname "$0")"

if python3 -c "import PyQt6" >/dev/null 2>&1; then
  python3 m23_monitor_qt.py
else
  echo "PyQt6 not available; falling back to legacy tkinter monitor."
  python3 m23_monitor.py
fi
