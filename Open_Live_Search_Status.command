#!/bin/zsh
cd "$(dirname "$0")"
export M23_STATUS_REFRESH_SECONDS="${M23_STATUS_REFRESH_SECONDS:-3}"
exec python3 live_search_status.py --watch --interval "$M23_STATUS_REFRESH_SECONDS"
