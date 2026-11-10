#!/bin/zsh
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

cd "$(dirname "$0")"
export M23_STATUS_REFRESH_SECONDS="${M23_STATUS_REFRESH_SECONDS:-3}"
exec python3 live_search_status.py --watch --interval "$M23_STATUS_REFRESH_SECONDS"
