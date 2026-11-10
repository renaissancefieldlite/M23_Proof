#!/bin/zsh
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

cd "$(dirname "$0")"
export M23_DESCENT_PARTITION_MODE="${M23_DESCENT_PARTITION_MODE:-scale_band}"
export M23_DESCENT_WORKERS="${M23_DESCENT_WORKERS:-8}"
exec python3 run_parallel_descent_channels.py --workers "$M23_DESCENT_WORKERS" --partition-mode "$M23_DESCENT_PARTITION_MODE"
