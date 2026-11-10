#!/bin/bash
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

cd "$(dirname "$0")"

# Legacy exact-lane Qt-only monitor surface.

export M23_WORKER_COUNT="${M23_WORKER_COUNT:-32}"
export M23_PARTITION_MODE="${M23_PARTITION_MODE:-chunk}"

python3 m23_monitor_qt.py
