#!/usr/bin/env python3
"""
live_search_status.py - Terminal status view for the descent search lane.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import time
from pathlib import Path

JSON_DIR = Path("testjson")
RUNTIME_DIR = JSON_DIR / "runtime"
STATE_PATH = JSON_DIR / "live_search_state.json"


def parse_args():
    parser = argparse.ArgumentParser(description="Watch live descent-search status.")
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=float, default=3.0)
    return parser.parse_args()


def read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return None


def process_running(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def read_pid(worker_index: int) -> int | None:
    path = RUNTIME_DIR / f"m23_descent_{worker_index}.pid"
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except Exception:
        return None


def tail_log(worker_index: int, lines: int = 3) -> str:
    path = RUNTIME_DIR / f"m23_descent_{worker_index}.log"
    try:
        data = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return "log unavailable"
    if not data:
        return "log empty"
    return " | ".join(data[-lines:])


def render() -> str:
    state = read_json(STATE_PATH) or {}
    lines = []
    lines.append("M23 Live Search Status")
    lines.append("=" * 24)
    lines.append(f"time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"status: {state.get('status', 'no-state')}")
    lines.append(f"runs_completed: {state.get('runs_completed', 0)}")
    lines.append(f"workers: {state.get('workers', '?')}")
    lines.append(f"partition_mode: {state.get('partition_mode', '?')}")
    lines.append(f"last_summary_file: {state.get('last_summary_file', 'none')}")

    current_stage = state.get("current_stage") or {}
    if current_stage:
        lines.append(
            "current_stage: "
            f"{current_stage.get('stage_index', '?')}/"
            f"{current_stage.get('stage_name', 'unknown')} "
            f"grid={current_stage.get('scale_count', '?')}x"
            f"{current_stage.get('shift_count', '?')} "
            f"candidates={current_stage.get('candidate_count', '?')}"
        )
    next_stage = state.get("next_stage") or {}
    if next_stage:
        lines.append(
            "next_stage: "
            f"{next_stage.get('stage_index', '?')}/"
            f"{next_stage.get('stage_name', 'unknown')} "
            f"grid={next_stage.get('scale_count', '?')}x"
            f"{next_stage.get('shift_count', '?')}"
        )

    best = state.get("best_overall") or {}
    if best:
        lines.append(
            "best_overall: "
            f"score={best.get('descent_score')} "
            f"leak={best.get('leakage_total')} "
            f"den_max={best.get('denominator_max')} "
            f"a={best.get('a')} b={best.get('b')}"
        )
    else:
        lines.append("best_overall: none yet")

    recent_runs = state.get("recent_runs") or []
    if recent_runs:
        last = recent_runs[-1]
        lines.append(
            "last_batch: "
            f"returncode={last.get('returncode')} "
            f"elapsed={last.get('elapsed_seconds')} "
            f"summary={last.get('summary_file')}"
        )

    followup = state.get("last_followup_summary") or {}
    if followup:
        lines.append(
            "last_followup: "
            f"rationalized={followup.get('rationalized_count', 0)}/"
            f"{followup.get('result_count', 0)} "
            f"cycle_rate={followup.get('cycle_rate', 0.0)} "
            f"tested_primes={followup.get('tested_prime_count', 0)} "
            f"matched_m23={followup.get('matched_m23_prime_count', 0)}"
        )
        lines.append(
            "followup_top: "
            f"signature_hits={followup.get('top_signature_hits', 0)} "
            f"irreducible_primes={followup.get('top_irreducible_prime_count', 0)} "
            f"rationalized={followup.get('top_rationalized', False)} "
            f"a={followup.get('top_a')} b={followup.get('top_b')}"
        )
    lines.append(f"last_followup_file: {state.get('last_followup_file', 'none')}")

    lines.append("")
    lines.append("Worker View")
    lines.append("-" * 11)
    workers = int(state.get("workers", 8) or 8)
    for worker_index in range(1, workers + 1):
        pid = read_pid(worker_index)
        running = process_running(pid)
        lines.append(
            f"worker {worker_index}: "
            f"{'RUNNING' if running else 'idle'} "
            f"pid={pid or '-'} "
            f"log={tail_log(worker_index)}"
        )

    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    if not args.watch:
        print(render())
        return 0

    try:
        while True:
            print("\033[2J\033[H", end="")
            print(render())
            time.sleep(max(0.5, args.interval))
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
