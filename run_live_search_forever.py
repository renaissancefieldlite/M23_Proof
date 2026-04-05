#!/usr/bin/env python3
"""
run_live_search_forever.py - Repeatedly execute descent search batches.

This keeps the live search lane moving in explicit batches:
run a batch, read the merged summary, update state, sleep, repeat.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

JSON_DIR = Path("testjson")
STATE_PATH = JSON_DIR / "live_search_state.json"
SUMMARY_PREFIX = "descent_search_summary_"


def parse_args():
    parser = argparse.ArgumentParser(description="Run the descent lane continuously.")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--partition-mode",
        default=os.environ.get("M23_DESCENT_PARTITION_MODE", "scale_band"),
        choices=["chunk", "stride", "scale_band", "shift_band", "ring"],
    )
    parser.add_argument("--sleep-seconds", type=float, default=5.0)
    parser.add_argument("--max-runs", type=int, default=0, help="0 means run until interrupted")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--script", default="run_parallel_descent_channels.py")
    return parser.parse_args()


def summary_files() -> list[Path]:
    return sorted(JSON_DIR.glob(f"{SUMMARY_PREFIX}*.json"), key=os.path.getmtime)


def atomic_json_dump(path: Path, payload: dict) -> None:
    path.parent.mkdir(exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    temp_path.replace(path)


def load_summary(path: Path | None) -> dict | None:
    if path is None or not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return None


def batch_best(summary: dict | None) -> dict | None:
    if not summary:
        return None
    merged = summary.get("merged_best") or []
    if not merged:
        return None
    return merged[0]


def summarize_best(best_row: dict | None) -> dict | None:
    if not best_row:
        return None
    best = best_row.get("best_transform") or {}
    transform = best.get("transform") or {}
    return {
        "source_file": best_row.get("source_file"),
        "worker": best_row.get("worker", {}).get("instance_id"),
        "descent_score": best.get("descent_score"),
        "leakage_total": best.get("leakage_total"),
        "denominator_max": best.get("denominator_max"),
        "a": transform.get("a"),
        "b": transform.get("b"),
    }


def is_better(candidate: dict | None, incumbent: dict | None) -> bool:
    if candidate is None:
        return False
    if incumbent is None:
        return True
    try:
        return float(candidate["descent_score"]) < float(incumbent["descent_score"])
    except Exception:
        return False


def write_state(
    *,
    status: str,
    runs_completed: int,
    args,
    last_summary_file: str | None,
    best_overall: dict | None,
    recent_runs: list[dict],
) -> None:
    payload = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "status": status,
        "workers": args.workers,
        "partition_mode": args.partition_mode,
        "sleep_seconds": args.sleep_seconds,
        "runs_completed": runs_completed,
        "last_summary_file": last_summary_file,
        "best_overall": best_overall,
        "recent_runs": recent_runs[-12:],
    }
    atomic_json_dump(STATE_PATH, payload)


def main() -> int:
    args = parse_args()
    JSON_DIR.mkdir(exist_ok=True)

    runs_completed = 0
    best_overall = None
    recent_runs: list[dict] = []

    try:
        while True:
            before = {path.name for path in summary_files()}
            write_state(
                status="running",
                runs_completed=runs_completed,
                args=args,
                last_summary_file=recent_runs[-1]["summary_file"] if recent_runs else None,
                best_overall=best_overall,
                recent_runs=recent_runs,
            )

            started = time.time()
            result = subprocess.run(
                [
                    args.python,
                    args.script,
                    "--workers",
                    str(args.workers),
                    "--partition-mode",
                    args.partition_mode,
                ],
                cwd=os.getcwd(),
                text=True,
            )
            elapsed = time.time() - started

            after_files = summary_files()
            new_files = [path for path in after_files if path.name not in before]
            summary_path = new_files[-1] if new_files else (after_files[-1] if after_files else None)
            summary = load_summary(summary_path)
            best_row = batch_best(summary)
            best_summary = summarize_best(best_row)
            if is_better(best_summary, best_overall):
                best_overall = best_summary

            runs_completed += 1
            recent_runs.append(
                {
                    "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "elapsed_seconds": elapsed,
                    "returncode": result.returncode,
                    "summary_file": str(summary_path) if summary_path else None,
                    "batch_best": best_summary,
                }
            )

            print(
                f"[live-search] batch={runs_completed} returncode={result.returncode} "
                f"elapsed={elapsed:.1f}s summary={summary_path}",
                flush=True,
            )
            if best_summary is not None:
                print(
                    "[live-search] batch-best "
                    f"score={best_summary['descent_score']:.6g} "
                    f"leak={best_summary['leakage_total']:.6g} "
                    f"den_max={best_summary['denominator_max']} "
                    f"a={best_summary['a']} b={best_summary['b']}",
                    flush=True,
                )
            if best_overall is not None:
                print(
                    "[live-search] overall-best "
                    f"score={best_overall['descent_score']:.6g} "
                    f"a={best_overall['a']} b={best_overall['b']}",
                    flush=True,
                )

            if result.returncode != 0:
                write_state(
                    status="error",
                    runs_completed=runs_completed,
                    args=args,
                    last_summary_file=str(summary_path) if summary_path else None,
                    best_overall=best_overall,
                    recent_runs=recent_runs,
                )
                return result.returncode

            if args.max_runs > 0 and runs_completed >= args.max_runs:
                write_state(
                    status="stopped",
                    runs_completed=runs_completed,
                    args=args,
                    last_summary_file=str(summary_path) if summary_path else None,
                    best_overall=best_overall,
                    recent_runs=recent_runs,
                )
                return 0

            write_state(
                status="sleeping",
                runs_completed=runs_completed,
                args=args,
                last_summary_file=str(summary_path) if summary_path else None,
                best_overall=best_overall,
                recent_runs=recent_runs,
            )
            time.sleep(max(0.0, args.sleep_seconds))
    except KeyboardInterrupt:
        write_state(
            status="stopped",
            runs_completed=runs_completed,
            args=args,
            last_summary_file=recent_runs[-1]["summary_file"] if recent_runs else None,
            best_overall=best_overall,
            recent_runs=recent_runs,
        )
        print("\n[live-search] stopped by user", flush=True)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
