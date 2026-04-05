#!/usr/bin/env python3
"""
run_parallel_descent_channels.py - Launch disjoint descent-search workers.

This is the reproducible runner for the Elkies descent lane. It starts multiple
descent_search.py workers, gives each worker its own section of the transform
space, waits for them to finish, and writes a merged summary file.
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
RUNTIME_DIR = JSON_DIR / "runtime"
SUMMARY_PREFIX = "descent_search_summary"
WORKER_LOG_PREFIX = "m23_descent"


def pid_path(worker_index: int) -> Path:
    return RUNTIME_DIR / f"{WORKER_LOG_PREFIX}_{worker_index}.pid"


def write_pid(worker_index: int, pid: int) -> None:
    pid_path(worker_index).write_text(str(pid), encoding="utf-8")


def remove_pid(worker_index: int) -> None:
    target = pid_path(worker_index)
    if target.exists():
        target.unlink()


def parse_args():
    parser = argparse.ArgumentParser(description="Run partitioned Elkies descent workers.")
    parser.add_argument("--workers", type=int, default=max(2, min(8, os.cpu_count() or 4)))
    parser.add_argument(
        "--partition-mode",
        default=os.environ.get("M23_DESCENT_PARTITION_MODE", "scale_band"),
        choices=["chunk", "stride", "scale_band", "shift_band", "ring"],
    )
    parser.add_argument(
        "--script",
        default="descent_search.py",
        help="Worker script to launch",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable to use",
    )
    return parser.parse_args()


def output_label():
    millis = int((time.time() % 1) * 1000)
    return f"{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}_{millis:03d}"


def launch_workers(args, label):
    processes = []
    JSON_DIR.mkdir(exist_ok=True)
    RUNTIME_DIR.mkdir(exist_ok=True)

    for worker_index in range(1, args.workers + 1):
        env = os.environ.copy()
        env["INSTANCE_ID"] = str(worker_index)
        env["WORKER_COUNT"] = str(args.workers)
        env["M23_DESCENT_PARTITION_MODE"] = args.partition_mode
        env["M23_RUN_LABEL"] = label

        log_path = RUNTIME_DIR / f"{WORKER_LOG_PREFIX}_{worker_index}.log"
        log_handle = log_path.open("wb")
        env["PYTHONUNBUFFERED"] = "1"

        process = subprocess.Popen(
            [args.python, "-u", args.script],
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            cwd=os.getcwd(),
            env=env,
        )
        log_handle.close()
        write_pid(worker_index, process.pid)
        processes.append((worker_index, process, log_path))
        print(
            f"Started descent worker {worker_index}/{args.workers} "
            f"(mode={args.partition_mode}, pid={process.pid}, log={log_path})",
            flush=True,
        )

    return processes


def wait_for_workers(processes):
    failures = []
    for worker_index, process, log_path in processes:
        return_code = process.wait()
        remove_pid(worker_index)
        if return_code == 0:
            print(f"Worker {worker_index} completed successfully", flush=True)
        else:
            failures.append((worker_index, return_code, log_path))
            print(
                f"Worker {worker_index} failed with code {return_code} (log: {log_path})",
                flush=True,
            )
    return failures


def collect_result_files(label):
    pattern = f"descent_search_worker*_{label}_*.json"
    return sorted(JSON_DIR.glob(pattern))


def build_summary(result_files, label, args):
    merged = []
    skipped_files = []
    for path in result_files:
        try:
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except Exception as exc:
            skipped_files.append({"path": str(path), "error": str(exc)})
            continue
        ranked = payload.get("ranked_transforms", [])
        if not ranked:
            continue
        best = ranked[0]
        merged.append(
            {
                "source_file": str(path),
                "worker": payload.get("worker", {}),
                "best_transform": best,
                "top_count": min(10, len(ranked)),
                "top_transforms": ranked[:10],
            }
        )

    merged.sort(key=lambda item: item["best_transform"]["descent_score"])

    summary = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "label": label,
        "workers": args.workers,
        "partition_mode": args.partition_mode,
        "result_files": [str(path) for path in result_files],
        "skipped_files": skipped_files,
        "merged_best": merged,
    }

    summary_path = JSON_DIR / f"{SUMMARY_PREFIX}_{label}.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary_path, merged


def main():
    args = parse_args()
    label = output_label()
    processes = launch_workers(args, label)
    failures = wait_for_workers(processes)

    result_files = collect_result_files(label)
    summary_path, merged = build_summary(result_files, label, args)

    print(f"\nWrote merged descent summary to {summary_path}", flush=True)
    for item in merged[:10]:
        best = item["best_transform"]
        worker = item["worker"].get("instance_id", "?")
        transform = best["transform"]
        print(
            f"worker={worker} score={best['descent_score']:.6g} "
            f"leak={best['leakage_total']:.6g} den_max={best['denominator_max']} "
            f"a={transform['a']} b={transform['b']}",
            flush=True,
        )

    if failures:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
