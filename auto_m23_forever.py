#!/usr/bin/env python3
"""
auto_m23_forever.py - Fully autonomous M23 search with cross-feeding
Runs until target consistency reached.
Supports multiple instances sharing best candidates via shared_best.json
"""

import glob
import json
import os
import subprocess
import sys
import time
from datetime import datetime

JSON_DIR = "testjson"
SHARED_BEST_FILE = os.path.join(JSON_DIR, "shared_best.json")
PHASE2_SCRIPT = "phase2_exact.py"
PHASE3_SCRIPT = "phase3_exact.py"
PHASE4_SCRIPT = "phase4_exact.py"

os.makedirs(JSON_DIR, exist_ok=True)

PRIMES_TESTED = 9
TARGET_IRRED_COUNT = 6
TARGET = TARGET_IRRED_COUNT / PRIMES_TESTED


def log_message(msg: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {msg}")
    sys.stdout.flush()


def list_result_files():
    pattern = os.path.join(JSON_DIR, "exact_test_results_*.json")
    return sorted(glob.glob(pattern), key=os.path.getmtime)


def get_latest_results():
    files = list_result_files()
    return files[-1] if files else None


def read_shared_best():
    try:
        if os.path.exists(SHARED_BEST_FILE):
            with open(SHARED_BEST_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return {"best_score": 0.0, "candidate": None, "source_file": None}


def write_shared_best(score, candidate_file, candidate_data):
    current = read_shared_best()
    if score <= current.get("best_score", 0.0):
        return

    data = {
        "best_score": score,
        "timestamp": time.time(),
        "source_file": candidate_file,
        "candidate": candidate_data,
    }
    tmp = SHARED_BEST_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, SHARED_BEST_FILE)
    log_message(f"NEW GLOBAL BEST: {score * 100:.1f}% from {os.path.basename(candidate_file)}")


def get_best_consistency():
    shared = read_shared_best()
    best_score = shared.get("best_score", 0.0)

    for result_file in list_result_files():
        try:
            with open(result_file, "r", encoding="utf-8") as fp:
                data = json.load(fp)
        except Exception:
            continue

        for cand in data:
            result = cand.get("result", {})
            if result.get("success", False):
                score = float(result.get("consistency_score", 0.0))
                if score > best_score:
                    best_score = score
                    write_shared_best(score, result_file, cand.get("candidate"))
    return best_score


def check_for_success():
    best = get_best_consistency()
    count = int(round(best * PRIMES_TESTED))
    log_message(f"Current best consistency: {best * 100:.1f}% ({count}/{PRIMES_TESTED} primes)")
    return best >= TARGET


def run_phase(phase_num, script_name):
    log_message(f"Running Phase {phase_num}: {script_name}")
    result = subprocess.run(
        [sys.executable, script_name],
        capture_output=True,
        text=True,
        cwd=os.getcwd(),
        stdin=subprocess.DEVNULL,
    )

    if result.stdout:
        tail = result.stdout[-1200:]
        log_message(f"stdout tail:\n{tail}")

    if result.returncode != 0:
        log_message(f"Phase {phase_num} returned code {result.returncode}")
        if result.stderr:
            log_message(f"stderr tail:\n{result.stderr[-1200:]}")
        return False

    log_message(f"Phase {phase_num} complete")
    return True


def main():
    iteration = 0
    max_iterations = 1000
    instance_id = os.environ.get("INSTANCE_ID", "1")
    worker_count = os.environ.get("WORKER_COUNT", "unknown")

    log_message("=" * 60)
    log_message(f"M23 AUTO-PILOT - INSTANCE {instance_id}")
    log_message(f"Configured worker count: {worker_count}")
    log_message(f"JSON directory: {JSON_DIR}")
    log_message(f"Shared best: {SHARED_BEST_FILE}")
    log_message(
        f"Target threshold: {TARGET_IRRED_COUNT}/{PRIMES_TESTED} irreducible prime checks"
    )
    log_message("=" * 60)

    while iteration < max_iterations:
        iteration += 1
        log_message(f"\n{'=' * 60}")
        log_message(f"INSTANCE {instance_id} ITERATION {iteration}")
        log_message(f"{'=' * 60}")

        if not run_phase(2, PHASE2_SCRIPT):
            log_message("Phase 2 failed. Retrying in 10 seconds...")
            time.sleep(10)
            continue

        if not run_phase(3, PHASE3_SCRIPT):
            log_message("Phase 3 failed. Retrying in 5 seconds...")
            time.sleep(5)
            continue

        if check_for_success():
            log_message("\n" + "=" * 60)
            log_message("TARGET REACHED")
            log_message(f"Best consistency: {get_best_consistency() * 100:.1f}%")
            log_message(f"Check {SHARED_BEST_FILE} for winning candidate")
            log_message("=" * 60)
            break

        log_message("Running Phase 4 (refinement)...")
        run_phase(4, PHASE4_SCRIPT)
        time.sleep(2)

    if iteration >= max_iterations:
        log_message(f"\nStopped after {max_iterations} iterations (safety cap)")
        log_message(f"Final best: {get_best_consistency() * 100:.1f}%")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log_message("\nStopped by user")
        sys.exit(0)
