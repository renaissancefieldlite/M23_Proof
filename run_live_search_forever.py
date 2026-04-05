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

from sympy import Rational

from descent_search import DEFAULT_SCALINGS, DEFAULT_SHIFTS, ordered_values
from mod23_signature_screen import DEFAULT_PRIMES

JSON_DIR = Path("testjson")
STATE_PATH = JSON_DIR / "live_search_state.json"
SUMMARY_PREFIX = "descent_search_summary_"
FOLLOWUP_PREFIX = "descent_followup_"
FOLLOWUP_SCRIPT = "descent_followup_screen.py"

EXTRA_SCALINGS = [
    Rational(-4, 1),
    Rational(-5, 2),
    Rational(-5, 3),
    Rational(-5, 4),
    Rational(5, 4),
    Rational(5, 3),
    Rational(5, 2),
    Rational(4, 1),
]

EXTRA_SHIFTS = [
    Rational(-10, 1),
    Rational(-9, 1),
    Rational(-7, 1),
    Rational(-9, 2),
    Rational(-7, 2),
    Rational(7, 2),
    Rational(9, 2),
    Rational(7, 1),
    Rational(9, 1),
    Rational(10, 1),
]


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
    parser.add_argument(
        "--followup-limit",
        type=int,
        default=int(os.environ.get("M23_LIVE_FOLLOWUP_LIMIT", "12") or "12"),
        help="number of top transforms to promote into modular follow-up each batch",
    )
    parser.add_argument(
        "--followup-primes",
        default=os.environ.get(
            "M23_LIVE_FOLLOWUP_PRIMES",
            ",".join(str(prime) for prime in DEFAULT_PRIMES),
        ),
        help="comma-separated primes used by the modular follow-up screen",
    )
    parser.add_argument(
        "--start-stage",
        type=int,
        default=int(os.environ.get("M23_LIVE_START_STAGE", "1") or "1"),
        help="1-based schedule stage to begin from",
    )
    parser.add_argument(
        "--pressure-cap",
        type=float,
        default=(
            float(os.environ["M23_DESCENT_PRESSURE_CAP"])
            if os.environ.get("M23_DESCENT_PRESSURE_CAP")
            else None
        ),
        help="optional cap on denominator pressure (denominator_max) during descent",
    )
    parser.add_argument(
        "--height-abs-cap",
        type=float,
        default=(
            float(os.environ["M23_DESCENT_HEIGHT_ABS_CAP"])
            if os.environ.get("M23_DESCENT_HEIGHT_ABS_CAP")
            else None
        ),
        help="optional cap on the maximum absolute rational coefficient component",
    )
    parser.add_argument(
        "--leakage-cap",
        type=float,
        default=(
            float(os.environ["M23_DESCENT_LEAKAGE_CAP"])
            if os.environ.get("M23_DESCENT_LEAKAGE_CAP")
            else None
        ),
        help="optional cap on total leakage during descent",
    )
    parser.add_argument(
        "--dead-lane-limit",
        type=int,
        default=int(os.environ.get("M23_DESCENT_DEAD_LANE_LIMIT", "0") or "0"),
        help="optional consecutive rejection limit before a worker stops early",
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--script", default="run_parallel_descent_channels.py")
    return parser.parse_args()


def summary_files() -> list[Path]:
    return sorted(JSON_DIR.glob(f"{SUMMARY_PREFIX}*.json"), key=os.path.getmtime)


def followup_files() -> list[Path]:
    return sorted(JSON_DIR.glob(f"{FOLLOWUP_PREFIX}*.json"), key=os.path.getmtime)


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


def progressive_schedule() -> list[dict]:
    scale_values = ordered_values(
        list(DEFAULT_SCALINGS) + EXTRA_SCALINGS,
        center=Rational(1, 1),
        order="center_out",
    )
    shift_values = ordered_values(
        list(DEFAULT_SHIFTS) + EXTRA_SHIFTS,
        center=Rational(0, 1),
        order="center_out",
    )

    stage_defs = [
        ("affine_seed", 4, 5),
        ("affine_core", 6, 7),
        ("affine_inner", 8, 9),
        ("affine_mid", 10, 11),
        ("affine_mid_wide", 12, 13),
        ("affine_outer_base", 14, 17),
        ("affine_full_base", 16, 21),
        ("affine_expanded_1", 20, 25),
        ("affine_expanded_2", len(scale_values), len(shift_values)),
    ]

    schedule = []
    for stage_index, (stage_name, scale_count, shift_count) in enumerate(stage_defs, start=1):
        scalings = scale_values[:scale_count]
        shifts = shift_values[:shift_count]
        schedule.append(
            {
                "stage_index": stage_index,
                "stage_name": stage_name,
                "scale_count": len(scalings),
                "shift_count": len(shifts),
                "candidate_count": len(scalings) * len(shifts),
                "scalings": [str(value) for value in scalings],
                "shifts": [str(value) for value in shifts],
            }
        )
    return schedule


def stage_summary(stage: dict | None) -> dict | None:
    if not stage:
        return None
    return {
        "stage_index": stage.get("stage_index"),
        "stage_name": stage.get("stage_name"),
        "scale_count": stage.get("scale_count"),
        "shift_count": stage.get("shift_count"),
        "candidate_count": stage.get("candidate_count"),
    }


def load_followup(path: Path | None) -> dict | None:
    return load_summary(path)


def summarize_followup(payload: dict | None) -> dict | None:
    if not payload:
        return None
    results = payload.get("results") or []
    top = results[0] if results else {}
    transform = top.get("transform") or {}
    overall = payload.get("overall_cycle_summary") or {}
    return {
        "result_count": int(payload.get("result_count", len(results))),
        "rationalized_count": int(payload.get("rationalized_count", 0)),
        "tested_prime_count": int(overall.get("tested_prime_count", 0)),
        "matched_m23_prime_count": int(overall.get("matched_m23_prime_count", 0)),
        "cycle_rate": float(overall.get("exact_m23_cycle_rate", 0.0)),
        "a23_exclusion_status": overall.get("a23_exclusion_status"),
        "top_signature_hits": int(top.get("signature_hit_count", 0) or 0),
        "top_irreducible_prime_count": int(top.get("irreducible_prime_count", 0) or 0),
        "top_rationalized": bool(top.get("rationalized", False)),
        "top_a": transform.get("a"),
        "top_b": transform.get("b"),
    }


def run_followup(summary_path: Path | None, args, stage: dict) -> tuple[Path | None, dict | None]:
    if summary_path is None or not summary_path.exists():
        return None, None

    output_path = JSON_DIR / f"{FOLLOWUP_PREFIX}{summary_path.stem}.json"
    command = [
        args.python,
        FOLLOWUP_SCRIPT,
        str(summary_path),
        "--followup-limit",
        str(args.followup_limit),
        "--primes",
        args.followup_primes,
        "--output",
        str(output_path),
    ]
    before = {path.name for path in followup_files()}
    result = subprocess.run(command, cwd=os.getcwd(), text=True, capture_output=True)
    after = followup_files()
    new_files = [path for path in after if path.name not in before]
    followup_path = output_path if output_path.exists() else (new_files[-1] if new_files else None)
    payload = load_followup(followup_path)
    wrapped = {
        "returncode": result.returncode,
        "stdout_tail": result.stdout[-2000:],
        "stderr_tail": result.stderr[-1000:],
        "file": str(followup_path) if followup_path else None,
        "summary": summarize_followup(payload),
        "stage": stage_summary(stage),
    }
    return followup_path, wrapped


def write_state(
    *,
    status: str,
    runs_completed: int,
    args,
    last_summary_file: str | None,
    best_overall: dict | None,
    recent_runs: list[dict],
    current_stage: dict | None,
    next_stage: dict | None,
    last_followup_file: str | None,
    last_followup_summary: dict | None,
) -> None:
    payload = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "status": status,
        "workers": args.workers,
        "partition_mode": args.partition_mode,
        "active_caps": {
            "pressure_cap": args.pressure_cap,
            "height_abs_cap": args.height_abs_cap,
            "leakage_cap": args.leakage_cap,
            "dead_lane_limit": args.dead_lane_limit,
        },
        "sleep_seconds": args.sleep_seconds,
        "runs_completed": runs_completed,
        "last_summary_file": last_summary_file,
        "best_overall": best_overall,
        "current_stage": current_stage,
        "next_stage": next_stage,
        "last_followup_file": last_followup_file,
        "last_followup_summary": last_followup_summary,
        "recent_runs": recent_runs[-12:],
    }
    atomic_json_dump(STATE_PATH, payload)


def main() -> int:
    args = parse_args()
    JSON_DIR.mkdir(exist_ok=True)
    schedule = progressive_schedule()
    stage_cursor = max(0, int(args.start_stage) - 1)

    runs_completed = 0
    best_overall = None
    recent_runs: list[dict] = []
    last_followup_file = None
    last_followup_summary = None

    try:
        while True:
            stage = schedule[stage_cursor % len(schedule)]
            next_stage = schedule[(stage_cursor + 1) % len(schedule)]
            before = {path.name for path in summary_files()}
            write_state(
                status="running",
                runs_completed=runs_completed,
                args=args,
                last_summary_file=recent_runs[-1]["summary_file"] if recent_runs else None,
                best_overall=best_overall,
                recent_runs=recent_runs,
                current_stage=stage_summary(stage),
                next_stage=stage_summary(next_stage),
                last_followup_file=last_followup_file,
                last_followup_summary=last_followup_summary,
            )

            started = time.time()
            run_env = {
                **os.environ,
                "M23_DESCENT_SCALINGS": ",".join(stage["scalings"]),
                "M23_DESCENT_SHIFTS": ",".join(stage["shifts"]),
                "M23_DESCENT_SCALE_ORDER": "center_out",
                "M23_DESCENT_SHIFT_ORDER": "center_out",
            }
            if args.pressure_cap is not None:
                run_env["M23_DESCENT_PRESSURE_CAP"] = str(args.pressure_cap)
            if args.height_abs_cap is not None:
                run_env["M23_DESCENT_HEIGHT_ABS_CAP"] = str(args.height_abs_cap)
            if args.leakage_cap is not None:
                run_env["M23_DESCENT_LEAKAGE_CAP"] = str(args.leakage_cap)
            if args.dead_lane_limit:
                run_env["M23_DESCENT_DEAD_LANE_LIMIT"] = str(args.dead_lane_limit)
            result = subprocess.run(
                [
                    args.python,
                    args.script,
                    "--workers",
                    str(args.workers),
                    "--partition-mode",
                    args.partition_mode,
                    *(
                        ["--pressure-cap", str(args.pressure_cap)]
                        if args.pressure_cap is not None
                        else []
                    ),
                    *(
                        ["--height-abs-cap", str(args.height_abs_cap)]
                        if args.height_abs_cap is not None
                        else []
                    ),
                    *(
                        ["--leakage-cap", str(args.leakage_cap)]
                        if args.leakage_cap is not None
                        else []
                    ),
                    *(["--dead-lane-limit", str(args.dead_lane_limit)] if args.dead_lane_limit else []),
                ],
                cwd=os.getcwd(),
                text=True,
                env=run_env,
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

            followup_path = None
            followup_result = None
            if summary_path is not None and result.returncode == 0:
                followup_path, followup_result = run_followup(summary_path, args, stage)
                if followup_result is not None:
                    last_followup_file = followup_result.get("file")
                    last_followup_summary = followup_result.get("summary")

            runs_completed += 1
            recent_runs.append(
                {
                    "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "elapsed_seconds": elapsed,
                    "returncode": result.returncode,
                    "summary_file": str(summary_path) if summary_path else None,
                    "batch_best": best_summary,
                    "stage": stage_summary(stage),
                    "followup_file": str(followup_path) if followup_path else None,
                    "followup": followup_result,
                }
            )

            print(
                f"[live-search] batch={runs_completed} returncode={result.returncode} "
                f"elapsed={elapsed:.1f}s stage={stage['stage_name']} "
                f"grid={stage['scale_count']}x{stage['shift_count']} summary={summary_path}",
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
            if followup_result is not None and followup_result.get("summary") is not None:
                summary_bits = followup_result["summary"]
                print(
                    "[live-search] followup "
                    f"rationalized={summary_bits['rationalized_count']}/"
                    f"{summary_bits['result_count']} "
                    f"cycle_rate={summary_bits['cycle_rate']:.3f} "
                    f"top_signature_hits={summary_bits['top_signature_hits']}",
                    flush=True,
                )
            elif followup_result is not None:
                print(
                    "[live-search] followup "
                    f"returncode={followup_result['returncode']} "
                    f"file={followup_result['file']}",
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
                    current_stage=stage_summary(stage),
                    next_stage=stage_summary(next_stage),
                    last_followup_file=last_followup_file,
                    last_followup_summary=last_followup_summary,
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
                    current_stage=stage_summary(stage),
                    next_stage=stage_summary(next_stage),
                    last_followup_file=last_followup_file,
                    last_followup_summary=last_followup_summary,
                )
                return 0

            stage_cursor += 1
            write_state(
                status="sleeping",
                runs_completed=runs_completed,
                args=args,
                last_summary_file=str(summary_path) if summary_path else None,
                best_overall=best_overall,
                recent_runs=recent_runs,
                current_stage=stage_summary(schedule[stage_cursor % len(schedule)]),
                next_stage=stage_summary(schedule[(stage_cursor + 1) % len(schedule)]),
                last_followup_file=last_followup_file,
                last_followup_summary=last_followup_summary,
            )
            time.sleep(max(0.0, args.sleep_seconds))
    except KeyboardInterrupt:
        current_stage = schedule[stage_cursor % len(schedule)]
        next_stage = schedule[(stage_cursor + 1) % len(schedule)]
        write_state(
            status="stopped",
            runs_completed=runs_completed,
            args=args,
            last_summary_file=recent_runs[-1]["summary_file"] if recent_runs else None,
            best_overall=best_overall,
            recent_runs=recent_runs,
            current_stage=stage_summary(current_stage),
            next_stage=stage_summary(next_stage),
            last_followup_file=last_followup_file,
            last_followup_summary=last_followup_summary,
        )
        print("\n[live-search] stopped by user", flush=True)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
