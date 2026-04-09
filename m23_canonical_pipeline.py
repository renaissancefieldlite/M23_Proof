#!/usr/bin/env python3
"""
m23_canonical_pipeline.py - Wire the canonical generator/filter split together.

This is the first-pass implementation of the new M23 architecture:
  1. generate bounded low-degree Tschirnhaus transforms over the Elkies quartic field anchor
  2. screen them with exact leakage metrics and modular signatures

If the quadratic family is too narrow, that becomes the next blocker and we
widen the generator family instead of guessing.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

JSON_DIR = Path("testjson")


def run_stage(
    *,
    family: str,
    basis_values: str,
    candidate_limit: int,
    screen_limit: int,
    primes: str,
    candidate_path: Path,
    screen_path: Path,
    survivor_path: Path,
) -> int:
    generate_cmd = [
        sys.executable,
        "tschirnhaus_generate.py",
        "--family",
        family,
        f"--basis-values={basis_values}",
        "--limit",
        str(candidate_limit),
        "--output",
        str(candidate_path),
    ]
    screen_cmd = [
        sys.executable,
        "mod23_signature_screen.py",
        str(candidate_path),
        "--screen-limit",
        str(screen_limit),
        "--primes",
        primes,
        "--output",
        str(screen_path),
    ]
    lift_cmd = [
        sys.executable,
        "lift_survivors.py",
        str(screen_path),
        "--output",
        str(survivor_path),
    ]

    print(
        "Running stage:",
        f"family={family}",
        f"basis_values={basis_values}",
        f"candidate_limit={candidate_limit}",
        f"screen_limit={screen_limit}",
    )
    subprocess.run(generate_cmd, check=True)
    subprocess.run(screen_cmd, check=True)
    subprocess.run(lift_cmd, check=True)

    with survivor_path.open("r", encoding="utf-8") as handle:
        survivor_payload = json.load(handle)
    survivor_count = int(survivor_payload.get("survivor_count", 0))
    print(f"Lifted survivor count: {survivor_count}")
    return survivor_count


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--family",
        default="quadratic_tschirnhaus_f_basis",
        choices=["quadratic_tschirnhaus_f_basis", "affine_f_basis"],
    )
    parser.add_argument("--basis-values", default="-1,0,1")
    parser.add_argument("--candidate-limit", type=int, default=64)
    parser.add_argument("--screen-limit", type=int, default=16)
    parser.add_argument("--primes", default="47,139,277,461,599")
    parser.add_argument("--survivor-index", type=int, default=0)
    parser.add_argument("--sampler-prime", type=int, default=100000007)
    parser.add_argument("--sampler-sample-count", type=int, default=64)
    parser.add_argument("--sampler-start-t0", type=int, default=0)
    parser.add_argument("--sampler-step", type=int, default=1)
    parser.add_argument(
        "--stage-mode",
        choices=["single", "auto_widen"],
        default="auto_widen",
        help="single runs one configured stage; auto_widen retries broader generator settings until survivors appear",
    )
    args = parser.parse_args()

    JSON_DIR.mkdir(exist_ok=True)
    candidate_path = JSON_DIR / "tschirnhaus_candidates_current.json"
    screen_path = JSON_DIR / "mod23_screen_current.json"
    survivor_path = JSON_DIR / "lift_survivors_current.json"
    sample_path = JSON_DIR / "m23_fixed_prime_sample_top_survivor_current.json"
    sampler_cmd = [
        sys.executable,
        "m23_fixed_prime_sampler.py",
        "--prime",
        str(args.sampler_prime),
        "--sample-count",
        str(args.sampler_sample_count),
        "--start-t0",
        str(args.sampler_start_t0),
        "--step",
        str(args.sampler_step),
        "--survivor-json",
        str(survivor_path),
        "--survivor-index",
        str(args.survivor_index),
        "--output",
        str(sample_path),
    ]

    if args.stage_mode == "single":
        stages = [
            {
                "family": args.family,
                "basis_values": args.basis_values,
                "candidate_limit": args.candidate_limit,
                "screen_limit": args.screen_limit,
            }
        ]
    else:
        base_screen = max(args.screen_limit, 16)
        base_candidates = max(args.candidate_limit, base_screen)
        stages = [
            {
                "family": args.family,
                "basis_values": args.basis_values,
                "candidate_limit": base_candidates,
                "screen_limit": base_screen,
            },
            {
                "family": "affine_f_basis",
                "basis_values": args.basis_values,
                "candidate_limit": max(base_candidates, 64),
                "screen_limit": max(base_screen, 32),
            },
            {
                "family": "quadratic_tschirnhaus_f_basis",
                "basis_values": "-2,-1,0,1,2",
                "candidate_limit": max(base_candidates, 96),
                "screen_limit": max(base_screen, 24),
            },
        ]

    survivor_count = 0
    for stage_index, stage in enumerate(stages, start=1):
        print(f"Canonical stage {stage_index}/{len(stages)}")
        survivor_count = run_stage(
            family=stage["family"],
            basis_values=stage["basis_values"],
            candidate_limit=stage["candidate_limit"],
            screen_limit=stage["screen_limit"],
            primes=args.primes,
            candidate_path=candidate_path,
            screen_path=screen_path,
            survivor_path=survivor_path,
        )
        if survivor_count > 0:
            break

    if survivor_count <= 0:
        print("No rationalized survivors available; skipping fixed-prime sampler handoff.")
        return 0
    if args.survivor_index < 0 or args.survivor_index >= survivor_count:
        raise IndexError(
            f"survivor index {args.survivor_index} out of range "
            f"(survivor_count={survivor_count})"
        )
    print("Sampling top lifted survivor at a fixed prime...")
    subprocess.run(sampler_cmd, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
