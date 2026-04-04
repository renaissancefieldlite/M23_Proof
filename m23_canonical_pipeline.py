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
import subprocess
import sys
from pathlib import Path

JSON_DIR = Path("testjson")


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
    args = parser.parse_args()

    JSON_DIR.mkdir(exist_ok=True)
    candidate_path = JSON_DIR / "tschirnhaus_candidates_current.json"
    screen_path = JSON_DIR / "mod23_screen_current.json"

    generate_cmd = [
        sys.executable,
        "tschirnhaus_generate.py",
        "--family",
        args.family,
        f"--basis-values={args.basis_values}",
        "--limit",
        str(args.candidate_limit),
        "--output",
        str(candidate_path),
    ]
    screen_cmd = [
        sys.executable,
        "mod23_signature_screen.py",
        str(candidate_path),
        "--screen-limit",
        str(args.screen_limit),
        "--primes",
        args.primes,
        "--output",
        str(screen_path),
    ]
    lift_cmd = [
        sys.executable,
        "lift_survivors.py",
        str(screen_path),
    ]

    print("Running canonical M23 generator...")
    subprocess.run(generate_cmd, check=True)
    print("Running canonical M23 modular screen...")
    subprocess.run(screen_cmd, check=True)
    print("Promoting screened survivors...")
    subprocess.run(lift_cmd, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
