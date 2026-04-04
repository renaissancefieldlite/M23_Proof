#!/usr/bin/env python3
"""
tschirnhaus_generate.py - First-pass canonical generator for the M23 descent lane.

This replaces the fake lambda/mu family with a bounded Tschirnhaus search over
the quartic field basis (1, g, g^2, g^3) anchored to the Elkies construction.

The default family is intentionally practical:
    h(y) = y^2 + a*y + b
where a, b live in the Elkies quartic field.

That gives us:
  1. a finite candidate set,
  2. clean worker partitioning,
  3. a mathematically meaningful generator we can screen downstream.

Affine remains available as a narrower fallback family.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import time
from pathlib import Path

from sympy import Rational

JSON_DIR = Path("testjson")


def parse_rational_values(raw: str) -> list[Rational]:
    values = []
    for chunk in raw.split(","):
        piece = chunk.strip()
        if not piece:
            continue
        values.append(Rational(piece))
    if not values:
        raise ValueError("no basis values parsed")
    return values


def parse_worker_config() -> tuple[str, int, int]:
    instance_id = os.environ.get("INSTANCE_ID", "1")
    try:
        worker_count = max(1, int(os.environ.get("WORKER_COUNT", "1")))
    except ValueError:
        worker_count = 1

    try:
        worker_index = max(1, int(instance_id))
    except ValueError:
        worker_index = 1

    return instance_id, worker_index, worker_count


def tuple_weight(values: tuple[Rational, ...]) -> tuple[float, int, tuple[str, ...]]:
    l1 = sum(abs(float(v)) for v in values)
    nonzero = sum(1 for v in values if v != 0)
    return l1, nonzero, tuple(str(v) for v in values)


def candidate_weight(a_basis: tuple[Rational, ...], b_basis: tuple[Rational, ...]) -> tuple[float, int, tuple[str, ...]]:
    return (
        sum(abs(float(v)) for v in a_basis + b_basis),
        sum(1 for v in a_basis + b_basis if v != 0),
        tuple(str(v) for v in a_basis + b_basis),
    )


def build_candidates(basis_values: list[Rational], family: str) -> list[dict]:
    a_tuples = list(itertools.product(basis_values, repeat=4))
    b_tuples = list(itertools.product(basis_values, repeat=4))

    a_tuples.sort(key=tuple_weight)
    b_tuples.sort(key=tuple_weight)

    candidates = []
    for a_basis in a_tuples:
        if all(value == 0 for value in a_basis):
            continue
        for b_basis in b_tuples:
            candidates.append(
                {
                    "family": family,
                    "a_basis": [str(value) for value in a_basis],
                    "b_basis": [str(value) for value in b_basis],
                    "weight": candidate_weight(a_basis, b_basis)[0],
                    "nonzero_terms": candidate_weight(a_basis, b_basis)[1],
                }
            )

    candidates.sort(
        key=lambda item: (
            item["weight"],
            item["nonzero_terms"],
            tuple(item["a_basis"] + item["b_basis"]),
        )
    )

    for index, candidate in enumerate(candidates):
        candidate["generator_index"] = index

    return candidates


def partition_candidates(candidates: list[dict]) -> tuple[str, list[dict]]:
    instance_id, worker_index, worker_count = parse_worker_config()
    if not candidates:
        return f"worker {instance_id} received no candidates", []

    start = math.floor((worker_index - 1) * len(candidates) / worker_count)
    end = math.floor(worker_index * len(candidates) / worker_count)
    selected = candidates[start:end]
    summary = (
        f"worker {instance_id}/{worker_count} scanning candidate chunk "
        f"[{start}:{end}] out of {len(candidates)} generated transforms"
    )
    return summary, selected


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--family",
        choices=["quadratic_tschirnhaus_f_basis", "affine_f_basis"],
        default=os.environ.get("M23_TSCHIRNHAUS_FAMILY", "quadratic_tschirnhaus_f_basis"),
    )
    parser.add_argument(
        "--basis-values",
        default=os.environ.get("M23_TSCHIRNHAUS_BASIS_VALUES", "-1,0,1"),
        help="comma-separated Rational values for each quartic-field basis coordinate",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=int(os.environ.get("M23_TSCHIRNHAUS_LIMIT", "0") or "0"),
        help="optional hard cap after worker partitioning",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="optional output path; defaults to testjson/tschirnhaus_candidates_workerX_TIMESTAMP.json",
    )
    args = parser.parse_args()

    JSON_DIR.mkdir(exist_ok=True)
    basis_values = parse_rational_values(args.basis_values)
    candidates = build_candidates(basis_values, args.family)
    section_summary, selected = partition_candidates(candidates)
    if args.limit > 0:
        selected = selected[: args.limit]

    instance_id, worker_index, worker_count = parse_worker_config()
    output_path = args.output or JSON_DIR / (
        f"tschirnhaus_candidates_worker{instance_id}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    )

    payload = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "family": args.family,
        "basis_values": [str(value) for value in basis_values],
        "worker": {
            "instance_id": instance_id,
            "worker_index": worker_index,
            "worker_count": worker_count,
            "section_summary": section_summary,
        },
        "candidate_count_total": len(candidates),
        "candidate_count_selected": len(selected),
        "candidates": selected,
    }

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    print(section_summary)
    print(f"Saved {len(selected)} candidate(s) to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
