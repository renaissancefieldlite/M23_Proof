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
import heapq
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


def tuple_priority(values: tuple[Rational, ...]) -> tuple[int, float, tuple[str, ...]]:
    l1, nonzero, lex = tuple_weight(values)
    return nonzero, l1, lex


def candidate_weight(a_basis: tuple[Rational, ...], b_basis: tuple[Rational, ...]) -> tuple[float, int, tuple[str, ...]]:
    return (
        sum(abs(float(v)) for v in a_basis + b_basis),
        sum(1 for v in a_basis + b_basis if v != 0),
        tuple(str(v) for v in a_basis + b_basis),
    )


def candidate_priority(a_basis: tuple[Rational, ...], b_basis: tuple[Rational, ...]) -> tuple[int, float, tuple[str, ...]]:
    weight, nonzero, lex = candidate_weight(a_basis, b_basis)
    return nonzero, weight, lex


def make_candidate(family: str, a_basis: tuple[Rational, ...], b_basis: tuple[Rational, ...]) -> dict:
    return {
        "family": family,
        "a_basis": [str(value) for value in a_basis],
        "b_basis": [str(value) for value in b_basis],
        "weight": candidate_weight(a_basis, b_basis)[0],
        "nonzero_terms": candidate_weight(a_basis, b_basis)[1],
    }


def bounded_candidates(
    a_tuples: list[tuple[Rational, ...]],
    b_tuples: list[tuple[Rational, ...]],
    family: str,
    target: int,
) -> list[dict]:
    if target <= 0 or not a_tuples or not b_tuples:
        return []

    visited = {(0, 0)}
    heap = [(candidate_priority(a_tuples[0], b_tuples[0]), 0, 0)]
    candidates = []

    while heap and len(candidates) < target:
        _, a_index, b_index = heapq.heappop(heap)
        candidates.append(make_candidate(family, a_tuples[a_index], b_tuples[b_index]))

        if a_index + 1 < len(a_tuples) and (a_index + 1, b_index) not in visited:
            visited.add((a_index + 1, b_index))
            heapq.heappush(
                heap,
                (candidate_priority(a_tuples[a_index + 1], b_tuples[b_index]), a_index + 1, b_index),
            )
        if b_index + 1 < len(b_tuples) and (a_index, b_index + 1) not in visited:
            visited.add((a_index, b_index + 1))
            heapq.heappush(
                heap,
                (candidate_priority(a_tuples[a_index], b_tuples[b_index + 1]), a_index, b_index + 1),
            )

    return candidates


def build_candidates(
    basis_values: list[Rational],
    family: str,
    *,
    generation_limit: int = 0,
    worker_count: int = 1,
) -> tuple[list[dict], int, str]:
    a_tuples = list(itertools.product(basis_values, repeat=4))
    b_tuples = list(itertools.product(basis_values, repeat=4))

    a_tuples.sort(key=tuple_priority)
    b_tuples.sort(key=tuple_priority)
    a_tuples = [values for values in a_tuples if any(value != 0 for value in values)]
    total_possible = len(a_tuples) * len(b_tuples)

    if generation_limit > 0:
        generation_mode = "bounded_priority"
        target = min(total_possible, generation_limit * max(1, worker_count))
        candidates = bounded_candidates(a_tuples, b_tuples, family, target)
    else:
        generation_mode = "full_cartesian"
        candidates = []
        for a_basis in a_tuples:
            for b_basis in b_tuples:
                candidates.append(make_candidate(family, a_basis, b_basis))

    for index, candidate in enumerate(candidates):
        candidate["generator_index"] = index

    return candidates, total_possible, generation_mode


def partition_candidates(candidates: list[dict], total_possible: int, generation_mode: str) -> tuple[str, list[dict]]:
    instance_id, worker_index, worker_count = parse_worker_config()
    if not candidates:
        return f"worker {instance_id} received no candidates", []

    start = math.floor((worker_index - 1) * len(candidates) / worker_count)
    end = math.floor(worker_index * len(candidates) / worker_count)
    selected = candidates[start:end]
    summary = (
        f"worker {instance_id}/{worker_count} scanning candidate chunk "
        f"[{start}:{end}] out of {len(candidates)} generated transforms "
        f"(mode={generation_mode}, order=sparsity_then_weight, estimated_total={total_possible})"
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
    _, _, worker_count = parse_worker_config()
    candidates, total_possible, generation_mode = build_candidates(
        basis_values,
        args.family,
        generation_limit=args.limit,
        worker_count=worker_count,
    )
    section_summary, selected = partition_candidates(candidates, total_possible, generation_mode)
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
        "candidate_generation_mode": generation_mode,
        "candidate_ordering_mode": "sparsity_then_weight",
        "candidate_count_total": len(candidates),
        "candidate_count_estimated_total": total_possible,
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
