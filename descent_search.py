#!/usr/bin/env python3
"""
descent_search.py - Partitioned descent search over the explicit Elkies anchor.

This script treats the Elkies quartic-field construction as the anchor object
and scores affine transforms x -> a*x + b by:
  1. reducing every transformed coefficient into the basis (1, g, g^2, g^3),
  2. measuring non-rational leakage,
  3. measuring rational coefficient height,
  4. recording denominator pressure.

The search can be split across multiple workers so the descent lane covers
disjoint transform bands instead of duplicating the same pass.
"""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

from sympy import N, Poly, Rational, denom, expand, rem

from elkies_exact_core import build_elkies_construction

JSON_DIR = Path("testjson")
OUTPUT_PREFIX = "descent_search"

DEFAULT_SCALINGS = [
    Rational(-3, 1),
    Rational(-2, 1),
    Rational(-3, 2),
    Rational(-4, 3),
    Rational(-1, 1),
    Rational(-3, 4),
    Rational(-2, 3),
    Rational(-1, 2),
    Rational(1, 2),
    Rational(2, 3),
    Rational(3, 4),
    Rational(1, 1),
    Rational(4, 3),
    Rational(3, 2),
    Rational(2, 1),
    Rational(3, 1),
]

DEFAULT_SHIFTS = [
    Rational(-8, 1),
    Rational(-6, 1),
    Rational(-5, 1),
    Rational(-4, 1),
    Rational(-3, 1),
    Rational(-5, 2),
    Rational(-2, 1),
    Rational(-3, 2),
    Rational(-1, 1),
    Rational(-1, 2),
    Rational(0, 1),
    Rational(1, 2),
    Rational(1, 1),
    Rational(3, 2),
    Rational(2, 1),
    Rational(5, 2),
    Rational(3, 1),
    Rational(4, 1),
    Rational(5, 1),
    Rational(6, 1),
    Rational(8, 1),
]


def parse_worker_config():
    instance_id = os.environ.get("INSTANCE_ID", "1")
    try:
        worker_count = max(1, int(os.environ.get("WORKER_COUNT", "1")))
    except ValueError:
        worker_count = 1

    try:
        worker_index = max(1, int(instance_id))
    except ValueError:
        worker_index = 1

    partition_mode = (
        os.environ.get("M23_DESCENT_PARTITION_MODE", os.environ.get("M23_PARTITION_MODE", "scale_band"))
        .strip()
        .lower()
        or "scale_band"
    )
    return instance_id, worker_index, worker_count, partition_mode


def numeric_key(value):
    return float(N(value, 50))


def ordered_values(values, *, center, order):
    unique = sorted(set(values), key=numeric_key)
    if order == "descending":
        return list(reversed(unique))
    if order == "outer_in":
        return sorted(unique, key=lambda item: (-abs(numeric_key(item - center)), numeric_key(item)))
    if order == "ascending":
        return unique
    return sorted(unique, key=lambda item: (abs(numeric_key(item - center)), numeric_key(item)))


def parse_rational_list(env_name, defaults):
    raw = os.environ.get(env_name, "").strip()
    if not raw:
        return defaults

    values = []
    for chunk in raw.split(","):
        piece = chunk.strip()
        if not piece:
            continue
        try:
            values.append(Rational(piece))
        except Exception as exc:
            raise ValueError(f"Invalid rational in {env_name}: {piece}") from exc

    if not values:
        raise ValueError(f"{env_name} produced no values")
    return values


def build_affine_candidates():
    scale_order = os.environ.get("M23_DESCENT_SCALE_ORDER", "center_out").strip().lower() or "center_out"
    shift_order = os.environ.get("M23_DESCENT_SHIFT_ORDER", "center_out").strip().lower() or "center_out"

    scalings = ordered_values(
        parse_rational_list("M23_DESCENT_SCALINGS", DEFAULT_SCALINGS),
        center=Rational(1, 1),
        order=scale_order,
    )
    shifts = ordered_values(
        parse_rational_list("M23_DESCENT_SHIFTS", DEFAULT_SHIFTS),
        center=Rational(0, 1),
        order=shift_order,
    )

    center_scale_index = min(range(len(scalings)), key=lambda idx: abs(numeric_key(scalings[idx] - 1)))
    center_shift_index = min(range(len(shifts)), key=lambda idx: abs(numeric_key(shifts[idx])))

    candidates = []
    for scale_index, a in enumerate(scalings):
        for shift_index, b in enumerate(shifts):
            ring = abs(scale_index - center_scale_index) + abs(shift_index - center_shift_index)
            candidates.append(
                {
                    "a": a,
                    "b": b,
                    "scale_index": scale_index,
                    "shift_index": shift_index,
                    "scale_band": scale_index,
                    "shift_band": shift_index,
                    "ring_band": ring,
                }
            )
    return candidates, scalings, shifts


def partition_candidates(candidates, scalings, shifts):
    instance_id, worker_index, worker_count, partition_mode = parse_worker_config()
    indexed = list(enumerate(candidates))

    if not indexed:
        return [], f"Worker {instance_id} received no candidates"

    if partition_mode == "stride":
        selected = indexed[worker_index - 1 :: worker_count]
        summary = (
            f"worker {instance_id}/{worker_count} using stride partition "
            f"(offset {worker_index - 1}, span {worker_count})"
        )
    elif partition_mode == "shift_band":
        selected = [
            item for item in indexed if item[1]["shift_band"] % worker_count == (worker_index - 1)
        ]
        summary = (
            f"worker {instance_id}/{worker_count} using shift-band partition "
            f"(assigned {len(selected)} transforms across {len(shifts)} shift bands)"
        )
    elif partition_mode == "ring":
        selected = [
            item for item in indexed if item[1]["ring_band"] % worker_count == (worker_index - 1)
        ]
        summary = (
            f"worker {instance_id}/{worker_count} using ring partition "
            f"(assigned {len(selected)} transforms across center-out rings)"
        )
    elif partition_mode == "chunk":
        start = math.floor((worker_index - 1) * len(indexed) / worker_count)
        end = math.floor(worker_index * len(indexed) / worker_count)
        selected = indexed[start:end]
        summary = f"worker {instance_id}/{worker_count} scanning chunk [{start}:{end}] of {len(indexed)} transforms"
    else:
        selected = [
            item for item in indexed if item[1]["scale_band"] % worker_count == (worker_index - 1)
        ]
        summary = (
            f"worker {instance_id}/{worker_count} using scale-band partition "
            f"(assigned {len(selected)} transforms across {len(scalings)} scale bands)"
        )

    limit = int(os.environ.get("M23_DESCENT_MAX_CANDIDATES", "0") or "0")
    if limit > 0:
        selected = selected[:limit]
        summary += f", truncated to {limit}"

    return selected, summary


def reduce_in_field(expr, g, modulus):
    poly_g = Poly(expand(expr), g)
    mod_poly = Poly(modulus, g)
    reduced = rem(poly_g, mod_poly, g)
    return expand(reduced.as_expr())


def basis_vector(expr, g, modulus):
    reduced = reduce_in_field(expr, g, modulus)
    poly_g = Poly(reduced, g)
    mapping = {power[0]: coeff for power, coeff in poly_g.terms()}
    return reduced, [mapping.get(i, 0) for i in range(4)]


def coefficient_rows(poly_expr, x, g, modulus):
    poly_x = Poly(expand(poly_expr), x)
    coeffs = poly_x.all_coeffs()
    degree = poly_x.degree()
    rows = []

    for index, coeff in enumerate(coeffs):
        power = degree - index
        reduced, vector = basis_vector(coeff, g, modulus)
        rows.append(
            {
                "power": power,
                "reduced": str(reduced),
                "basis_vector": [str(item) for item in vector],
                "basis_exprs": vector,
            }
        )
    return rows


def leakage_penalty(rows):
    total = 0.0
    max_component = 0.0
    for row in rows:
        for component in row["basis_exprs"][1:]:
            size = abs(complex(N(component, 50)))
            total += size
            max_component = max(max_component, size)
    return total, max_component


def rational_height_penalty(rows):
    total = 0.0
    max_height = 0.0
    for row in rows:
        rational_component = row["basis_exprs"][0]
        size = abs(complex(N(rational_component, 50)))
        height = math.log10(1.0 + size)
        total += height
        max_height = max(max_height, height)
    return total, max_height


def denominator_penalty(rows):
    unique_denominators = set()
    total = 0.0
    max_den = 0.0

    for row in rows:
        for component in row["basis_exprs"]:
            try:
                den = int(abs(denom(component)))
            except Exception:
                den = 1
            den = max(1, den)
            unique_denominators.add(den)
            total += math.log10(den)
            max_den = max(max_den, den)

    return total, max_den, sorted(unique_denominators)


def candidate_score(rows):
    leakage_total, leakage_max = leakage_penalty(rows)
    height_total, height_max = rational_height_penalty(rows)
    denominator_total, denominator_max, denominators = denominator_penalty(rows)

    composite = (
        leakage_total
        + (0.25 * leakage_max)
        + (0.05 * height_total)
        + (0.10 * height_max)
        + (0.02 * denominator_total)
        + (0.01 * math.log10(1.0 + denominator_max))
    )
    return {
        "leakage_total": leakage_total,
        "leakage_max": leakage_max,
        "height_total": height_total,
        "height_max": height_max,
        "denominator_total": denominator_total,
        "denominator_max": denominator_max,
        "denominators": denominators,
        "descent_score": composite,
    }


def output_path_for_worker(instance_id):
    label = os.environ.get("M23_RUN_LABEL", "").strip()
    stamp = time.strftime("%Y%m%d_%H%M%S")
    parts = [OUTPUT_PREFIX, f"worker{instance_id}"]
    if label:
        parts.append(label)
    parts.append(stamp)
    return JSON_DIR / ("_".join(parts) + ".json")


def run_search():
    JSON_DIR.mkdir(exist_ok=True)

    candidates, scalings, shifts = build_affine_candidates()
    instance_id, worker_index, worker_count, partition_mode = parse_worker_config()
    selected, section_summary = partition_candidates(candidates, scalings, shifts)

    print(section_summary)
    print(f"Worker {instance_id} evaluating {len(selected)} transform(s)")
    print("Building Elkies anchor construction...")

    construction = build_elkies_construction()
    x = construction.x
    g = construction.g
    modulus = construction.field_modulus

    ranked = []
    for original_index, transform in selected:
        transformed = expand(construction.polynomial.subs(x, transform["a"] * x + transform["b"]))
        rows = coefficient_rows(transformed, x, g, modulus)
        score = candidate_score(rows)
        ranked.append(
            {
                "candidate_index": original_index,
                "transform": {
                    "a": str(transform["a"]),
                    "b": str(transform["b"]),
                    "kind": "affine",
                    "scale_index": transform["scale_index"],
                    "shift_index": transform["shift_index"],
                    "ring_band": transform["ring_band"],
                },
                **score,
                "sample_coefficients": rows[:5],
            }
        )

    ranked.sort(key=lambda item: item["descent_score"])
    payload = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "construction": "elkies_explicit",
        "search_kind": "affine_partitioned",
        "worker": {
            "instance_id": instance_id,
            "worker_index": worker_index,
            "worker_count": worker_count,
            "partition_mode": partition_mode,
            "section_summary": section_summary,
        },
        "grid": {
            "scalings": [str(value) for value in scalings],
            "shifts": [str(value) for value in shifts],
            "total_candidates": len(candidates),
            "selected_candidates": len(selected),
        },
        "ranked_transforms": ranked,
    }

    output_path = output_path_for_worker(instance_id)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    print(f"Saved descent-search results to {output_path}")
    for row in ranked[:10]:
        print(
            f"score={row['descent_score']:.6g} "
            f"leak={row['leakage_total']:.6g} "
            f"height={row['height_total']:.6g} "
            f"den_max={row['denominator_max']} "
            f"a={row['transform']['a']} b={row['transform']['b']}"
        )


if __name__ == "__main__":
    run_search()
