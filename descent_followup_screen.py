#!/usr/bin/env python3
"""
descent_followup_screen.py - Promote top affine descent transforms into modular checks.

This bridges the live affine descent lane into the existing modular/cycle-aware
verification lane. It reads a merged descent summary, replays the top affine
transforms against the Elkies anchor, and records rationalization plus
factor-degree signatures at selected primes.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from sympy import Rational, expand

from descent_search import candidate_score, coefficient_rows
from elkies_exact_core import build_elkies_construction
from m23_cycle_signatures import annotate_prime_entry, summarize_cycle_entries
from mod23_signature_screen import (
    DEFAULT_PRIMES,
    factor_degree_signature,
    is_rational_rows,
    rational_integer_coefficients,
)

JSON_DIR = Path("testjson")


def load_summary(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def unique_top_transforms(summary: dict, followup_limit: int) -> list[dict]:
    merged = summary.get("merged_best") or []
    rows = []
    seen = set()

    for item in merged:
        source_file = item.get("source_file")
        worker = dict(item.get("worker") or {})
        top_transforms = item.get("top_transforms") or []
        for row in top_transforms:
            transform = dict(row.get("transform") or {})
            key = (transform.get("a"), transform.get("b"), transform.get("kind"))
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    "source_file": source_file,
                    "worker": worker,
                    "row": row,
                }
            )

    rows.sort(key=lambda item: float(item["row"].get("descent_score", 10**99)))
    return rows[:followup_limit]


def screen_transform(item: dict, construction, primes: list[int]) -> dict:
    row = dict(item["row"])
    transform = dict(row.get("transform") or {})
    x = construction.x
    g = construction.g
    modulus = construction.field_modulus

    a = Rational(transform["a"])
    b = Rational(transform["b"])
    transformed = expand(construction.polynomial.subs(x, a * x + b))
    coeff_rows = coefficient_rows(transformed, x, g, modulus)
    score = candidate_score(coeff_rows)

    payload = {
        "transform": transform,
        "source_file": item.get("source_file"),
        "worker": item.get("worker", {}),
        "descent_score": score["descent_score"],
        "leakage_total": score["leakage_total"],
        "leakage_max": score["leakage_max"],
        "height_total": score["height_total"],
        "height_max": score["height_max"],
        "denominator_total": score["denominator_total"],
        "denominator_max": score["denominator_max"],
        "denominators": score["denominators"],
        "rationalized": False,
        "screened_primes": [],
        "sample_coefficients": [
            {
                "power": int(entry["power"]),
                "reduced": entry["reduced"],
                "basis_vector": list(entry["basis_vector"]),
            }
            for entry in coeff_rows[:5]
        ],
    }

    if not is_rational_rows(coeff_rows):
        payload["cycle_summary"] = summarize_cycle_entries([])
        return payload

    integer_coeffs, denominator_lcm = rational_integer_coefficients(coeff_rows)
    height = max(abs(value) for value in integer_coeffs) if integer_coeffs else 0

    payload["rationalized"] = True
    payload["denominator_lcm"] = denominator_lcm
    payload["integer_height"] = int(height)
    payload["leading_coefficient"] = int(integer_coeffs[0]) if integer_coeffs else 0
    payload["integer_coefficients"] = integer_coeffs

    irreducible_prime_count = 0
    signature_hit_count = 0
    screened_entries = []
    for prime in primes:
        try:
            degrees = factor_degree_signature(integer_coeffs, prime)
            annotated = annotate_prime_entry(
                {
                    "p": prime,
                    "factor_degrees": degrees,
                    "irreducible": degrees == [23],
                }
            )
            screened_entries.append(annotated)
            if annotated.get("irreducible"):
                irreducible_prime_count += 1
            if annotated.get("m23_cycle_match"):
                signature_hit_count += 1
        except Exception as exc:
            screened_entries.append(
                {
                    "p": prime,
                    "error": str(exc),
                }
            )

    payload["screened_primes"] = screened_entries
    payload["irreducible_prime_count"] = irreducible_prime_count
    payload["signature_hit_count"] = signature_hit_count
    payload["cycle_summary"] = summarize_cycle_entries(screened_entries)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("summary_json", type=Path)
    parser.add_argument(
        "--followup-limit",
        type=int,
        default=12,
        help="maximum number of unique top transforms to promote from the summary",
    )
    parser.add_argument(
        "--primes",
        default=",".join(str(p) for p in DEFAULT_PRIMES),
        help="comma-separated primes used for modular signature checks",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="optional output path; defaults to testjson/descent_followup_TIMESTAMP.json",
    )
    args = parser.parse_args()

    JSON_DIR.mkdir(exist_ok=True)
    summary = load_summary(args.summary_json)
    primes = [int(chunk.strip()) for chunk in args.primes.split(",") if chunk.strip()]
    construction = build_elkies_construction()
    selected = unique_top_transforms(summary, args.followup_limit)

    results = [screen_transform(item, construction, primes) for item in selected]
    results.sort(
        key=lambda item: (
            0 if item["rationalized"] else 1,
            -item.get("signature_hit_count", 0),
            -item.get("irreducible_prime_count", 0),
            item["descent_score"],
        )
    )

    overall_entries = []
    for row in results:
        overall_entries.extend(row.get("screened_primes", []))

    payload = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "mode": "descent_followup",
        "summary_file": str(args.summary_json),
        "summary_label": summary.get("label"),
        "followup_limit": args.followup_limit,
        "primes": primes,
        "result_count": len(results),
        "rationalized_count": sum(1 for row in results if row.get("rationalized")),
        "results": results,
        "overall_cycle_summary": summarize_cycle_entries(overall_entries),
    }

    output_path = args.output or JSON_DIR / (
        f"descent_followup_{time.strftime('%Y%m%d_%H%M%S')}.json"
    )
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    overall = payload["overall_cycle_summary"]
    print(f"Saved descent follow-up to {output_path}")
    print(
        "Follow-up summary:",
        f"results={payload['result_count']}",
        f"rationalized={payload['rationalized_count']}",
        f"tested_primes={overall.get('tested_prime_count', 0)}",
        f"matched_m23={overall.get('matched_m23_prime_count', 0)}",
        f"cycle_rate={overall.get('exact_m23_cycle_rate', 0.0):.3f}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
