#!/usr/bin/env python3
"""
mod23_signature_screen.py - Screen bounded Elkies descent transforms.

This module is the filter side of the canonical hook:
  generator: bounded Tschirnhaus transforms over F
  filter: leakage/height/denominator checks + modular factorization signatures

The current implementation is intentionally first-pass practical:
  - exact basis leakage is computed over (1, g, g^2, g^3)
  - only fully rationalized candidates are promoted into modular screening
  - modular screening records factor-degree signatures at actual primes
    p == 1 mod 23
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

from sympy import Poly, Rational, denom, expand, gcd, ilcm, resultant, symbols

from descent_search import candidate_score, coefficient_rows
from elkies_exact_core import build_elkies_construction
from m23_cycle_signatures import annotate_factor_degrees

JSON_DIR = Path("testjson")
DEFAULT_PRIMES = [47, 139, 277, 461, 599]


def basis_expr(values: list[str], g):
    coeffs = [Rational(value) for value in values]
    return expand(coeffs[0] + coeffs[1] * g + coeffs[2] * g**2 + coeffs[3] * g**3)


def is_rational_rows(rows: list[dict]) -> bool:
    for row in rows:
        if any(component != 0 for component in row["basis_exprs"][1:]):
            return False
    return True


def rational_integer_coefficients_from_values(rational_coeffs: list) -> tuple[list[int], int]:
    denominator_lcm = 1
    for coeff in rational_coeffs:
        denominator_lcm = ilcm(denominator_lcm, int(abs(denom(coeff))))
    denominator_lcm = max(1, int(denominator_lcm))
    integer_coeffs = [int(expand(coeff * denominator_lcm)) for coeff in rational_coeffs]
    common_divisor = 0
    for value in integer_coeffs:
        common_divisor = gcd(common_divisor, value)
    common_divisor = int(abs(common_divisor)) if common_divisor else 1
    if common_divisor > 1:
        integer_coeffs = [value // common_divisor for value in integer_coeffs]
    return integer_coeffs, denominator_lcm


def rational_integer_coefficients(rows: list[dict]) -> tuple[list[int], int]:
    rational_coeffs = [expand(row["basis_exprs"][0]) for row in rows]
    return rational_integer_coefficients_from_values(rational_coeffs)


def basis_support_histogram(rows: list[dict]) -> dict[str, int]:
    histogram: Counter[int] = Counter()
    for row in rows:
        support = sum(1 for component in row["basis_exprs"] if component != 0)
        histogram[support] += 1
    return {str(key): int(histogram[key]) for key in sorted(histogram)}


def rationalization_diagnostics(rows: list[dict]) -> dict:
    non_rational_rows = []
    non_rational_component_count = 0
    for row in rows:
        non_rational_positions = [
            index for index, component in enumerate(row["basis_exprs"][1:], start=1) if component != 0
        ]
        if non_rational_positions:
            non_rational_component_count += len(non_rational_positions)
            non_rational_rows.append(
                {
                    "power": int(row["power"]),
                    "non_rational_basis_positions": non_rational_positions,
                    "basis_vector": [str(component) for component in row["basis_exprs"]],
                }
            )

    return {
        "basis_support_histogram": basis_support_histogram(rows),
        "non_rational_row_count": len(non_rational_rows),
        "non_rational_component_count": int(non_rational_component_count),
        "first_non_rational_rows": non_rational_rows[:5],
    }


def common_scalar_rationalization(rows: list[dict]) -> dict | None:
    nonzero_rows = [row for row in rows if any(component != 0 for component in row["basis_exprs"])]
    if not nonzero_rows:
        rational_coeffs = [Rational(0) for _ in rows]
        integer_coeffs, denominator_lcm = rational_integer_coefficients_from_values(rational_coeffs)
        return {
            "mode": "common_scalar",
            "scalar_expr": "0",
            "scalar_basis_vector": ["0", "0", "0", "0"],
            "rational_coefficients": rational_coeffs,
            "integer_coefficients": integer_coeffs,
            "denominator_lcm": denominator_lcm,
        }

    base_row = nonzero_rows[0]
    base_vector = base_row["basis_exprs"]
    pivot_index = next((index for index, value in enumerate(base_vector) if value != 0), None)
    if pivot_index is None:
        return None

    rational_coeffs = []
    for row in rows:
        vector = row["basis_exprs"]
        if all(component == 0 for component in vector):
            rational_coeffs.append(Rational(0))
            continue

        ratio = expand(vector[pivot_index] / base_vector[pivot_index])
        for component, base_component in zip(vector, base_vector):
            if expand(component - ratio * base_component) != 0:
                return None
        rational_coeffs.append(ratio)

    integer_coeffs, denominator_lcm = rational_integer_coefficients_from_values(rational_coeffs)
    return {
        "mode": "common_scalar",
        "scalar_expr": str(base_row["reduced"]),
        "scalar_basis_vector": [str(component) for component in base_vector],
        "rational_coefficients": rational_coeffs,
        "integer_coefficients": integer_coeffs,
        "denominator_lcm": denominator_lcm,
    }


def factor_degree_signature(integer_coeffs: list[int], prime: int) -> list[int]:
    x = symbols("x")
    poly = Poly(integer_coeffs, x, modulus=prime)
    _, factors = poly.factor_list()
    degrees = []
    for factor, multiplicity in factors:
        degrees.extend([int(factor.degree())] * int(multiplicity))
    return sorted(degrees)


def signature_matches(degrees: list[int]) -> bool:
    return bool(annotate_factor_degrees(degrees).get("m23_cycle_match"))


def transformed_polynomial(candidate: dict, construction):
    g = construction.g
    z = construction.x
    a = basis_expr(candidate["a_basis"], g)
    b = basis_expr(candidate["b_basis"], g)
    family = candidate["family"]

    if family == "affine_f_basis":
        return expand(construction.polynomial.subs(z, a * z + b))

    if family == "quadratic_tschirnhaus_f_basis":
        y = symbols("y")
        source = build_elkies_construction(x_symbol=y, g_symbol=g)
        h = y**2 + a * y + b
        return expand(resultant(source.polynomial, z - h, y))

    raise ValueError(f"unsupported family: {family}")


def screen_candidate(candidate: dict, construction, primes: list[int]) -> dict:
    x = construction.x
    modulus = construction.field_modulus

    transformed = transformed_polynomial(candidate, construction)
    rows = coefficient_rows(transformed, x, construction.g, modulus)
    score = candidate_score(rows)

    payload = {
        "candidate": candidate,
        **score,
        "rationalized": False,
        "rationalization_mode": "none",
        "rationalization_diagnostics": {
            **rationalization_diagnostics(rows),
            "common_scalar_rationalizable": False,
        },
        "screened_primes": [],
    }

    if is_rational_rows(rows):
        integer_coeffs, denominator_lcm = rational_integer_coefficients(rows)
        payload["rationalization_mode"] = "direct"
    else:
        common_scalar = common_scalar_rationalization(rows)
        payload["rationalization_diagnostics"]["common_scalar_rationalizable"] = bool(common_scalar)
        if common_scalar is None:
            return payload
        integer_coeffs = common_scalar["integer_coefficients"]
        denominator_lcm = common_scalar["denominator_lcm"]
        payload["rationalization_mode"] = "common_scalar"
        payload["rationalization_scalar_expr"] = common_scalar["scalar_expr"]
        payload["rationalization_scalar_basis_vector"] = common_scalar["scalar_basis_vector"]

    if not integer_coeffs:
        return payload

    height = max(abs(value) for value in integer_coeffs) if integer_coeffs else 0

    payload["rationalized"] = True
    payload["denominator_lcm"] = denominator_lcm
    payload["integer_height"] = int(height)
    payload["leading_coefficient"] = int(integer_coeffs[0]) if integer_coeffs else 0
    payload["integer_coefficients"] = integer_coeffs

    irreducible_prime_count = 0
    signature_hit_count = 0
    for prime in primes:
        try:
            degrees = factor_degree_signature(integer_coeffs, prime)
            cycle_annotation = annotate_factor_degrees(degrees)
            if degrees == [23]:
                irreducible_prime_count += 1
            if signature_matches(degrees):
                signature_hit_count += 1
            payload["screened_primes"].append(
                {
                    "p": prime,
                    "factor_degrees": degrees,
                    "irreducible": degrees == [23],
                    "signature_match": signature_matches(degrees),
                    "m23_cycle_match": cycle_annotation.get("m23_cycle_match"),
                    "m23_atlas_label": cycle_annotation.get("m23_atlas_label"),
                    "m23_cycle_notation": cycle_annotation.get("m23_cycle_notation"),
                }
            )
        except Exception as exc:
            payload["screened_primes"].append(
                {
                    "p": prime,
                    "error": str(exc),
                }
            )

    payload["irreducible_prime_count"] = irreducible_prime_count
    payload["signature_hit_count"] = signature_hit_count
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate_json", type=Path)
    parser.add_argument(
        "--screen-limit",
        type=int,
        default=16,
        help="number of generated candidates to evaluate from the file",
    )
    parser.add_argument(
        "--primes",
        default=",".join(str(p) for p in DEFAULT_PRIMES),
        help="comma-separated primes p == 1 mod 23 used for modular signatures",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="optional output path; defaults to testjson/mod23_screen_workerX_TIMESTAMP.json",
    )
    args = parser.parse_args()

    JSON_DIR.mkdir(exist_ok=True)
    with args.candidate_json.open("r", encoding="utf-8") as handle:
        candidate_payload = json.load(handle)

    candidates = candidate_payload.get("candidates", [])[: args.screen_limit]
    primes = [int(chunk.strip()) for chunk in args.primes.split(",") if chunk.strip()]
    construction = build_elkies_construction()

    results = []
    for candidate in candidates:
        results.append(screen_candidate(candidate, construction, primes))

    results.sort(
        key=lambda item: (
            0 if item["rationalized"] else 1,
            -item.get("signature_hit_count", 0),
            -item.get("irreducible_prime_count", 0),
            item["descent_score"],
        )
    )

    instance_id = candidate_payload.get("worker", {}).get("instance_id", "1")
    output_path = args.output or JSON_DIR / (
        f"mod23_screen_worker{instance_id}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    )
    payload = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "construction": "elkies_explicit",
        "generator_file": str(args.candidate_json),
        "screen_limit": len(candidates),
        "primes": primes,
        "rationalization_summary": {
            "rationalized_count": sum(1 for row in results if row.get("rationalized")),
            "direct_rationalized_count": sum(1 for row in results if row.get("rationalization_mode") == "direct"),
            "common_scalar_rationalized_count": sum(
                1 for row in results if row.get("rationalization_mode") == "common_scalar"
            ),
            "common_scalar_opportunity_count": sum(
                1
                for row in results
                if row.get("rationalization_diagnostics", {}).get("common_scalar_rationalizable")
            ),
        },
        "results": results,
    }

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    rationalized_summary = payload["rationalization_summary"]
    rationalized = rationalized_summary["rationalized_count"]
    hottest = results[0] if results else None

    print(f"Saved {len(results)} screened candidate(s) to {output_path}")
    print(
        "Rationalization summary:",
        f"rationalized={rationalized}/{len(results)}",
        f"direct={rationalized_summary['direct_rationalized_count']}",
        f"common_scalar={rationalized_summary['common_scalar_rationalized_count']}",
        f"common_scalar_opportunities={rationalized_summary['common_scalar_opportunity_count']}",
    )
    if hottest is not None:
        print(
            "Top candidate:",
            f"descent_score={hottest['descent_score']:.6g}",
            f"rationalized={hottest['rationalized']}",
            f"signature_hits={hottest.get('signature_hit_count', 0)}",
            f"irreducible_primes={hottest.get('irreducible_prime_count', 0)}",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
