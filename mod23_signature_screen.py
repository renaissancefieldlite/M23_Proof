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
from pathlib import Path

from sympy import Poly, Rational, denom, expand, gcd, ilcm, resultant, symbols

from descent_search import candidate_score, coefficient_rows
from elkies_exact_core import build_elkies_construction

JSON_DIR = Path("testjson")
DEFAULT_PRIMES = [47, 139, 277, 461, 599]
ALLOWED_SIGNATURES = {
    (23,),
    (11, 12),
    (8, 15),
    (7, 8, 8),
    (5, 6, 12),
}


def basis_expr(values: list[str], g):
    coeffs = [Rational(value) for value in values]
    return expand(coeffs[0] + coeffs[1] * g + coeffs[2] * g**2 + coeffs[3] * g**3)


def is_rational_rows(rows: list[dict]) -> bool:
    for row in rows:
        if any(component != 0 for component in row["basis_exprs"][1:]):
            return False
    return True


def rational_integer_coefficients(rows: list[dict]) -> tuple[list[int], int]:
    rational_coeffs = [expand(row["basis_exprs"][0]) for row in rows]
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


def factor_degree_signature(integer_coeffs: list[int], prime: int) -> list[int]:
    x = symbols("x")
    poly = Poly(integer_coeffs, x, modulus=prime)
    _, factors = poly.factor_list()
    degrees = []
    for factor, multiplicity in factors:
        degrees.extend([int(factor.degree())] * int(multiplicity))
    return sorted(degrees)


def signature_matches(degrees: list[int]) -> bool:
    return tuple(degrees) in ALLOWED_SIGNATURES


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
        "screened_primes": [],
    }

    if not is_rational_rows(rows):
        return payload

    integer_coeffs, denominator_lcm = rational_integer_coefficients(rows)
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
        "results": results,
    }

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    rationalized = sum(1 for row in results if row["rationalized"])
    hottest = results[0] if results else None

    print(f"Saved {len(results)} screened candidate(s) to {output_path}")
    print(f"Rationalized survivors: {rationalized}/{len(results)}")
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
