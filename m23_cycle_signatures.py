#!/usr/bin/env python3
"""
m23_cycle_signatures.py - Paper-backed M23 cycle structures and helpers.

Source:
    Noam D. Elkies, "The complex polynomials P(x) with Gal(P(x) - t) ~= M23"
    Table 1 on pages 5-6.

The key point of this module is to turn Table 1 into reusable code instead of
leaving the exact lane and modular screen to rely on ad hoc signature lists.
"""

from __future__ import annotations

from collections import Counter
from fractions import Fraction


def repeated_cycle_lengths(*spec: tuple[int, int]) -> tuple[int, ...]:
    lengths: list[int] = []
    for cycle_length, count in spec:
        lengths.extend([int(cycle_length)] * int(count))
    return tuple(sorted(lengths))


M23_CYCLE_TABLE = [
    {
        "atlas_label": "1A",
        "cycle_notation": "1^23",
        "factor_degrees": repeated_cycle_lengths((1, 23)),
        "fraction": Fraction(1, 10200960),
    },
    {
        "atlas_label": "2A",
        "cycle_notation": "1^7 2^8",
        "factor_degrees": repeated_cycle_lengths((1, 7), (2, 8)),
        "fraction": Fraction(1, 2688),
    },
    {
        "atlas_label": "3A",
        "cycle_notation": "1^5 3^6",
        "factor_degrees": repeated_cycle_lengths((1, 5), (3, 6)),
        "fraction": Fraction(1, 180),
    },
    {
        "atlas_label": "4A",
        "cycle_notation": "1^3 2^2 4^4",
        "factor_degrees": repeated_cycle_lengths((1, 3), (2, 2), (4, 4)),
        "fraction": Fraction(1, 32),
    },
    {
        "atlas_label": "5A",
        "cycle_notation": "1^3 5^4",
        "factor_degrees": repeated_cycle_lengths((1, 3), (5, 4)),
        "fraction": Fraction(1, 15),
    },
    {
        "atlas_label": "6A",
        "cycle_notation": "1 2^2 3^2 6^2",
        "factor_degrees": repeated_cycle_lengths((1, 1), (2, 2), (3, 2), (6, 2)),
        "fraction": Fraction(1, 12),
    },
    {
        "atlas_label": "7A/7B",
        "cycle_notation": "1^2 7^3",
        "factor_degrees": repeated_cycle_lengths((1, 2), (7, 3)),
        "fraction": Fraction(1, 7),
    },
    {
        "atlas_label": "8A",
        "cycle_notation": "1 2 4 8^2",
        "factor_degrees": repeated_cycle_lengths((1, 1), (2, 1), (4, 1), (8, 2)),
        "fraction": Fraction(1, 8),
    },
    {
        "atlas_label": "11A/11B",
        "cycle_notation": "1 11^2",
        "factor_degrees": repeated_cycle_lengths((1, 1), (11, 2)),
        "fraction": Fraction(2, 11),
    },
    {
        "atlas_label": "14A/14B",
        "cycle_notation": "2 7 14",
        "factor_degrees": repeated_cycle_lengths((2, 1), (7, 1), (14, 1)),
        "fraction": Fraction(1, 7),
    },
    {
        "atlas_label": "15A/15B",
        "cycle_notation": "3 5 15",
        "factor_degrees": repeated_cycle_lengths((3, 1), (5, 1), (15, 1)),
        "fraction": Fraction(2, 15),
    },
    {
        "atlas_label": "23A/23B",
        "cycle_notation": "23",
        "factor_degrees": repeated_cycle_lengths((23, 1)),
        "fraction": Fraction(2, 23),
    },
]

M23_SIGNATURE_MAP = {
    row["factor_degrees"]: row
    for row in M23_CYCLE_TABLE
}


def fixed_k_subset_count(factor_degrees: tuple[int, ...], k: int) -> int:
    """
    Elkies page 6 lemma:
    the number of fixed k-subsets is the X^k coefficient of
    product_i (1 + X^{c_i}) for cycle lengths c_i.
    """

    if k < 0:
        return 0
    dp = [0] * (k + 1)
    dp[0] = 1
    for cycle_length in factor_degrees:
        for size in range(k, cycle_length - 1, -1):
            dp[size] += dp[size - cycle_length]
    return dp[k]


def annotate_factor_degrees(
    factor_degrees: list[int] | tuple[int, ...] | None,
    max_subset_k: int = 5,
) -> dict:
    if not isinstance(factor_degrees, (list, tuple)):
        return {
            "m23_cycle_match": False,
            "m23_atlas_label": None,
            "m23_cycle_notation": None,
            "m23_fraction_num": None,
            "m23_fraction_den": None,
            "m23_fraction": None,
            "m23_k_subset_counts": {},
        }

    try:
        normalized = tuple(sorted(int(value) for value in factor_degrees))
    except Exception:
        normalized = tuple()

    row = M23_SIGNATURE_MAP.get(normalized)
    if row is None:
        return {
            "m23_cycle_match": False,
            "m23_atlas_label": None,
            "m23_cycle_notation": None,
            "m23_fraction_num": None,
            "m23_fraction_den": None,
            "m23_fraction": None,
            "m23_k_subset_counts": {
                str(k): fixed_k_subset_count(normalized, k)
                for k in range(1, max_subset_k + 1)
            },
        }

    fraction = row["fraction"]
    return {
        "m23_cycle_match": True,
        "m23_atlas_label": row["atlas_label"],
        "m23_cycle_notation": row["cycle_notation"],
        "m23_fraction_num": int(fraction.numerator),
        "m23_fraction_den": int(fraction.denominator),
        "m23_fraction": float(fraction),
        "m23_k_subset_counts": {
            str(k): fixed_k_subset_count(normalized, k)
            for k in range(1, max_subset_k + 1)
        },
    }


def annotate_prime_entry(entry: dict, factor_key: str = "factor_degrees") -> dict:
    annotated = dict(entry)
    annotated.update(annotate_factor_degrees(annotated.get(factor_key)))
    return annotated


def summarize_cycle_entries(entries: list[dict], max_subset_k: int = 5) -> dict:
    tested_prime_count = 0
    matched_m23_prime_count = 0
    skipped_prime_count = 0
    signature_histogram: Counter[str] = Counter()
    atlas_histogram: Counter[str] = Counter()
    subset_totals = {str(k): 0 for k in range(1, max_subset_k + 1)}
    matched_primes: list[int] = []
    unknown_primes: list[int] = []
    tested_primes: list[int] = []
    t0_sample_count = 0

    for raw_entry in entries or []:
        entry = annotate_prime_entry(raw_entry)
        if entry.get("skipped") or entry.get("error"):
            skipped_prime_count += 1
            continue

        factor_degrees = entry.get("factor_degrees")
        if not isinstance(factor_degrees, list):
            continue

        tested_prime_count += 1
        prime = entry.get("p")
        if isinstance(prime, int):
            tested_primes.append(prime)
        if isinstance(entry.get("t0"), int):
            t0_sample_count += 1
        signature_histogram[",".join(str(value) for value in factor_degrees)] += 1

        if entry.get("m23_cycle_match"):
            matched_m23_prime_count += 1
            atlas_label = entry.get("m23_atlas_label")
            if atlas_label:
                atlas_histogram[atlas_label] += 1
            if isinstance(prime, int):
                matched_primes.append(prime)
            for k, value in entry.get("m23_k_subset_counts", {}).items():
                subset_totals[k] = subset_totals.get(k, 0) + int(value)
        else:
            if isinstance(prime, int):
                unknown_primes.append(prime)

    exact_m23_cycle_rate = (
        matched_m23_prime_count / tested_prime_count if tested_prime_count else 0.0
    )
    unique_primes = sorted(set(tested_primes))
    distribution_rows = []
    for row in M23_CYCLE_TABLE:
        atlas_label = row["atlas_label"]
        observed = int(atlas_histogram.get(atlas_label, 0))
        expected = float(row["fraction"] * tested_prime_count)
        distribution_rows.append(
            {
                "atlas_label": atlas_label,
                "cycle_notation": row["cycle_notation"],
                "observed_count": observed,
                "expected_count_m23": expected,
                "delta": observed - expected,
            }
        )

    if tested_prime_count == 0:
        a23_exclusion_status = "not_ready"
        a23_exclusion_note = "No cycle data available yet."
    elif len(unique_primes) == 1 and t0_sample_count == tested_prime_count:
        a23_exclusion_status = "fixed_prime_sample"
        a23_exclusion_note = (
            "Cycle data now comes from many t0 values over one residue field. "
            "This is the right surface for the Table-2 / Weil-bound lane, but it "
            "is still only a sample unless the field is exhausted or the sampling "
            "plan is justified."
        )
    else:
        a23_exclusion_status = "not_ready"
        a23_exclusion_note = (
            "Current exact outputs mix primes and record one factorization per prime. "
            "Elkies' non-A23 step needs many t0 samples over one large residue field "
            "before a Table-2 / Weil-bound style contradiction can be attempted."
        )

    return {
        "tested_prime_count": tested_prime_count,
        "skipped_prime_count": skipped_prime_count,
        "matched_m23_prime_count": matched_m23_prime_count,
        "unknown_signature_count": max(0, tested_prime_count - matched_m23_prime_count),
        "exact_m23_cycle_rate": exact_m23_cycle_rate,
        "unique_prime_count": len(unique_primes),
        "sampled_primes": unique_primes,
        "t0_sample_count": t0_sample_count,
        "atlas_histogram": dict(sorted(atlas_histogram.items())),
        "signature_histogram": dict(sorted(signature_histogram.items())),
        "distribution_rows": distribution_rows,
        "subset_orbit_totals": subset_totals,
        "matched_primes": matched_primes,
        "unknown_primes": unknown_primes,
        "a23_exclusion_status": a23_exclusion_status,
        "a23_exclusion_note": a23_exclusion_note,
    }
