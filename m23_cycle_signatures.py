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
import math


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


def expected_cycle_distribution() -> dict[str, Fraction]:
    return {
        row["atlas_label"]: row["fraction"]
        for row in M23_CYCLE_TABLE
    }


def expected_fixed_k_subset_averages(max_subset_k: int = 5) -> dict[str, float]:
    expected: dict[str, float] = {}
    for k in range(1, max_subset_k + 1):
        total = 0.0
        for row in M23_CYCLE_TABLE:
            total += float(row["fraction"]) * fixed_k_subset_count(row["factor_degrees"], k)
        expected[str(k)] = total
    return expected


def expected_fixed_k_subset_statistics(max_subset_k: int = 5) -> dict[str, dict[str, float]]:
    statistics: dict[str, dict[str, float]] = {}
    for k in range(1, max_subset_k + 1):
        mean = 0.0
        second_moment = 0.0
        for row in M23_CYCLE_TABLE:
            fixed_count = fixed_k_subset_count(row["factor_degrees"], k)
            probability = float(row["fraction"])
            mean += probability * fixed_count
            second_moment += probability * (fixed_count ** 2)
        statistics[str(k)] = {
            "expected_average_m23": mean,
            "expected_second_moment_m23": second_moment,
            "expected_variance_m23": second_moment - (mean ** 2),
        }
    return statistics


def fixed_k_value_distribution(k: int) -> dict[int, Fraction]:
    distribution: dict[int, Fraction] = {}
    for row in M23_CYCLE_TABLE:
        fixed_count = fixed_k_subset_count(row["factor_degrees"], k)
        distribution[fixed_count] = distribution.get(fixed_count, Fraction(0, 1)) + row["fraction"]
    return dict(sorted(distribution.items()))


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


def expected_n5_distribution() -> dict[int, Fraction]:
    return {
        0: Fraction(4, 5),
        1: Fraction(2, 15),
        4: Fraction(1, 15),
    }


def summarize_n5_entries(entries: list[dict]) -> dict:
    expected = expected_n5_distribution()
    observed_counts: Counter[int] = Counter()
    total_samples = 0
    support_compatible = True

    for raw_entry in entries or []:
        entry = annotate_prime_entry(raw_entry)
        if entry.get("skipped") or entry.get("error"):
            continue

        factor_degrees = entry.get("factor_degrees")
        if not isinstance(factor_degrees, list):
            continue

        n5 = sum(1 for value in factor_degrees if int(value) == 5)
        observed_counts[n5] += 1
        total_samples += 1
        if n5 not in expected:
            support_compatible = False

    ordered_keys = sorted(set(expected) | set(observed_counts))
    observed_counts_json = {str(key): int(observed_counts.get(key, 0)) for key in ordered_keys}
    expected_json = {str(key): float(expected[key]) for key in sorted(expected)}

    if total_samples == 0:
        observed_frequencies_json = {str(key): 0.0 for key in ordered_keys}
        log_likelihood = None
        kl_divergence = None
        g_test_statistic = None
    else:
        observed_frequencies_json = {
            str(key): (observed_counts.get(key, 0) / total_samples)
            for key in ordered_keys
        }
        if support_compatible:
            log_likelihood = 0.0
            kl_divergence = 0.0
            for key in sorted(expected):
                count = observed_counts.get(key, 0)
                prob = float(expected[key])
                if count > 0:
                    log_likelihood += count * math.log(prob)
                    observed_prob = count / total_samples
                    kl_divergence += observed_prob * math.log(observed_prob / prob)
            g_test_statistic = 2.0 * total_samples * kl_divergence
        else:
            log_likelihood = None
            kl_divergence = None
            g_test_statistic = None

    return {
        "n5_observed_counts": observed_counts_json,
        "n5_observed_frequencies": observed_frequencies_json,
        "n5_expected_distribution_m23": expected_json,
        "n5_log_likelihood_m23": log_likelihood,
        "n5_kl_divergence_m23": kl_divergence,
        "n5_g_test_statistic_m23": g_test_statistic,
        "n5_support_compatible": bool(support_compatible),
    }


def summarize_cycle_entries(entries: list[dict], max_subset_k: int = 5) -> dict:
    tested_prime_count = 0
    matched_m23_prime_count = 0
    skipped_prime_count = 0
    signature_histogram: Counter[str] = Counter()
    atlas_histogram: Counter[str] = Counter()
    subset_totals = {str(k): 0 for k in range(1, max_subset_k + 1)}
    subset_square_totals = {str(k): 0 for k in range(1, max_subset_k + 1)}
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
        for k, value in entry.get("m23_k_subset_counts", {}).items():
            int_value = int(value)
            subset_totals[k] = subset_totals.get(k, 0) + int_value
            subset_square_totals[k] = subset_square_totals.get(k, 0) + (int_value * int_value)

        if entry.get("m23_cycle_match"):
            matched_m23_prime_count += 1
            atlas_label = entry.get("m23_atlas_label")
            if atlas_label:
                atlas_histogram[atlas_label] += 1
            if isinstance(prime, int):
                matched_primes.append(prime)
        else:
            if isinstance(prime, int):
                unknown_primes.append(prime)

    exact_m23_cycle_rate = (
        matched_m23_prime_count / tested_prime_count if tested_prime_count else 0.0
    )
    unique_primes = sorted(set(tested_primes))
    distribution_rows = []
    expected_distribution = expected_cycle_distribution()
    expected_subset_statistics = expected_fixed_k_subset_statistics(max_subset_k=max_subset_k)
    observed_frequency_histogram: dict[str, float] = {}
    expected_distribution_json = {
        atlas_label: float(fraction)
        for atlas_label, fraction in expected_distribution.items()
    }
    for row in M23_CYCLE_TABLE:
        atlas_label = row["atlas_label"]
        observed = int(atlas_histogram.get(atlas_label, 0))
        expected = float(row["fraction"] * tested_prime_count)
        observed_frequency = (
            observed / tested_prime_count if tested_prime_count else 0.0
        )
        expected_frequency = float(row["fraction"])
        observed_frequency_histogram[atlas_label] = observed_frequency
        distribution_rows.append(
            {
                "atlas_label": atlas_label,
                "cycle_notation": row["cycle_notation"],
                "observed_count": observed,
                "observed_frequency": observed_frequency,
                "expected_count_m23": expected,
                "expected_frequency_m23": expected_frequency,
                "delta": observed - expected,
            }
        )

    full_cycle_support_compatible = (
        tested_prime_count > 0 and matched_m23_prime_count == tested_prime_count
    )
    if full_cycle_support_compatible:
        full_cycle_log_likelihood_m23 = 0.0
        full_cycle_kl_divergence_m23 = 0.0
        for row in M23_CYCLE_TABLE:
            atlas_label = row["atlas_label"]
            observed = int(atlas_histogram.get(atlas_label, 0))
            expected_prob = float(expected_distribution[atlas_label])
            if observed > 0:
                full_cycle_log_likelihood_m23 += observed * math.log(expected_prob)
                observed_prob = observed / tested_prime_count
                full_cycle_kl_divergence_m23 += observed_prob * math.log(
                    observed_prob / expected_prob
                )
        full_cycle_g_test_statistic_m23 = 2.0 * tested_prime_count * full_cycle_kl_divergence_m23
    else:
        full_cycle_log_likelihood_m23 = None
        full_cycle_kl_divergence_m23 = None
        full_cycle_g_test_statistic_m23 = None

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

    subset_observed_averages = {}
    subset_average_rows = []
    for k in range(1, max_subset_k + 1):
        key = str(k)
        observed_total = int(subset_totals.get(key, 0))
        observed_square_total = int(subset_square_totals.get(key, 0))
        observed_average = (
            observed_total / tested_prime_count if tested_prime_count else None
        )
        if tested_prime_count > 1:
            observed_variance = (
                observed_square_total
                - ((observed_total ** 2) / tested_prime_count)
            ) / (tested_prime_count - 1)
        else:
            observed_variance = None
        expected_average = expected_subset_statistics[key]["expected_average_m23"]
        expected_variance = expected_subset_statistics[key]["expected_variance_m23"]
        subset_observed_averages[key] = observed_average
        subset_average_rows.append(
            {
                "k": k,
                "observed_total": observed_total,
                "observed_square_total": observed_square_total,
                "observed_average": observed_average,
                "expected_average_m23": expected_average,
                "delta": (
                    observed_average - expected_average
                    if observed_average is not None
                    else None
                ),
                "observed_variance": observed_variance,
                "expected_variance_m23": expected_variance,
                "variance_delta": (
                    observed_variance - expected_variance
                    if observed_variance is not None
                    else None
                ),
            }
        )

    transitivity_rows = [row for row in subset_average_rows if int(row["k"]) <= 4]
    transitivity_summary = {
        "k_values": [int(row["k"]) for row in transitivity_rows],
        "max_abs_delta_k1_to_k4": (
            max(abs(float(row["delta"])) for row in transitivity_rows)
            if transitivity_rows
            else None
        ),
        "mean_abs_delta_k1_to_k4": (
            sum(abs(float(row["delta"])) for row in transitivity_rows) / len(transitivity_rows)
            if transitivity_rows
            else None
        ),
    }
    k5_focus = next(
        (row for row in subset_average_rows if int(row["k"]) == 5),
        None,
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
        "atlas_observed_frequencies": dict(sorted(observed_frequency_histogram.items())),
        "atlas_expected_distribution_m23": dict(sorted(expected_distribution_json.items())),
        "signature_histogram": dict(sorted(signature_histogram.items())),
        "distribution_rows": distribution_rows,
        "full_cycle_support_compatible": bool(full_cycle_support_compatible),
        "full_cycle_log_likelihood_m23": full_cycle_log_likelihood_m23,
        "full_cycle_kl_divergence_m23": full_cycle_kl_divergence_m23,
        "full_cycle_g_test_statistic_m23": full_cycle_g_test_statistic_m23,
        "subset_orbit_totals": subset_totals,
        "subset_square_totals": subset_square_totals,
        "subset_observed_averages": subset_observed_averages,
        "subset_expected_averages_m23": {
            key: value["expected_average_m23"]
            for key, value in expected_subset_statistics.items()
        },
        "subset_expected_variances_m23": {
            key: value["expected_variance_m23"]
            for key, value in expected_subset_statistics.items()
        },
        "subset_average_rows": subset_average_rows,
        "transitivity_summary": transitivity_summary,
        "k5_focus": k5_focus,
        "matched_primes": matched_primes,
        "unknown_primes": unknown_primes,
        "a23_exclusion_status": a23_exclusion_status,
        "a23_exclusion_note": a23_exclusion_note,
    }
