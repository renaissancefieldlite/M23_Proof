#!/usr/bin/env python3
"""
m23_cycle_verifier.py - Annotate cycle signatures in M23 result files.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import time
from fractions import Fraction
from pathlib import Path

from m23_cycle_signatures import (
    annotate_prime_entry,
    expected_cycle_distribution,
    fixed_k_value_distribution,
    summarize_cycle_entries,
    summarize_n5_entries,
)

JSON_DIR = Path("testjson")


def latest_path(pattern: str) -> Path | None:
    matches = sorted(JSON_DIR.glob(pattern), key=os.path.getmtime)
    return matches[-1] if matches else None


def default_input() -> Path | None:
    for pattern in (
        "m23_fixed_prime_sample_*.json",
        "elkies_exact_results_*.json",
        "exact_test_results_*.json",
        "mod23_screen_*.json",
    ):
        path = latest_path(pattern)
        if path is not None:
            return path
    return None


def verify_elkies_payload(payload: dict) -> dict:
    result = dict(payload.get("result") or {})
    per_prime = [annotate_prime_entry(entry) for entry in result.get("per_prime", [])]
    result["per_prime"] = per_prime
    result["cycle_summary"] = summarize_cycle_entries(per_prime)
    return {
        "mode": "elkies_exact",
        "result": result,
    }


def verify_fixed_prime_payload(payload: dict) -> dict:
    result = dict(payload.get("result") or {})
    samples = [annotate_prime_entry(entry) for entry in result.get("samples", [])]
    result["samples"] = samples
    result["cycle_summary"] = summarize_cycle_entries(samples)
    result.update(summarize_n5_entries(samples))
    return {
        "mode": "fixed_prime_sample",
        "result": result,
    }


def verify_exact_scan_payload(payload: list[dict]) -> dict:
    rows = []
    for row in payload:
        item = dict(row)
        result = dict(item.get("result") or {})
        per_prime = [annotate_prime_entry(entry) for entry in result.get("per_prime", [])]
        result["per_prime"] = per_prime
        result["cycle_summary"] = summarize_cycle_entries(per_prime)
        item["result"] = result
        rows.append(item)

    overall_entries = []
    for row in rows:
        overall_entries.extend(row.get("result", {}).get("per_prime", []))

    return {
        "mode": "exact_scan",
        "rows": rows,
        "overall_cycle_summary": summarize_cycle_entries(overall_entries),
    }


def verify_screen_payload(payload: dict) -> dict:
    results = []
    overall_entries = []
    for row in payload.get("results", []):
        item = dict(row)
        screened = [annotate_prime_entry(entry) for entry in item.get("screened_primes", [])]
        item["screened_primes"] = screened
        item["cycle_summary"] = summarize_cycle_entries(screened)
        overall_entries.extend(screened)
        results.append(item)

    return {
        "mode": "mod23_screen",
        "results": results,
        "overall_cycle_summary": summarize_cycle_entries(overall_entries),
    }


def load_payload(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_fixed_prime_result(path: Path) -> dict:
    payload = load_payload(path)

    if isinstance(payload, dict) and isinstance(payload.get("result"), dict):
        result = payload.get("result") or {}
        if isinstance(result.get("samples"), list):
            return verify_fixed_prime_payload(payload)["result"]

    if isinstance(payload, dict) and isinstance(payload.get("report"), dict):
        report = payload.get("report") or {}
        if report.get("mode") == "fixed_prime_sample" and isinstance(report.get("result"), dict):
            return verify_fixed_prime_payload({"result": report.get("result") or {}})["result"]

    raise SystemExit(f"Unsupported fixed-prime payload for combine mode: {path}")


def expand_input_patterns(patterns: list[str]) -> list[Path]:
    resolved: list[Path] = []
    seen: set[Path] = set()
    for pattern in patterns:
        matches = sorted(Path(match) for match in glob.glob(pattern))
        if not matches and Path(pattern).exists():
            matches = [Path(pattern)]
        for match in matches:
            normalized = match.resolve()
            if normalized not in seen:
                seen.add(normalized)
                resolved.append(normalized)
    return resolved


def expected_log_likelihood_per_sample() -> float:
    expected = expected_cycle_distribution()
    return sum(
        float(probability) * math.log(float(probability))
        for probability in expected.values()
    )


def exact_fixed_k_lower_tail_probability(
    sample_count: int,
    observed_total: int,
    value_distribution: dict[int, Fraction],
) -> Fraction:
    if sample_count <= 0:
        return Fraction(1, 1)
    if observed_total < 0:
        return Fraction(0, 1)

    dp: dict[int, Fraction] = {0: Fraction(1, 1)}
    for _ in range(sample_count):
        next_dp: dict[int, Fraction] = {}
        for running_total, running_prob in dp.items():
            for value, probability in value_distribution.items():
                new_total = running_total + int(value)
                if new_total > observed_total:
                    continue
                next_dp[new_total] = next_dp.get(new_total, Fraction(0, 1)) + (
                    running_prob * probability
                )
        dp = next_dp
        if not dp:
            return Fraction(0, 1)

    return sum(dp.values(), Fraction(0, 1))


def combine_fixed_prime_results(paths: list[Path]) -> dict:
    if not paths:
        raise SystemExit("Combine mode did not receive any matching fixed-prime sample files.")

    per_prime = []
    combined_samples = []
    for path in paths:
        result = load_fixed_prime_result(path)
        samples = result.get("samples", [])
        summary = result.get("cycle_summary", {})
        prime_value = result.get("prime")
        tested_count = int(summary.get("tested_prime_count", 0))

        per_prime.append(
            {
                "input_file": str(path),
                "prime": int(prime_value) if isinstance(prime_value, int) else prime_value,
                "tested_sample_count": tested_count,
                "full_cycle_support_compatible": bool(
                    summary.get("full_cycle_support_compatible", False)
                ),
                "full_cycle_log_likelihood_m23": summary.get("full_cycle_log_likelihood_m23"),
                "full_cycle_kl_divergence_m23": summary.get("full_cycle_kl_divergence_m23"),
                "unknown_signature_count": int(summary.get("unknown_signature_count", 0)),
                "exact_m23_cycle_rate": summary.get("exact_m23_cycle_rate", 0.0),
            }
        )
        combined_samples.extend(samples)

    combined_cycle_summary = summarize_cycle_entries(combined_samples)
    combined_n5_summary = summarize_n5_entries(combined_samples)
    total_samples = int(combined_cycle_summary.get("tested_prime_count", 0))
    unique_prime_count = int(combined_cycle_summary.get("unique_prime_count", 0))
    k5_focus = combined_cycle_summary.get("k5_focus")
    if isinstance(k5_focus, dict):
        k5_observed_total = int(k5_focus.get("observed_total", 0))
        k5_distribution = fixed_k_value_distribution(5)
        k5_tail_fraction = exact_fixed_k_lower_tail_probability(
            sample_count=total_samples,
            observed_total=k5_observed_total,
            value_distribution=k5_distribution,
        )
        combined_cycle_summary["k5_tail_summary"] = {
            "observed_total": k5_observed_total,
            "sample_count": total_samples,
            "lower_tail_probability_num": int(k5_tail_fraction.numerator),
            "lower_tail_probability_den": int(k5_tail_fraction.denominator),
            "lower_tail_probability": float(k5_tail_fraction),
        }

    if unique_prime_count < 2:
        verdict = "insufficient_prime_count"
    elif not combined_cycle_summary.get("full_cycle_support_compatible", False):
        verdict = "reject_unknown_cycle_type"
    else:
        verdict = "multi_prime_m23_cycle_compatible"

    return {
        "mode": "fixed_prime_multi_prime_combine",
        "source_file_count": len(paths),
        "source_files": [str(path) for path in paths],
        "unique_prime_count": unique_prime_count,
        "primes": [
            item.get("prime")
            for item in per_prime
        ],
        "per_prime": per_prime,
        "combined_cycle_summary": combined_cycle_summary,
        "combined_n5_summary": combined_n5_summary,
        "combined_log_likelihood_m23": combined_cycle_summary.get(
            "full_cycle_log_likelihood_m23"
        ),
        "combined_kl_divergence_m23": combined_cycle_summary.get(
            "full_cycle_kl_divergence_m23"
        ),
        "combined_g_test_statistic_m23": combined_cycle_summary.get(
            "full_cycle_g_test_statistic_m23"
        ),
        "expected_combined_log_likelihood_m23": (
            total_samples * expected_log_likelihood_per_sample()
            if total_samples > 0
            else None
        ),
        "verdict": verdict,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_json", nargs="?", type=Path, default=None)
    parser.add_argument(
        "--combine",
        nargs="+",
        default=None,
        help="combine multiple fixed-prime sample JSON files or glob patterns into one report",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="optional output path; defaults to testjson/m23_cycle_report_TIMESTAMP.json",
    )
    args = parser.parse_args()

    if args.combine:
        input_paths = expand_input_patterns(args.combine)
        report = combine_fixed_prime_results(input_paths)
        summary = report["combined_cycle_summary"]
        input_label = ",".join(str(path) for path in input_paths)
    else:
        input_path = args.input_json or default_input()
        if input_path is None:
            raise SystemExit("No compatible JSON result file found under testjson/.")

        payload = load_payload(input_path)
        if isinstance(payload, dict) and isinstance(payload.get("result"), dict):
            result = payload.get("result") or {}
            if isinstance(result.get("samples"), list):
                report = verify_fixed_prime_payload(payload)
            else:
                report = verify_elkies_payload(payload)
            summary = report["result"]["cycle_summary"]
        elif isinstance(payload, list):
            report = verify_exact_scan_payload(payload)
            summary = report["overall_cycle_summary"]
        elif isinstance(payload, dict) and isinstance(payload.get("results"), list):
            report = verify_screen_payload(payload)
            summary = report["overall_cycle_summary"]
        else:
            raise SystemExit(f"Unsupported result format: {input_path}")
        input_label = str(input_path)

    output_path = args.output or JSON_DIR / (
        f"m23_cycle_report_{time.strftime('%Y%m%d_%H%M%S')}.json"
    )
    wrapped = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "input_file": input_label,
        "report": report,
    }
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(wrapped, handle, indent=2)

    print(f"Saved M23 cycle report to {output_path}")
    if report.get("mode") == "fixed_prime_multi_prime_combine":
        print(
            "Combined cycle summary:",
            f"source_files={report.get('source_file_count', 0)}",
            f"unique_primes={report.get('unique_prime_count', 0)}",
            f"tested_samples={summary.get('tested_prime_count', 0)}",
            f"matched_m23={summary.get('matched_m23_prime_count', 0)}",
            f"cycle_rate={summary.get('exact_m23_cycle_rate', 0.0):.3f}",
        )
        print(
            "Combined full-cycle metrics:",
            f"verdict={report.get('verdict')}",
            f"support_compatible={summary.get('full_cycle_support_compatible', False)}",
            f"logL={report.get('combined_log_likelihood_m23')}",
            f"expected_logL={report.get('expected_combined_log_likelihood_m23')}",
            f"KL={report.get('combined_kl_divergence_m23')}",
            f"G={report.get('combined_g_test_statistic_m23')}",
            f"unknown_signatures={summary.get('unknown_signature_count', 0)}",
        )
        print("Per-prime contributions:")
        for row in report.get("per_prime", []):
            print(
                " ",
                f"p={row.get('prime')}",
                f"samples={row.get('tested_sample_count', 0)}",
                f"support_compatible={row.get('full_cycle_support_compatible', False)}",
                f"logL={row.get('full_cycle_log_likelihood_m23')}",
                f"KL={row.get('full_cycle_kl_divergence_m23')}",
                f"unknown_signatures={row.get('unknown_signature_count', 0)}",
            )
        print("Combined full-cycle expected vs observed (M23):")
        for row in summary.get("distribution_rows", []):
            print(
                " ",
                f"{row.get('atlas_label')}:",
                f"obs={row.get('observed_count', 0)}",
                f"obs_freq={row.get('observed_frequency', 0.0):.4f}",
                f"exp={row.get('expected_count_m23', 0.0):.3f}",
                f"exp_freq={row.get('expected_frequency_m23', 0.0):.4f}",
                f"delta={row.get('delta', 0.0):+.3f}",
                f"cycle={row.get('cycle_notation')}",
            )
        print("Fixed-k subset averages (Table-2 lane):")
        for row in summary.get("subset_average_rows", []):
            print(
                " ",
                f"k={row.get('k')}:",
                f"obs_total={row.get('observed_total')}",
                f"obs_avg={row.get('observed_average')}",
                f"exp_avg={row.get('expected_average_m23')}",
                f"delta={row.get('delta')}",
                f"obs_var={row.get('observed_variance')}",
                f"exp_var={row.get('expected_variance_m23')}",
                f"var_delta={row.get('variance_delta')}",
            )
        transitivity = summary.get("transitivity_summary", {})
        print(
            "Transitivity summary:",
            f"k_values={transitivity.get('k_values')}",
            f"max_abs_delta={transitivity.get('max_abs_delta_k1_to_k4')}",
            f"mean_abs_delta={transitivity.get('mean_abs_delta_k1_to_k4')}",
        )
        k5_focus = summary.get("k5_focus")
        if isinstance(k5_focus, dict):
            print(
                "K5 focus:",
                f"obs_avg={k5_focus.get('observed_average')}",
                f"exp_avg={k5_focus.get('expected_average_m23')}",
                f"delta={k5_focus.get('delta')}",
                f"obs_var={k5_focus.get('observed_variance')}",
                f"exp_var={k5_focus.get('expected_variance_m23')}",
                f"var_delta={k5_focus.get('variance_delta')}",
            )
        k5_tail = summary.get("k5_tail_summary")
        if isinstance(k5_tail, dict):
            print(
                "K5 exact tail:",
                f"observed_total={k5_tail.get('observed_total')}",
                f"samples={k5_tail.get('sample_count')}",
                f"p_lower={k5_tail.get('lower_tail_probability')}",
                f"exact={k5_tail.get('lower_tail_probability_num')}/{k5_tail.get('lower_tail_probability_den')}",
            )
        combined_n5 = report.get("combined_n5_summary", {})
        print(
            "Combined N5 metrics:",
            f"support_compatible={combined_n5.get('n5_support_compatible', False)}",
            f"logL={combined_n5.get('n5_log_likelihood_m23')}",
            f"KL={combined_n5.get('n5_kl_divergence_m23')}",
            f"G={combined_n5.get('n5_g_test_statistic_m23')}",
        )
        return 0

    print(
        "Cycle summary:",
        f"tested_primes={summary.get('tested_prime_count', 0)}",
        f"matched_m23={summary.get('matched_m23_prime_count', 0)}",
        f"cycle_rate={summary.get('exact_m23_cycle_rate', 0.0):.3f}",
    )
    print(
        "Full-cycle metrics:",
        f"support_compatible={summary.get('full_cycle_support_compatible', False)}",
        f"logL={summary.get('full_cycle_log_likelihood_m23')}",
        f"KL={summary.get('full_cycle_kl_divergence_m23')}",
        f"G={summary.get('full_cycle_g_test_statistic_m23')}",
        f"unknown_signatures={summary.get('unknown_signature_count', 0)}",
    )
    print("Full-cycle expected vs observed (M23):")
    for row in summary.get("distribution_rows", []):
        print(
            " ",
            f"{row.get('atlas_label')}:",
            f"obs={row.get('observed_count', 0)}",
            f"obs_freq={row.get('observed_frequency', 0.0):.4f}",
            f"exp={row.get('expected_count_m23', 0.0):.3f}",
            f"exp_freq={row.get('expected_frequency_m23', 0.0):.4f}",
            f"delta={row.get('delta', 0.0):+.3f}",
            f"cycle={row.get('cycle_notation')}",
        )
    print("Fixed-k subset averages (Table-2 lane):")
    for row in summary.get("subset_average_rows", []):
        print(
            " ",
            f"k={row.get('k')}:",
            f"obs_total={row.get('observed_total')}",
            f"obs_avg={row.get('observed_average')}",
            f"exp_avg={row.get('expected_average_m23')}",
            f"delta={row.get('delta')}",
            f"obs_var={row.get('observed_variance')}",
            f"exp_var={row.get('expected_variance_m23')}",
            f"var_delta={row.get('variance_delta')}",
        )
    transitivity = summary.get("transitivity_summary", {})
    print(
        "Transitivity summary:",
        f"k_values={transitivity.get('k_values')}",
        f"max_abs_delta={transitivity.get('max_abs_delta_k1_to_k4')}",
        f"mean_abs_delta={transitivity.get('mean_abs_delta_k1_to_k4')}",
    )
    k5_focus = summary.get("k5_focus")
    if isinstance(k5_focus, dict):
        print(
            "K5 focus:",
            f"obs_avg={k5_focus.get('observed_average')}",
            f"exp_avg={k5_focus.get('expected_average_m23')}",
            f"delta={k5_focus.get('delta')}",
            f"obs_var={k5_focus.get('observed_variance')}",
            f"exp_var={k5_focus.get('expected_variance_m23')}",
            f"var_delta={k5_focus.get('variance_delta')}",
        )
    if report.get("mode") == "fixed_prime_sample":
        fixed_result = report.get("result", {})
        n5_counts = fixed_result.get("n5_observed_counts", {})
        n5_expected = fixed_result.get("n5_expected_distribution_m23", {})
        print(
            "N5 summary:",
            f"counts=0:{n5_counts.get('0', 0)},1:{n5_counts.get('1', 0)},4:{n5_counts.get('4', 0)}",
        )
        print(
            "N5 expected (M23):",
            f"0:{n5_expected.get('0', 0.0):.4f}",
            f"1:{n5_expected.get('1', 0.0):.4f}",
            f"4:{n5_expected.get('4', 0.0):.4f}",
        )
        print(
            "N5 metrics:",
            f"support_compatible={fixed_result.get('n5_support_compatible', False)}",
            f"logL={fixed_result.get('n5_log_likelihood_m23')}",
            f"KL={fixed_result.get('n5_kl_divergence_m23')}",
            f"G={fixed_result.get('n5_g_test_statistic_m23')}",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
