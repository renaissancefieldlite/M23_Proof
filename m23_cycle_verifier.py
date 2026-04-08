#!/usr/bin/env python3
"""
m23_cycle_verifier.py - Annotate cycle signatures in M23 result files.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from m23_cycle_signatures import annotate_prime_entry, summarize_cycle_entries, summarize_n5_entries

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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_json", nargs="?", type=Path, default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="optional output path; defaults to testjson/m23_cycle_report_TIMESTAMP.json",
    )
    args = parser.parse_args()

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

    output_path = args.output or JSON_DIR / (
        f"m23_cycle_report_{time.strftime('%Y%m%d_%H%M%S')}.json"
    )
    wrapped = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "input_file": str(input_path),
        "report": report,
    }
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(wrapped, handle, indent=2)

    print(f"Saved M23 cycle report to {output_path}")
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
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
