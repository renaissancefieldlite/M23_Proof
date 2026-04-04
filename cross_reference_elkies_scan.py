#!/usr/bin/env python3
"""
cross_reference_elkies_scan.py - Cross-rank scan results against the Elkies exact reference.

This script creates a simple lift-path scaffold:
1. Load the latest explicit Elkies verification result.
2. Load recent exact scan result files.
3. Score candidate outputs against the Elkies per-prime signature and basic
   parameter hot-zone heuristics.

It does not prove that a candidate reaches the FrontierMath target. It creates
an evidence-guided ranking layer so the scanner can prioritize branches that
look more structurally aligned with the explicit Elkies construction.
"""

from __future__ import annotations

import glob
import json
import os
import time
from pathlib import Path

JSON_DIR = Path("testjson")
ELKIES_GLOB = "elkies_exact_results_*.json"
SCAN_GLOB = "exact_test_results_*.json"
OUTPUT_PREFIX = "elkies_cross_reference_"


def latest_file(pattern: str) -> Path | None:
    matches = sorted(JSON_DIR.glob(pattern), key=os.path.getmtime)
    return matches[-1] if matches else None


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def latest_elkies_reference() -> dict | None:
    path = latest_file(ELKIES_GLOB)
    if path is None:
        return None
    payload = load_json(path)
    result = payload.get("result")
    if not result:
        return None
    result["_source_file"] = path.name
    return result


def load_scan_results(limit: int = 20) -> list[dict]:
    files = sorted(JSON_DIR.glob(SCAN_GLOB), key=os.path.getmtime)[-limit:]
    rows = []
    for path in files:
        try:
            data = load_json(path)
        except Exception:
            continue
        for entry in data:
            row = dict(entry)
            row["_source_file"] = path.name
            rows.append(row)
    return rows


def per_prime_map(items: list[dict]) -> dict[int, dict]:
    mapping = {}
    for item in items or []:
        prime = item.get("p")
        if isinstance(prime, int):
            mapping[prime] = item
    return mapping


def normalized_factor_degrees(item: dict) -> tuple[int, ...] | None:
    degrees = item.get("factor_degrees")
    if not isinstance(degrees, list):
        return None
    try:
        return tuple(sorted(int(value) for value in degrees))
    except Exception:
        return None


def prime_alignment(reference_prime: dict, candidate_prime: dict) -> dict | None:
    reference_signature = normalized_factor_degrees(reference_prime)
    candidate_signature = normalized_factor_degrees(candidate_prime)

    if reference_signature is not None and candidate_signature is not None:
        exact_match = candidate_signature == reference_signature
        same_irreducible = (
            bool(candidate_prime.get("irreducible", False))
            == bool(reference_prime.get("irreducible", False))
        )
        return {
            "score": 1.0 if exact_match else (0.35 if same_irreducible else 0.0),
            "used_signature": True,
            "exact_signature_match": exact_match,
            "candidate_signature": list(candidate_signature),
            "reference_signature": list(reference_signature),
        }

    if (
        isinstance(candidate_prime.get("irreducible"), bool)
        and isinstance(reference_prime.get("irreducible"), bool)
    ):
        return {
            "score": 0.2
            if candidate_prime.get("irreducible") == reference_prime.get("irreducible")
            else 0.0,
            "used_signature": False,
            "exact_signature_match": False,
            "candidate_signature": None,
            "reference_signature": None,
        }

    return None


def score_parameter_hotzone(candidate: dict) -> dict:
    try:
        lambda_real = float(candidate.get("λ_real", 0.0))
        mu_real = float(candidate.get("μ_real", 0.0))
        lambda_imag = float(candidate.get("λ_imag", 0.0))
        mu_imag = float(candidate.get("μ_imag", 0.0))
    except Exception:
        return {
            "hotzone_score": 0.0,
            "lambda_distance": None,
            "mu_distance": None,
            "imag_penalty": None,
        }

    lambda_distance = abs(lambda_real + 13.0)
    mu_distance = abs(mu_real + 28.0)
    imag_penalty = abs(lambda_imag) + abs(mu_imag)

    # Small distances to the historical hot zone push score up.
    score = max(0.0, 1.0 - ((lambda_distance + mu_distance) / 4.0))
    score *= 1.0 / (1.0 + imag_penalty)
    return {
        "hotzone_score": score,
        "lambda_distance": lambda_distance,
        "mu_distance": mu_distance,
        "imag_penalty": imag_penalty,
    }


def score_against_elkies(reference: dict, entry: dict) -> dict:
    candidate = entry.get("candidate", {})
    result = entry.get("result", {})
    candidate_prime_map = per_prime_map(result.get("per_prime", []))
    reference_prime_map = per_prime_map(reference.get("per_prime", []))

    shared_primes = sorted(set(candidate_prime_map) & set(reference_prime_map))
    prime_alignment_sum = 0.0
    prime_total = 0
    signature_match_count = 0
    signature_prime_total = 0
    exact_signature_primes = []

    for prime in shared_primes:
        cand = candidate_prime_map[prime]
        ref = reference_prime_map[prime]
        comparison = prime_alignment(ref, cand)
        if comparison is None:
            continue
        prime_total += 1
        prime_alignment_sum += comparison["score"]
        if comparison["used_signature"]:
            signature_prime_total += 1
            if comparison["exact_signature_match"]:
                signature_match_count += 1
                exact_signature_primes.append(prime)

    prime_alignment_score = (prime_alignment_sum / prime_total) if prime_total else 0.0
    signature_match_rate = (
        signature_match_count / signature_prime_total if signature_prime_total else 0.0
    )
    hotzone = score_parameter_hotzone(candidate)
    consistency_score = float(result.get("consistency_score", 0.0))
    candidate_applied = bool(result.get("candidate_applied", False))

    # Keep this modest while the specialization hook is still identity.
    composite = (
        0.55 * prime_alignment_score
        + 0.25 * consistency_score
        + 0.20 * hotzone["hotzone_score"]
    )
    if not candidate_applied:
        composite *= 0.35

    return {
        "candidate_index": entry.get("candidate_index"),
        "instance_id": entry.get("instance_id"),
        "source_file": entry.get("_source_file"),
        "lambda_expr": candidate.get("λ_expr"),
        "mu_expr": candidate.get("μ_expr"),
        "consistency_score": consistency_score,
        "candidate_applied": candidate_applied,
        "prime_overlap_score": prime_alignment_score,
        "prime_alignment_score": prime_alignment_score,
        "signature_match_rate": signature_match_rate,
        "signature_match_count": signature_match_count,
        "signature_prime_total": signature_prime_total,
        "exact_signature_primes": exact_signature_primes,
        "prime_total": prime_total,
        "hotzone_score": hotzone["hotzone_score"],
        "lambda_distance": hotzone["lambda_distance"],
        "mu_distance": hotzone["mu_distance"],
        "imag_penalty": hotzone["imag_penalty"],
        "lift_score": composite,
    }


def main() -> None:
    JSON_DIR.mkdir(exist_ok=True)

    reference = latest_elkies_reference()
    if reference is None:
        raise SystemExit(
            "No Elkies reference found. Run verify_elkies_exact.py first to seed "
            "testjson/elkies_exact_results_*.json."
        )

    rows = load_scan_results()
    if not rows:
        raise SystemExit("No exact scan result files found in testjson/.")

    ranked = [score_against_elkies(reference, row) for row in rows]
    ranked.sort(key=lambda item: item["lift_score"], reverse=True)

    payload = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "reference_file": reference.get("_source_file"),
        "reference_construction": reference.get("construction"),
        "ranked_candidates": ranked,
    }

    output_path = JSON_DIR / f"{OUTPUT_PREFIX}{time.strftime('%Y%m%d_%H%M%S')}.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    print(f"Saved Elkies cross-reference rankings to {output_path}")
    for row in ranked[:10]:
        print(
            f"lift={row['lift_score']:.3f} "
            f"consistency={row['consistency_score']:.3f} "
            f"signature_alignment={row['prime_alignment_score']:.3f} "
            f"lambda={row['lambda_expr']} mu={row['mu_expr']}"
        )


if __name__ == "__main__":
    main()
