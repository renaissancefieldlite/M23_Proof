#!/usr/bin/env python3
"""
lift_survivors.py - Promote screened M23 candidates into a clean survivor set.

This is the cleanup stage of the current canonical pipeline:
  1. generator    -> bounded Tschirnhaus transforms
  2. filter       -> leakage + modular signatures
  3. lift/cleanup -> keep the rationalized survivors worth deeper work

The current implementation does not yet do full p-adic Newton lifting or LLL.
It promotes the best already-rationalized survivors into a compact artifact so
the next implementation pass has a clean handoff point.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

JSON_DIR = Path("testjson")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("screen_json", type=Path)
    parser.add_argument("--min-signature-hits", type=int, default=1)
    parser.add_argument("--max-height", type=int, default=10**8)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    JSON_DIR.mkdir(exist_ok=True)
    with args.screen_json.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    survivors = []
    for result in payload.get("results", []):
        if not result.get("rationalized"):
            continue
        if result.get("integer_height", 10**18) > args.max_height:
            continue
        if result.get("signature_hit_count", 0) < args.min_signature_hits:
            continue
        survivors.append(
            {
                "candidate": result["candidate"],
                "descent_score": result["descent_score"],
                "rationalization_mode": result.get("rationalization_mode"),
                "rationalization_scalar_expr": result.get("rationalization_scalar_expr"),
                "rationalization_scalar_basis_vector": result.get("rationalization_scalar_basis_vector"),
                "rationalization_diagnostics": result.get("rationalization_diagnostics", {}),
                "integer_height": result.get("integer_height"),
                "denominator_lcm": result.get("denominator_lcm"),
                "signature_hit_count": result.get("signature_hit_count", 0),
                "irreducible_prime_count": result.get("irreducible_prime_count", 0),
                "screened_primes": result.get("screened_primes", []),
                "integer_coefficients": result.get("integer_coefficients", []),
            }
        )

    survivors.sort(
        key=lambda item: (
            -item["signature_hit_count"],
            -item["irreducible_prime_count"],
            item["integer_height"],
            item["descent_score"],
        )
    )

    output_path = args.output or JSON_DIR / f"lift_survivors_{time.strftime('%Y%m%d_%H%M%S')}.json"
    result_payload = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "screen_json": str(args.screen_json),
        "min_signature_hits": args.min_signature_hits,
        "max_height": args.max_height,
        "survivor_count": len(survivors),
        "survivors": survivors,
    }

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(result_payload, handle, indent=2)

    print(f"Saved {len(survivors)} survivor(s) to {output_path}")
    if survivors:
        top = survivors[0]
        print(
            "Top survivor:",
            f"signature_hits={top['signature_hit_count']}",
            f"irreducible_primes={top['irreducible_prime_count']}",
            f"height={top['integer_height']}",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
