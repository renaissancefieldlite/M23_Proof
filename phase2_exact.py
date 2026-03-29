"""
phase2_exact.py
M23 Inverse Galois Attack - Phase 2 (EXACT ALGEBRAIC VERSION)
Uses sympy for exact arithmetic in Q(sqrt(-23)) and Q(g).
Author: Mirror Architect D / Codex 67
"""

import glob
import json
import os
from sympy import Rational, symbols, sqrt

JSON_DIR = "testjson"
CANDIDATE_FILE = os.path.join(JSON_DIR, "exact_candidates.json")
REFINED_GLOB = os.path.join(JSON_DIR, "exact_refined_*.json")

os.makedirs(JSON_DIR, exist_ok=True)

g = symbols("g")
quartic_modulus = g**4 + g**3 + 9 * g**2 - 10 * g + 8
sqrt_m23 = sqrt(-23)


def load_elkies_families(json_file: str = "elkies_families.json") -> list:
    """Load families from Phase 1 JSON export."""
    try:
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"Loaded {len(data)} families from {json_file}")
        return data
    except FileNotFoundError:
        print(f"Error: {json_file} not found. Run phase1.py first.")
        return []
    except json.JSONDecodeError:
        print(f"Error: {json_file} is corrupted.")
        return []


def dedupe_candidates(candidates: list) -> list:
    unique = []
    seen = set()
    for candidate in candidates:
        key = (
            candidate["λ_real"],
            candidate["λ_imag"],
            candidate["μ_real"],
            candidate["μ_imag"],
        )
        if key not in seen:
            seen.add(key)
            unique.append(candidate)
    return unique


def load_refined_candidates() -> list:
    """Load the most recent refined candidates from Phase 4."""
    files = sorted(glob.glob(REFINED_GLOB), key=os.path.getmtime)
    if not files:
        print("No refined candidates found. Using default seeds.")
        return []

    latest_file = files[-1]
    try:
        with open(latest_file, "r", encoding="utf-8") as f:
            candidates = json.load(f)
        candidates = dedupe_candidates(candidates)
        print(f"Loaded {len(candidates)} refined candidates from {latest_file}")
        return candidates
    except Exception as exc:
        print(f"Could not load {latest_file}: {exc}")
        return []


def candidate_record(lr, li, mr, mi):
    return {
        "λ_real": str(lr),
        "λ_imag": str(li),
        "μ_real": str(mr),
        "μ_imag": str(mi),
        "λ_expr": f"{lr} + {li}*I",
        "μ_expr": f"{mr} + {mi}*I",
    }


def generate_exact_grid():
    """Generate exact rational parameter grid around hot zone."""
    λ_real_vals = [Rational(x, 10) for x in range(-135, -119)]
    μ_real_vals = [Rational(x, 10) for x in range(-285, -269)]
    imag_vals = [
        -2,
        -Rational(3, 2),
        -1,
        -Rational(1, 2),
        0,
        Rational(1, 2),
        1,
        Rational(3, 2),
        2,
        Rational(5, 2),
    ]

    candidates = []
    total = len(λ_real_vals) * len(imag_vals) * len(μ_real_vals) * len(imag_vals)
    print(f"Generating {total} exact algebraic candidates...")

    count = 0
    for λr in λ_real_vals:
        for λi in imag_vals:
            for μr in μ_real_vals:
                for μi in imag_vals:
                    if abs(float(λr + 13)) > 1.0:
                        continue
                    if abs(float(μr + 28)) > 1.0:
                        continue
                    candidates.append(candidate_record(λr, λi, μr, μi))
                    count += 1

    print(f"Generated {count} filtered exact candidates")
    return dedupe_candidates(candidates)


def select_candidates():
    refined = load_refined_candidates()
    if refined:
        print("Using latest refined candidates from Phase 4")
        return refined
    return generate_exact_grid()


if __name__ == "__main__":
    print("=" * 70)
    print("M23 Inverse Galois Attack - Phase 2 (EXACT ALGEBRAIC)")
    print("Using sympy for exact arithmetic in Q(sqrt(-23)) and Q(g)")
    print(f"JSON directory: {JSON_DIR}")
    print("=" * 70)

    candidates = select_candidates()

    with open(CANDIDATE_FILE, "w", encoding="utf-8") as f:
        json.dump(candidates, f, indent=2)

    print(f"\nSaved {len(candidates)} exact candidates to {CANDIDATE_FILE}")

    load_elkies_families()

    print("\nNext steps:")
    print("1. Run phase3_exact.py to test these candidates with exact arithmetic")
    print("2. Use Sage/Magma for real Galois group verification")
