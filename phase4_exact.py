"""
phase4_exact.py
M23 Inverse Galois Attack - Phase 4 (EXACT ALGEBRAIC VERSION)
Refines best candidates from Phase 3 by generating tighter grids.
Author: Mirror Architect D / Codex 67
"""

import glob
import json
import os
import time
from sympy import Rational

JSON_DIR = "testjson"
RESULT_GLOB = os.path.join(JSON_DIR, "exact_test_results_*.json")
REFINED_PREFIX = os.path.join(JSON_DIR, "exact_refined_")
PARTIAL_FILE = os.path.join(JSON_DIR, "partial.json")

os.makedirs(JSON_DIR, exist_ok=True)


def load_latest_results():
    """Load the most recent exact test results."""
    files = sorted(glob.glob(RESULT_GLOB), key=os.path.getmtime)

    if not files and os.path.exists(PARTIAL_FILE):
        files = [PARTIAL_FILE]

    if not files:
        print("No test results found. Run phase3_exact.py first.")
        return None

    latest = files[-1]
    try:
        with open(latest, "r", encoding="utf-8") as f:
            results = json.load(f)
        print(f"Loaded results from {latest}")
        return results
    except Exception as exc:
        print(f"Error loading results: {exc}")
        return None


def get_best_candidates(results, n=5):
    """Extract top N candidates by consistency score, then tested_count."""
    scored = []
    for result_entry in results:
        result = result_entry.get("result", {})
        tested = int(result.get("tested_count", 0))
        score = float(result.get("consistency_score", 0.0))
        success = bool(result.get("success", False))

        if tested <= 0:
            continue

        scored.append(
            {
                "candidate": result_entry["candidate"],
                "score": score,
                "irreducible": int(result.get("irreducible_count", 0)),
                "tested_count": tested,
                "success": success,
            }
        )

    scored.sort(
        key=lambda x: (
            x["score"],
            x["irreducible"],
            x["tested_count"],
            1 if x["success"] else 0,
        ),
        reverse=True,
    )
    return scored[:n]


def refine_around_candidate(candidate, real_step="1/20", imag_step="1/20"):
    """Generate tighter grid around a promising candidate."""
    λr = Rational(candidate["λ_real"])
    λi = Rational(candidate["λ_imag"])
    μr = Rational(candidate["μ_real"])
    μi = Rational(candidate["μ_imag"])

    real_step = Rational(real_step)
    imag_step = Rational(imag_step)

    refined = []
    for dλr in range(-4, 5):
        for dλi in range(-2, 3):
            for dμr in range(-4, 5):
                for dμi in range(-2, 3):
                    new_λr = λr + dλr * real_step
                    new_λi = λi + dλi * imag_step
                    new_μr = μr + dμr * real_step
                    new_μi = μi + dμi * imag_step

                    refined.append(
                        {
                            "λ_real": str(new_λr),
                            "λ_imag": str(new_λi),
                            "μ_real": str(new_μr),
                            "μ_imag": str(new_μi),
                            "λ_expr": f"{new_λr} + {new_λi}*I",
                            "μ_expr": f"{new_μr} + {new_μi}*I",
                        }
                    )

    return refined


def save_refined_candidates(refined_list):
    """Save refined candidates to be used in next Phase 2 iteration."""
    all_refined = []
    for candidates in refined_list:
        all_refined.extend(candidates)

    unique = []
    seen = set()
    for candidate in all_refined:
        key = (
            candidate["λ_real"],
            candidate["λ_imag"],
            candidate["μ_real"],
            candidate["μ_imag"],
        )
        if key not in seen:
            seen.add(key)
            unique.append(candidate)

    print(f"Generated {len(unique)} unique refined candidates")

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"{REFINED_PREFIX}{timestamp}.json"

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(unique, f, indent=2)

    print(f"Saved to {filename}")
    return filename


def main():
    print("=" * 70)
    print("M23 Inverse Galois Attack - Phase 4 (EXACT ALGEBRAIC)")
    print("Refining best candidates with tighter grids")
    print("=" * 70)

    results = load_latest_results()
    if not results:
        return

    best = get_best_candidates(results, n=3)

    if not best:
        print("No tested candidates found to refine.")
        return

    print("\nBest candidates found:")
    for i, entry in enumerate(best):
        print(
            f"\n{i + 1}. Score: {entry['score']:.3f} "
            f"({entry['irreducible']}/{entry['tested_count']} irreducible, success={entry['success']})"
        )
        print(f"   lambda = {entry['candidate']['λ_expr']}")
        print(f"   mu = {entry['candidate']['μ_expr']}")

    print("\nGenerating refined grids...")
    refined_list = []
    for entry in best:
        refined = refine_around_candidate(entry["candidate"])
        refined_list.append(refined)
        print(f"Generated {len(refined)} candidates around lambda={entry['candidate']['λ_expr']}")

    filename = save_refined_candidates(refined_list)

    print("\nPhase 4 complete.")
    print("Next: Run phase2_exact.py again to test refined candidates")
    print(f"The new candidates are in {filename}")


if __name__ == "__main__":
    main()
