"""
phase4_exact.py
M23 Inverse Galois Attack – Phase 4 (EXACT ALGEBRAIC VERSION)
Refines best candidates from Phase 3 by generating tighter grids.
Author: Mirror Architect D / Codex 67
"""

import json
import glob
from sympy import Rational, sqrt, I
import os

# =============================================================================
# 1. Load Latest Phase 3 Results
# =============================================================================

def load_latest_results():
    """Load the most recent exact test results."""
    files = glob.glob("exact_test_results_*.json")
    if not files:
        print("❌ No test results found. Run phase3_exact.py first.")
        return None
    
    latest = max(files)
    try:
        with open(latest, 'r') as f:
            results = json.load(f)
        print(f"📥 Loaded results from {latest}")
        return results
    except Exception as e:
        print(f"❌ Error loading results: {e}")
        return None

# =============================================================================
# 2. Extract Best Candidates
# =============================================================================

def get_best_candidates(results, n=5):
    """Extract top N candidates by consistency score."""
    scored = []
    for r in results:
        if r['result']['success'] and r['result']['consistency_score'] > 0:
            scored.append({
                'candidate': r['candidate'],
                'score': r['result']['consistency_score'],
                'irreducible': r['result']['irreducible_count']
            })
    
    scored.sort(key=lambda x: x['score'], reverse=True)
    return scored[:n]

# =============================================================================
# 3. Generate Refined Grid Around Best Candidates
# =============================================================================

def refine_around_candidate(candidate, step=0.05):
    """Generate tighter grid around a promising candidate."""
    
    λr = Rational(candidate['λ_real'])
    λi = Rational(candidate['λ_imag'])
    μr = Rational(candidate['μ_real'])
    μi = Rational(candidate['μ_imag'])
    
    refined = []
    step_r = Rational(int(step * 100), 100)  # Convert to rational
    
    # Generate grid ±0.2 around best candidate with smaller step
    for dr in [-4, -3, -2, -1, 0, 1, 2, 3, 4]:
        for di in [-2, -1, 0, 1, 2]:
            new_λr = λr + dr * step_r
            new_λi = λi + di * step_r
            new_μr = μr + dr * step_r
            new_μi = μi + di * step_r
            
            refined.append({
                'λ_real': str(new_λr),
                'λ_imag': str(new_λi),
                'μ_real': str(new_μr),
                'μ_imag': str(new_μi),
                'λ_expr': f"{new_λr} + {new_λi}*I",
                'μ_expr': f"{new_μr} + {new_μi}*I"
            })
    
    return refined

# =============================================================================
# 4. Save Refined Candidates for Next Iteration
# =============================================================================

def save_refined_candidates(refined_list):
    """Save refined candidates to be used in next Phase 2 iteration."""
    
    # Flatten list if we got multiple from each candidate
    all_refined = []
    for candidates in refined_list:
        all_refined.extend(candidates)
    
    # Remove duplicates
    unique = []
    seen = set()
    for c in all_refined:
        key = (c['λ_real'], c['λ_imag'], c['μ_real'], c['μ_imag'])
        if key not in seen:
            seen.add(key)
            unique.append(c)
    
    print(f"   Generated {len(unique)} unique refined candidates")
    
    # Save with timestamp
    import time
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"exact_refined_{timestamp}.json"
    
    with open(filename, 'w') as f:
        json.dump(unique, f, indent=2)
    
    print(f"   💾 Saved to {filename}")
    return filename

# =============================================================================
# 5. Main
# =============================================================================

def main():
    print("=" * 70)
    print("M23 Inverse Galois Attack – Phase 4 (EXACT ALGEBRAIC)")
    print("Refining best candidates with tighter grids")
    print("=" * 70)
    
    # Load latest results
    results = load_latest_results()
    if not results:
        return
    
    # Get best candidates
    best = get_best_candidates(results, n=3)
    
    if not best:
        print("❌ No successful candidates found to refine.")
        return
    
    print(f"\n🎯 Best candidates found:")
    for i, b in enumerate(best):
        print(f"\n   {i+1}. Score: {b['score']:.3f} ({b['irreducible']}/9 irreducible)")
        print(f"      λ = {b['candidate']['λ_expr']}")
        print(f"      μ = {b['candidate']['μ_expr']}")
    
    # Refine around each best candidate
    print(f"\n🔬 Generating refined grids...")
    refined_list = []
    for b in best:
        refined = refine_around_candidate(b['candidate'])
        refined_list.append(refined)
        print(f"   Generated {len(refined)} candidates around λ={b['candidate']['λ_expr']}")
    
    # Save all refined candidates
    filename = save_refined_candidates(refined_list)
    
    print(f"\n✅ Phase 4 complete.")
    print(f"   Next: Run phase2_exact.py again to test refined candidates")
    print(f"   The new candidates are in {filename}")

if __name__ == "__main__":
    main()