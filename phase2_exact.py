"""
phase2_exact.py
M23 Inverse Galois Attack – Phase 2 (EXACT ALGEBRAIC VERSION)
Uses sympy for exact arithmetic in ℚ(√-23) and ℚ(g).
FIXED: All JSON saved to testjson/ folder
Author: Mirror Architect D / Codex 67
"""

import json
import glob
import os
from sympy import symbols, Poly, expand, sqrt, I, Rational, Number
from sympy import I as SyI
import numpy as np

# =============================================================================
# Configuration
# =============================================================================

JSON_DIR = "testjson"
os.makedirs(JSON_DIR, exist_ok=True)

# =============================================================================
# 1. Load Phase 1 Data and Refined Candidates
# =============================================================================

def load_elkies_families(json_file: str = "elkies_families.json") -> list:
    """Load families from Phase 1 JSON export."""
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
        print(f"📥 Loaded {len(data)} families from {json_file}")
        return data
    except FileNotFoundError:
        print(f"❌ Error: {json_file} not found. Run phase1.py first.")
        return []
    except json.JSONDecodeError:
        print(f"❌ Error: {json_file} is corrupted.")
        return []

def load_refined_candidates() -> list:
    """Load the most recent refined candidates from Phase 4."""
    pattern = os.path.join(JSON_DIR, "exact_refined_*.json")
    files = glob.glob(pattern)
    if not files:
        print("   No refined candidates found. Using default seeds.")
        return []
    
    latest_file = max(files)
    try:
        with open(latest_file, 'r') as f:
            candidates = json.load(f)
        print(f"   Loaded {len(candidates)} refined candidates from {latest_file}")
        return candidates
    except:
        print(f"   Could not load {latest_file}")
        return []

# =============================================================================
# 2. Define the Number Fields
# =============================================================================

# Define the quartic field ℚ(g) from Elkies' paper
g = symbols('g')
quartic_modulus = g**4 + g**3 + 9*g**2 - 10*g + 8

# √-23 appears in the field
sqrt_m23 = sqrt(-23)

# =============================================================================
# 3. Parameter Space Grid (Exact Rational)
# =============================================================================

def generate_exact_grid():
    """Generate exact rational parameter grid around hot zone."""
    
    # λ real: around -13 to -12 (step 0.1)
    λ_real_vals = [Rational(x, 10) for x in range(-135, -119, 1)]
    # μ real: around -28 to -27 (step 0.1)
    μ_real_vals = [Rational(x, 10) for x in range(-285, -269, 1)]
    # Imaginary parts: multiples of 0.5
    imag_vals = [
        -2, -Rational(3,2), -1, -Rational(1,2), 0,
        Rational(1,2), 1, Rational(3,2), 2, Rational(5,2)
    ]
    
    candidates = []
    total = len(λ_real_vals) * len(imag_vals) * len(μ_real_vals) * len(imag_vals)
    print(f"   Generating {total} exact algebraic candidates...")
    
    count = 0
    for λr in λ_real_vals:
        for λi in imag_vals:
            for μr in μ_real_vals:
                for μi in imag_vals:
                    # Filter to region near best candidate
                    if abs(float(λr + 13)) > 1.0:
                        continue
                    if abs(float(μr + 28)) > 1.0:
                        continue
                    
                    candidates.append({
                        'λ_real': str(λr),
                        'λ_imag': str(λi),
                        'μ_real': str(μr),
                        'μ_imag': str(μi),
                        'λ_expr': f"{λr} + {λi}*I",
                        'μ_expr': f"{μr} + {μi}*I"
                    })
                    count += 1
    
    print(f"   Generated {count} filtered exact candidates")
    return candidates

# =============================================================================
# 4. Main
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("M23 Inverse Galois Attack – Phase 2 (EXACT ALGEBRAIC)")
    print("Using sympy for exact arithmetic in ℚ(√-23) and ℚ(g)")
    print(f"JSON directory: {JSON_DIR}")
    print("=" * 70)
    
    # Generate exact parameter grid
    candidates = generate_exact_grid()
    
    # Save to file in JSON_DIR
    filename = os.path.join(JSON_DIR, 'exact_candidates.json')
    with open(filename, 'w') as f:
        json.dump(candidates, f, indent=2)
    
    print(f"\n✅ Saved {len(candidates)} exact candidates to {filename}")
    
    # Load families (for reference)
    families = load_elkies_families()
    
    print("\nNext steps:")
    print("1. Run phase3_exact.py to test these candidates with exact arithmetic")
    print("2. Use Sage/Magma for real Galois group verification")
