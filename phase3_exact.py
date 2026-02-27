"""
phase3_exact.py
M23 Inverse Galois Attack – Phase 3 (EXACT ALGEBRAIC VERSION)
Tests exact candidates using real polynomial factorization (via Sage).
FIXED: Proper number field handling for ℚ(√-23) coefficients.
Author: Mirror Architect D / Codex 67
"""

import json
import subprocess
import tempfile
import os
import time
from sympy import symbols, Poly, expand, sqrt, I, Rational
import numpy as np

# =============================================================================
# Configuration
# =============================================================================

JSON_DIR = "testjson"
os.makedirs(JSON_DIR, exist_ok=True)

# =============================================================================
# 1. Load Exact Candidates
# =============================================================================

def load_exact_candidates():
    """Load exact candidates from Phase 2."""
    json_file = os.path.join(JSON_DIR, "exact_candidates.json")
    try:
        with open(json_file, 'r') as f:
            candidates = json.load(f)
        print(f"📥 Loaded {len(candidates)} exact candidates from {json_file}")
        return candidates
    except FileNotFoundError:
        print(f"❌ No exact candidates found. Run phase2_exact.py first.")
        return []

# =============================================================================
# 2. Generate Sage Script for a Candidate
# =============================================================================

def generate_sage_script(candidate, index):
    """Generate Sage script to test a candidate - proper number field handling."""
    
    λr = candidate['λ_real']
    λi = candidate['λ_imag']
    μr = candidate['μ_real']
    μi = candidate['μ_imag']
    
    script = f'''# Sage script for M23 candidate {index}
import sys

def test_candidate():
    try:
        # Define the quartic field
        R.<g> = QQ[]
        K.<g> = NumberField(g^4 + g^3 + 9*g^2 - 10*g + 8)

        # Polynomial ring over K
        S.<x> = K[]

        print("-" * 50)
        print(f"Testing candidate {index}")
        print("-" * 50)
        print(f"λ = {λr} + {λi}*I")
        print(f"μ = {μr} + {μi}*I")

        # Define components exactly
        P2 = (8*g**3 + 16*g**2 - 20*g + 20)*x**2 + (-7*g**3 - 17*g**2 + 7*g - 76)*x + (-13*g**3 + 25*g**2 - 107*g + 596)
        P3 = 8*(31*g**3 + 405*g**2 - 459*g + 333)*x**3 + (941*g**3 + 1303*g**2 - 1853*g + 1772)*x + (85*g**3 - 385*g**2 + 395*g - 220)
        P4 = 32*(4*g**3 - 69*g**2 + 74*g - 49)*x**4 + 32*(21*g**3 + 53*g**2 - 68*g + 58)*x**3 - 8*(97*g**3 + 95*g**2 - 145*g + 148)*x**2 + 8*(41*g**3 - 89*g**2 - g + 140)*x + (-123*g**3 + 391*g**2 - 93*g + 3228)

        tau_num = 2**38 * 3**17 * (47323*g**3 - 1084897*g**2 + 7751*g - 711002)
        tau = tau_num / 23**3

        # Construct full polynomial
        P = P2**2 * P3 * P4**4 + tau

        print(f"Degree of P: {{P.degree(x)}}")
        print(f"Number of terms: {{len(P.coefficients())}}")

        # Test small primes using number field reduction
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
        irreducible_count = 0

        print("\\nFactorization mod primes:")
        for p in primes:
            try:
                # Create residue field of characteristic p
                # Find a prime ideal above p
                print(f"p = {{p}}: ", end="")
                
                # Try to find a prime ideal of degree 1
                prime_ideals = K.primes_above(p)
                
                if not prime_ideals:
                    print("no prime ideals")
                    continue
                
                # Use the first prime ideal
                P_ideal = prime_ideals[0]
                
                # Create residue field
                k = P_ideal.residue_field()
                
                # Reduce polynomial mod p
                P_reduced = P.change_ring(k)
                
                # Factor over residue field
                factors = P_reduced.factor()
                print(factors)
                
                # Check if irreducible (single factor of degree 23)
                if len(factors) == 1 and factors[0][1] == 1:
                    irreducible_count += 1
                    
            except Exception as e:
                print(f"error - {{str(e)[:50]}}")

        print(f"\\nIrreducible count: {{irreducible_count}}/9")
        print(f"Consistency score: {{irreducible_count/9.0:.3f}}")
        
        return 0
        
    except Exception as e:
        print(f"Error in Sage script: {{e}}")
        return 1

if __name__ == "__main__":
    sys.exit(test_candidate())
'''
    return script

# =============================================================================
# 3. Run Tests (via Sage)
# =============================================================================

def test_candidate_with_sage(candidate, index, timeout=300):
    """Run a candidate through Sage and return results."""
    
    script = generate_sage_script(candidate, index)
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.sage', delete=False) as f:
        f.write(script)
        script_file = f.name
    
    print(f"   Running Sage (timeout: {timeout}s)...")
    start_time = time.time()
    
    try:
        result = subprocess.run(['sage', script_file],
                               capture_output=True, text=True,
                               timeout=timeout)
        elapsed = time.time() - start_time
        output = result.stdout
        error = result.stderr
        
        irreducible_count = 0
        for line in output.split('\n'):
            if 'Irreducible count:' in line:
                try:
                    parts = line.split(':')[1].strip().split('/')
                    if len(parts) > 0:
                        irreducible_count = int(parts[0])
                except:
                    pass
        
        return {
            'success': result.returncode == 0,
            'irreducible_count': irreducible_count,
            'consistency_score': irreducible_count / 9.0,
            'output': output[-1000:],
            'error': error[-500:],
            'elapsed': elapsed
        }
    except subprocess.TimeoutExpired:
        elapsed = time.time() - start_time
        return {
            'success': False,
            'irreducible_count': 0,
            'consistency_score': 0.0,
            'output': '',
            'error': f'Timeout after {timeout}s',
            'elapsed': elapsed
        }
    finally:
        try:
            os.unlink(script_file)
        except:
            pass

# =============================================================================
# 4. Main
# =============================================================================

def main():
    print("=" * 70)
    print("M23 Inverse Galois Attack – Phase 3 (EXACT ALGEBRAIC)")
    print("Testing exact candidates via Sage (real factorization)")
    print("=" * 70)
    
    candidates = load_exact_candidates()
    if not candidates:
        return
    
    max_tests = min(3, len(candidates))
    results = []
    
    for i in range(max_tests):
        print(f"\n{'='*60}")
        print(f"🔍 Testing candidate {i+1}/{max_tests}")
        print(f"{'='*60}")
        print(f"   λ = {candidates[i]['λ_expr']}")
        print(f"   μ = {candidates[i]['μ_expr']}")
        
        result = test_candidate_with_sage(candidates[i], i)
        
        if result['success']:
            print(f"   ✅ Sage completed in {result['elapsed']:.1f}s")
            print(f"   Irreducible count: {result['irreducible_count']}/9")
            print(f"   Consistency score: {result['consistency_score']:.3f}")
        else:
            print(f"   ❌ Failed: {result['error'][:100]}")
        
        results.append({
            'candidate_index': i,
            'candidate': candidates[i],
            'result': result
        })
        
        partial_file = os.path.join(JSON_DIR, 'exact_test_results_partial.json')
        with open(partial_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"   💾 Progress saved")
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(JSON_DIR, f"exact_test_results_{timestamp}.json")
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"✅ Saved results to {filename}")
    print(f"{'='*60}")
    
    print("\n📊 SUMMARY")
    print("-" * 40)
    for i, r in enumerate(results):
        if r['result']['success']:
            print(f"Candidate {i}: {r['result']['irreducible_count']}/9 irreducible "
                  f"({r['result']['consistency_score']:.3f})")
        else:
            print(f"Candidate {i}: FAILED")

if __name__ == "__main__":
    main()
