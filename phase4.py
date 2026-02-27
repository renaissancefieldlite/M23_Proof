"""
phase4_feedback_loop.py
M23 Inverse Galois Attack – Phase 4
Uses Phase 3 results to refine candidate generation.
Implements Codex 5.3 recursive envelope compression.
Author: Mirror Architect D / Codex 67
"""

import numpy as np
import json
from typing import List, Dict, Any, Tuple
from collections import Counter
import datetime

# =============================================================================
# 1. Load Phase 3 Results
# =============================================================================

def load_phase3_results(json_file: str = None) -> Dict:
    """Load Phase 3 results."""
    if json_file is None:
        import glob
        files = glob.glob("m23_phase3_results_*.json")
        if not files:
            print("❌ No Phase 3 results found.")
            return None
        json_file = max(files)
        print(f"📁 Auto-detected: {json_file}")
    
    with open(json_file, 'r') as f:
        data = json.load(f)
    print(f"📥 Loaded Phase 3 results from {json_file}")
    return data


# =============================================================================
# 2. Codex 5.3 Envelope Compression
# =============================================================================

class Codex53EnvelopeCompressor:
    """
    Implements recursive envelope compression for parameter space refinement.
    Based on the Codex 5.3 attractor concept.
    """
    
    def __init__(self, compression_factor: float = 0.68):
        self.compression_factor = compression_factor
        self.iteration = 0
        self.envelope_history = []
        
    def compress_parameter_space(self, candidates: List[Dict], 
                                scores: List[float]) -> List[Dict]:
        """
        Apply envelope compression to candidate parameters.
        Higher-scoring candidates get expanded, lower-scoring get compressed.
        """
        if not candidates:
            return []
            
        self.iteration += 1
        
        # Normalize scores to [0, 1]
        scores = np.array(scores)
        scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-10)
        
        # Generate new candidates around high-scoring ones
        new_candidates = []
        
        for i, cand in enumerate(candidates):
            if scores[i] > 0.7:  # High-scoring candidates
                # Expand around these
                variations = self._expand_candidate(cand, scores[i])
                new_candidates.extend(variations)
            elif scores[i] > 0.3:  # Medium-scoring
                # Keep but don't expand
                new_candidates.append(cand)
            # Low-scoring are dropped (compressed away)
        
        # Remove duplicates
        unique = self._deduplicate(new_candidates)
        
        print(f"   Iteration {self.iteration}: {len(candidates)} → {len(unique)} candidates")
        
        return unique
    
    def _expand_candidate(self, candidate: Dict, score: float) -> List[Dict]:
        """Generate variations around a high-scoring candidate."""
        variations = []
        params = candidate["params"]
        
        # Expansion factor based on score
        expansion = 0.5 * (1 + score)
        
        for dr in [-2, -1, 0, 1, 2]:
            for di in [-1, 0, 1]:
                if dr == 0 and di == 0:
                    continue
                    
                new_params = []
                for p in params:
                    new_p = [p[0] + dr * expansion, p[1] + di * expansion]
                    new_params.append(new_p)
                
                new_cand = candidate.copy()
                new_cand["params"] = new_params
                new_cand["rationality_score"] = score * 0.9  # Slight decay
                variations.append(new_cand)
        
        return variations
    
    def _deduplicate(self, candidates: List[Dict]) -> List[Dict]:
        """Remove duplicate candidates."""
        unique = []
        seen = set()
        
        for cand in candidates:
            # Create key from rounded params
            params_key = tuple([(round(p[0], 3), round(p[1], 3)) 
                               for p in cand["params"]])
            if params_key not in seen:
                seen.add(params_key)
                unique.append(cand)
        
        return unique


# =============================================================================
# 3. Feedback Analysis
# =============================================================================

class FeedbackAnalyzer:
    """
    Analyzes Phase 3 results to guide next iteration.
    """
    
    def __init__(self, phase3_data: Dict):
        self.data = phase3_data
        self.results = phase3_data.get("results", [])
        
    def analyze_failure_patterns(self) -> Dict:
        """
        Identify why candidates failed prime tests.
        """
        patterns = Counter()
        prime_coverage = Counter()
        
        for r in self.results:
            # Count prime tests passed
            passed_primes = len(r.get("prime_scores", {}))
            prime_coverage[passed_primes] += 1
            
            # Note consistency scores
            if r["consistency_score"] < 0.3:
                patterns["very_low_consistency"] += 1
            elif r["consistency_score"] < 0.6:
                patterns["moderate_consistency"] += 1
            
        return {
            "prime_coverage": dict(prime_coverage),
            "patterns": dict(patterns),
            "total": len(self.results)
        }
    
    def get_best_candidates(self, n: int = 20) -> List[Dict]:
        """Return best candidates despite failing."""
        sorted_res = sorted(self.results, 
                           key=lambda x: x["consistency_score"], 
                           reverse=True)
        return sorted_res[:n]


# =============================================================================
# 4. Main Execution
# =============================================================================

def main():
    print("=" * 70)
    print("M23 Inverse Galois Attack – Phase 4: Feedback Loop")
    print("Codex 5.3 Envelope Compression + Recursive Refinement")
    print("=" * 70)
    
    # Load Phase 3 results
    phase3 = load_phase3_results()
    if not phase3:
        return
    
    # Analyze failures
    analyzer = FeedbackAnalyzer(phase3)
    patterns = analyzer.analyze_failure_patterns()
    
    print("\n📊 Failure Analysis:")
    print(f"   Total candidates tested: {patterns['total']}")
    print(f"   Prime coverage: {patterns['prime_coverage']}")
    print(f"   Consistency patterns: {patterns['patterns']}")
    
    # Get best candidates for refinement
    best = analyzer.get_best_candidates(20)
    print(f"\n🎯 Best {len(best)} candidates (for refinement):")
    
    best_list = []
    for i, r in enumerate(best[:5]):  # Show top 5
        print(f"\n   {i+1}. Candidate {r['candidate_id']}")
        print(f"      Family: {r['family']}")
        print(f"      Consistency: {r['consistency_score']:.3f}")
        print(f"      Primes: {list(r.get('prime_scores', {}).keys())}")
        
        # Reconstruct candidate for compression
        cand = {
            "family": r["family"],
            "params": r["params"],
            "rationality_score": r["consistency_score"]
        }
        best_list.append(cand)
    
    # Apply envelope compression
    print("\n🔄 Applying Codex 5.3 envelope compression...")
    compressor = Codex53EnvelopeCompressor(compression_factor=0.68)
    
    # First iteration
    refined = compressor.compress_parameter_space(
        best_list, 
        [r["consistency_score"] for r in best]
    )
    
    # Save refined candidates for next Phase 2 iteration
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"m23_refined_candidates_{timestamp}.json"
    
    with open(filename, "w") as f:
        json.dump(refined, f, indent=2)
    
    print(f"\n💾 Saved {len(refined)} refined candidates to {filename}")
    print(f"\n🔄 Ready for Phase 2 iteration 2 with refined candidates.")
    print(f"\n✅ Phase 4 complete. Loop back to Phase 2.")


if __name__ == "__main__":
    main()