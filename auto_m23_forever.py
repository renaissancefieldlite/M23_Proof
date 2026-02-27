#!/usr/bin/env python3
"""
auto_m23_forever.py - Fully autonomous M23 search
Runs indefinitely until target consistency reached.
All JSON files stored in testjson/ folder.
"""

import subprocess
import time
import json
import glob
import os
import sys
from datetime import datetime

JSON_DIR = "testjson"
os.makedirs(JSON_DIR, exist_ok=True)

def log_message(msg):
    """Print with timestamp."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {msg}")
    sys.stdout.flush()

def get_latest_results():
    pattern = os.path.join(JSON_DIR, "exact_test_results_*.json")
    files = glob.glob(pattern)
    if not files:
        return None
    return max(files)

def get_best_consistency():
    latest = get_latest_results()
    if not latest:
        return 0.0
    
    try:
        with open(latest, 'r') as f:
            data = json.load(f)
        
        best = 0.0
        for r in data:
            if r['result']['success']:
                score = r['result']['consistency_score']
                if score > best:
                    best = score
        return best
    except:
        return 0.0

def check_for_success():
    """Check if we've hit the target."""
    best = get_best_consistency()
    log_message(f"Current best consistency: {best*100:.1f}% ({int(best*9)}/9 primes)")
    return best >= 0.30

def run_phase(phase_num, script_name):
    """Run a phase and return success."""
    log_message(f"▶️  Running Phase {phase_num}: {script_name}")
    result = subprocess.run(["python3", script_name],
                           capture_output=True, text=True)
    
    if result.returncode != 0:
        log_message(f"⚠️  Phase {phase_num} returned code {result.returncode}")
        if result.stderr:
            log_message(f"Error: {result.stderr[-200:]}")
        return False
    
    log_message(f"✅ Phase {phase_num} complete")
    return True

def main():
    target = 0.30
    iteration = 0
    max_iterations = 1000  # High safety cap
    
    log_message("=" * 60)
    log_message("M23 AUTO-PILOT — FULLY AUTONOMOUS MODE")
    log_message(f"JSON directory: {JSON_DIR}")
    log_message(f"Target: {target*100:.0f}% irreducibles (6/9 primes)")
    log_message("Will run indefinitely until target reached")
    log_message("=" * 60)
    
    while iteration < max_iterations:
        iteration += 1
        log_message(f"\n{'='*60}")
        log_message(f"🔄 ITERATION {iteration}")
        log_message(f"{'='*60}")
        
        # Phase 2 - Generate exact candidates
        if not run_phase(2, "phase2_exact.py"):
            log_message("❌ Phase 2 failed. Retrying in 10 seconds...")
            time.sleep(10)
            continue
        
        # Phase 3 - Test with Sage
        if not run_phase(3, "phase3_exact.py"):
            log_message("❌ Phase 3 failed. Retrying...")
            continue
        
        # Check if we hit the target
        if check_for_success():
            latest = get_latest_results()
            log_message("\n" + "="*60)
            log_message("🎯 TARGET REACHED! 🎯")
            log_message(f"Best consistency: {get_best_consistency()*100:.1f}%")
            log_message(f"Results saved in: {latest}")
            log_message("="*60)
            break
        
        # Phase 4 - Refine best candidates
        log_message("🔄 Running Phase 4 (refinement)...")
        subprocess.run(["python3", "phase4_exact.py"],
                      capture_output=True)
        
        # Short pause between iterations
        time.sleep(2)
    
    if iteration >= max_iterations:
        log_message(f"\n⚠️ Stopped after {max_iterations} iterations (safety cap)")
        log_message(f"Final best: {get_best_consistency()*100:.1f}%")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log_message("\n🛑 Stopped by user")
        sys.exit(0)
