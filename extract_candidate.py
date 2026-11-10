#!/usr/bin/env python3
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""
Quick script to extract Candidate 1394 from the massive JSON file.
Run with: python3 extract_candidate.py
"""

import json
import sys

def extract_candidate(filepath, candidate_id=1394):
    """Extract a specific candidate by ID from large JSON file."""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # The file structure might be different - let's explore
        print(f"📂 File loaded: {filepath}")
        print(f"📊 Data type: {type(data).__name__}")
        
        if isinstance(data, dict):
            print(f"🔑 Top-level keys: {list(data.keys())}")
            
            # Try common structures
            if 'results' in data:
                candidates = data['results']
            elif 'candidates' in data:
                candidates = data['candidates']
            else:
                candidates = data
        else:
            candidates = data
        
        # If it's a list, search through it
        if isinstance(candidates, list):
            print(f"📋 Found list with {len(candidates)} items")
            
            for i, item in enumerate(candidates[:5]):  # Show first 5 as sample
                print(f"\n  Item {i}: {json.dumps(item, indent=2)[:200]}...")
            
            # Search for candidate_id
            found = None
            for item in candidates:
                if isinstance(item, dict):
                    # Check various possible ID fields
                    if item.get('candidate_id') == candidate_id or \
                       item.get('id') == candidate_id or \
                       item.get('Candidate') == candidate_id:
                        found = item
                        break
            
            if found:
                print(f"\n✅ Found Candidate {candidate_id}:")
                print(json.dumps(found, indent=2))
                return found
            else:
                print(f"\n❌ Candidate {candidate_id} not found in list")
        
        elif isinstance(candidates, dict):
            # Maybe it's a dict with candidate IDs as keys
            if str(candidate_id) in candidates:
                print(f"\n✅ Found Candidate {candidate_id}:")
                print(json.dumps(candidates[str(candidate_id)], indent=2))
                return candidates[str(candidate_id)]
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

if __name__ == "__main__":
    # Find the most recent Phase 3 results file
    import glob
    files = glob.glob("m23_phase3_results_*.json")
    if not files:
        print("❌ No Phase 3 results files found")
        sys.exit(1)
    
    latest = max(files)
    print(f"📁 Using latest file: {latest}")
    
    candidate = extract_candidate(latest, 1394)
    
    if candidate:
        # Save to separate file
        outfile = "candidate_1394.json"
        with open(outfile, 'w') as f:
            json.dump(candidate, f, indent=2)
        print(f"\n💾 Saved to {outfile}")