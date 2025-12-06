#!/usr/bin/env python3
"""
Test script to run investigation detectors.

Usage:
    python tools/test_detectors.py data/fdb_logs.duckdb
"""

import sys
import json
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from tools.investigation_tools import Detectors


def main():
    if len(sys.argv) < 2:
        print("Usage: python tools/test_detectors.py <db_path>")
        sys.exit(1)
    
    db_path = sys.argv[1]
    
    if not Path(db_path).exists():
        print(f"Error: Database file not found: {db_path}")
        sys.exit(1)
    
    print(f"\n🔍 Running detectors on: {db_path}\n")
    print("=" * 70)

    det = Detectors(db_path)
    
    # Run all detectors
    detectors = [
        ("Storage Engine Pressure", det.storage_engine_pressure),
        ("Ratekeeper Throttling", det.ratekeeper_throttling),
        ("Missing TLogs", det.missing_tlogs),
        ("Recovery Loop", det.recovery_loop),
        ("Coordination Loss", det.coordination_loss),
        ("Z-Score Hotspots", det.zscore_hotspots),
    ]
    
    results = {}
    
    for name, detector_func in detectors:
        print(f"\n📊 {name}...")
        try:
            result = detector_func(db_path)
            results[name] = result
            
            if result.get("detected", False):
                print(f"   ⚠️  DETECTED: {name}")
                if "count" in result:
                    print(f"   Count: {result['count']}")
                if "max_lag" in result:
                    print(f"   Max Lag: {result.get('max_lag')}")
                if "throttling_events" in result:
                    print(f"   Throttling Events: {len(result.get('throttling_events', []))}")
                if "loop_count" in result:
                    print(f"   Loops Detected: {result.get('loop_count')}")
            else:
                print(f"   ✓ No issues detected")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
            results[name] = {"error": str(e)}
    
    print("\n" + "=" * 70)
    print("\n📋 Summary:")
    print(f"   Detectors Run: {len(detectors)}")
    detected_count = sum(1 for r in results.values() if r.get("detected", False))
    print(f"   Issues Detected: {detected_count}")
    
    # Print detailed results if requested
    if len(sys.argv) > 2 and sys.argv[2] == "--verbose":
        print("\n📄 Detailed Results:")
        print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
