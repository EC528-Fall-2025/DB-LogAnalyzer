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

# Import directly to avoid __init__.py dependencies
import importlib.util
investigation_tools_path = project_root / "tools" / "investigation_tools.py"
spec = importlib.util.spec_from_file_location("investigation_tools", investigation_tools_path)
investigation_tools = importlib.util.module_from_spec(spec)
spec.loader.exec_module(investigation_tools)

detect_storage_engine_pressure = investigation_tools.detect_storage_engine_pressure
detect_ratekeeper_throttling = investigation_tools.detect_ratekeeper_throttling
detect_missing_tlogs = investigation_tools.detect_missing_tlogs
detect_recovery_loop = investigation_tools.detect_recovery_loop
detect_coordination_loss = investigation_tools.detect_coordination_loss
detect_version_skew = investigation_tools.detect_version_skew
detect_process_class_mismatch = investigation_tools.detect_process_class_mismatch


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
    
    # Run all detectors
    detectors = [
        ("Storage Engine Pressure", detect_storage_engine_pressure),
        ("Ratekeeper Throttling", detect_ratekeeper_throttling),
        ("Missing TLogs", detect_missing_tlogs),
        ("Recovery Loop", detect_recovery_loop),
        ("Coordination Loss", detect_coordination_loss),
        ("Version Skew", detect_version_skew),
        ("Process Class Mismatch", detect_process_class_mismatch),
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

