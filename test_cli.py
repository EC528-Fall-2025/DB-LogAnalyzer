#!/usr/bin/env python3
"""
Quick test script for CLI functionality
"""

import sys
import os
from cli.main import CLI

def test_cli():
    """Test basic CLI functionality"""
    print("=== Testing FDB Log Analyzer CLI ===\n")
    
    cli = CLI()
    
    # Test 1: Display help
    print("1. Testing help command...")
    try:
        cli.run(["--help"])
    except SystemExit:
        # --help triggers SystemExit, which is expected behavior
        print("✓ Help command executed successfully")
    print("\n" + "="*50 + "\n")
    
    # Test 2: Initialize database
    print("2. Testing database initialization...")
    if os.path.exists("data/schema.sql"):
        cli.run(["--db", "test.duckdb", "init", "--schema", "data/schema.sql"])
        print("✓ Database initialized successfully")
    else:
        print("⚠ schema.sql file not found, skipping initialization test")
    print("\n" + "="*50 + "\n")
    
    # Test 3: Parse logs (if sample file exists)
    print("3. Testing parse functionality...")
    if os.path.exists("data/sample_log.json"):
        cli.run(["parse", "data/sample_log.json", "--limit", "3"])
        print("✓ Parse functionality working")
    else:
        print("⚠ Sample log file not found, skipping parse test")

    # Test 4: Pipeline execution (if sample file exists)
    print("4. Testing pipeline command...")
    if os.path.exists("data/sample_log.json"):
        cli.run(["pipeline", "--input", "data/sample_log.json", "--output", "test_pipeline.duckdb"])
        print("✓ Pipeline command executed successfully")
    else:
        print("⚠ Sample log file not found, skipping pipeline test")
    print("\n" + "="*50 + "\n")
    
    # Test 5: Rollup command
    print("5. Testing rollup command...")
    cli.run(["--db", "test_pipeline.duckdb", "rollup", "--interval", "60"])
    print("✓ Rollup command executed successfully")
    
    print("\n=== Testing Complete ===")
    
    # Clean up test files
    if os.path.exists("test.duckdb"):
        os.remove("test.duckdb")
        print("Test database file cleaned up")
    if os.path.exists("test_pipeline.duckdb"):
        os.remove("test_pipeline.duckdb")
        print("Pipeline test database file cleaned up")

if __name__ == "__main__":
    test_cli()
