#!/usr/bin/env python3
"""
Test script for InvestigationAgent.

Usage:
    python tools/agentic_loop/query_test.py <db_path> [question]
"""

import sys
import os
from pathlib import Path

# Add project root to path (needed when running as script)
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from datetime import datetime
from tools.agentic_loop.investigation_agent import InvestigationAgent


def main():
    """Main test function."""
    if len(sys.argv) < 2:
        print("Usage: python tools/agentic_loop/query_test.py <db_path> [question] [--confidence THRESHOLD] [--max-iterations N]")
        print("\nArguments:")
        print("  db_path              Path to DuckDB database file (required)")
        print("  question             Question to investigate (optional)")
        print("  --confidence THRESHOLD  Confidence threshold (0.0-1.0, default: 0.8)")
        print("  --max-iterations N   Maximum iterations (default: 10)")
        print("\nExample:")
        print("  python tools/agentic_loop/query_test.py data/fdb_logs.duckdb 'What issue is being tested?'")
        print("  python tools/agentic_loop/query_test.py data/fdb_logs.duckdb --confidence 0.9")
        sys.exit(1)
    
    db_path = sys.argv[1]
    question = "What issue or scenario is being tested in these logs?"
    confidence_threshold = 0.8
    max_iterations = 10
    use_rag = None
    rag_corpus = None
    
    # Parse arguments
    i = 2
    while i < len(sys.argv):
        if sys.argv[i] == "--confidence" and i + 1 < len(sys.argv):
            confidence_threshold = float(sys.argv[i + 1])
            if not 0.0 <= confidence_threshold <= 1.0:
                print(f"Error: Confidence threshold must be between 0.0 and 1.0, got {confidence_threshold}")
                sys.exit(1)
            i += 2
        elif sys.argv[i] == "--max-iterations" and i + 1 < len(sys.argv):
            max_iterations = int(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == "--use-rag":
            use_rag = True
            i += 1
        elif sys.argv[i] == "--no-rag":
            use_rag = False
            i += 1
        elif sys.argv[i] == "--rag-corpus" and i + 1 < len(sys.argv):
            rag_corpus = sys.argv[i + 1]
            i += 2
        elif not sys.argv[i].startswith("--"):
            question = sys.argv[i]
            i += 1
        else:
            print(f"Error: Unknown argument: {sys.argv[i]}")
            sys.exit(1)
    
    if not Path(db_path).exists():
        print(f"Error: Database file not found: {db_path}")
        sys.exit(1)
    
    # Get API key from environment or prompt
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Warning: GEMINI_API_KEY not set. Please set it as an environment variable.")
        print("  export GEMINI_API_KEY='your-api-key'")
        sys.exit(1)
    
    print(f"\n🔍 Testing InvestigationAgent")
    print(f"   Database: {db_path}")
    print(f"   Question: {question}")
    print(f"   Confidence Threshold: {confidence_threshold}")
    print(f"   Max Iterations: {max_iterations}\n")
    
    try:
        # Create agent with custom confidence threshold
        agent = InvestigationAgent(
            db_path=db_path,
            max_iterations=max_iterations,
            confidence_threshold=confidence_threshold,
            use_rag=use_rag,
            rag_corpus=rag_corpus,
        )
        
        # Run investigation
        result = agent.investigate(
            initial_question=question,
            api_key=api_key
        )
        
        # Print results
        print("\n" + "=" * 70)
        print("RESULTS:")
        print("=" * 70)
        print(f"\nHypothesis: {result.hypothesis}")
        print(f"Confidence: {result.confidence:.2f}")
        print(f"Reasoning: {result.reasoning}")
        print(f"Tools Used: {', '.join(result.tools_used)}")
        print("\n")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
