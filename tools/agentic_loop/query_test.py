import sys
from pathlib import Path

# Add project root to path (needed when running as script)
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from datetime import datetime
from tools.agentic_loop.investigation_agent import InvestigationContext, QueryGenerator, InvestigationAgent

# Create context
context = InvestigationContext(
    db_path="data/fdb_logs.duckdb"
)

# Create generator
generator = QueryGenerator(default_limit=1000)


# Test full investigation agent
print("\n" + "=" * 60)
print("Testing Full Investigation Agent")
print("=" * 60)

# Check if API key is available
import os
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("\n⚠️  GEMINI_API_KEY not set. Skipping full investigation test.")
    print("   Set GEMINI_API_KEY environment variable to test investigate() method.")
else:
    print("\n🔍 Running full investigation...")
    print("   Question: 'What is the specific issue being tested in this simulation?'")
    print("   This may take a moment (calls LLM)...\n")
    
    try:
        # Initialize agent - allow more iterations to find root cause
        agent = InvestigationAgent(
            db_path="data/fdb_logs.duckdb",
            max_iterations= 50,  # Allow more iterations to investigate root cause
            confidence_threshold=0.90
        )
        
        # Run investigation
        result = agent.investigate(
            initial_question="What is the specific issue being tested in this simulation? NOT FILE_NOT_FOUND",
            api_key=api_key,
            log_file_paths="/Users/vanshikachaddha/Documents/Boston University/Fourth Year/Cloud Computing/Dan's Logs/Log 1/simlogs_writeduringreadclean"
        )
        
        # Display results
        print("\n" + "=" * 60)
        print("INVESTIGATION RESULTS")
        print("=" * 60)
        print(f"\nHypothesis: {result.hypothesis}")
        print(f"Confidence: {result.confidence:.2f}")
        print(f"Iterations: {result.iterations}")
        print(f"Queries Executed: {result.query_count}")
        print(f"\nEvidence Events: {len(result.evidence_events)}")
        if result.evidence_events:
            print("\nTop Evidence Events:")
            for i, event in enumerate(result.evidence_events[:5], 1):
                print(f"  {i}. {event.event} - {event.ts} - severity {event.severity}")
        if result.reasoning:
            print(f"\nReasoning: {result.reasoning}")
        print("\nInvestigation completed!")
        
    except Exception as e:
        print(f"\nError during investigation: {e}")
        import traceback
        traceback.print_exc()