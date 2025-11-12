# scripts/test_load_step1.py
import sys
from db_log_analyzer.ingestion.load_code_coverage import load_code_coverage_jsonl

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python scripts/test_load_step1.py data\\fdb_recovery_knowledgebase.jsonl")
        sys.exit(1)

    path = sys.argv[1]
    docs = load_code_coverage_jsonl(path)
    print(f"Loaded {len(docs)} documents.\n")

    if docs:
        print("Preview (first 250 chars):\n")
        print(docs[0].page_content[:250])
        print("\nMetadata:")
        print(docs[0].metadata)
