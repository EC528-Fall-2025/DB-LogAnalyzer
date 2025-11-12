import os, sys
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from db_log_analyzer.ingestion.load_code_coverage import load_code_coverage_jsonl
from db_log_analyzer.ingestion.chunking import split_docs

if __name__ == "__main__":
    path = "data/fdb_recovery_knowledgebase.jsonl"
    docs = load_code_coverage_jsonl(path)
    print(f"Loaded {len(docs)} documents.")

    chunks = split_docs(docs)
    print(f"Produced {len(chunks)} chunks.\n")

    # Show a few preview chunks
    for i, ch in enumerate(chunks[:3]):
        print(f"--- Chunk {i} ---")
        print(ch.page_content[:300])
        print("Metadata:", ch.metadata)
        print()
