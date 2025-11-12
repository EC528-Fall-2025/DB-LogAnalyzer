import os, sys
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path: sys.path.insert(0, REPO)

from db_log_analyzer.ingestion.log_loader import load_events_df
from db_log_analyzer.ingestion.log_to_documents import logs_df_to_documents
from db_log_analyzer.ingestion.log_chunking import choose_log_chunks, _filter_noisy

if __name__ == "__main__":
    db_path = "data/fdb_logs.duckdb"

    df = load_events_df(db_path)
    print(f"Loaded {len(df)} rows from {db_path}")

    docs = logs_df_to_documents(df, source_file=db_path)
    print(f"Converted to {len(docs)} row-documents")

    # Drop noisy events like BuggifySection
    docs = _filter_noisy(docs)
    print(f"After noise filter: {len(docs)} row-documents")

    chunks, mode = choose_log_chunks(docs)
    print(f"Produced {len(chunks)} log chunks (mode={mode})\n")

    # Preview first 2 chunks
    for i, c in enumerate(chunks[:2]):
        print(f"--- Chunk {i} ---")
        print("Type:", c.metadata.get("chunk_type"))
        print("Size:", c.metadata.get("chunk_size"))
        print("Span:", c.metadata.get("start_ts"), "→", c.metadata.get("end_ts"))
        for ln in c.page_content.splitlines()[:10]:
            print(ln)
        print()
