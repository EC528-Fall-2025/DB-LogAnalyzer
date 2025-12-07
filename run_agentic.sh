#!/bin/bash

# === Agentic RCA Pipeline Runner ===
# Usage:
#   ./run_agentic.sh "What issue is being tested?"
#
# Requirements:
#   - .env file present (copy from .env.example)
#   - logs in ./simlogs
#   - secrets/sa.json present
#   - Docker Desktop running

QUERY="$1"

if [ -z "$QUERY" ]; then
    echo "Error: You must provide a query string."
    echo "Example:"
    echo "  ./run_agentic.sh \"What issue is being tested?\""
    exit 1
fi

# === Step 1: Load logs into DuckDB ===
echo "=== Loading logs into DuckDB ==="
docker run --rm -it \
  --env-file .env \
  -v "$(pwd)/simlogs:/logs" \
  -v "$(pwd)/data:/data" \
  -v "$(pwd)/secrets/sa.json:/secrets/sa.json:ro" \
  -e GOOGLE_APPLICATION_CREDENTIALS=/secrets/sa.json \
  vanshikachaddha/fdb-log-analyzer \
  load /logs --all --db /data/fdb_logs.duckdb

# === Step 2: Run RAG-powered agentic RCA ===
echo "=== Running RAG-powered RCA ==="
docker run --rm -it \
  --env-file .env \
  -v "$(pwd)/data:/data" \
  -v "$(pwd)/secrets/sa.json:/secrets/sa.json:ro" \
  -e GOOGLE_APPLICATION_CREDENTIALS=/secrets/sa.json \
  vanshikachaddha/fdb-log-analyzer \
  tools/agentic_loop/query_test.py \
      /data/fdb_logs.duckdb \
      "$QUERY" \
      --confidence 0.9 \
      --max-iterations 25 \
      --use-rag \
      --rag-corpus "$RAG_CORPUS_RESOURCE"

echo "=== Done! ==="