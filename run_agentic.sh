#!/bin/bash

# === Agentic RCA Pipeline Runner ===

QUERY="$1"

if [ -z "$QUERY" ]; then
    echo "Error: You must provide a query string."
    echo "Example:"
    echo "  ./run_agentic.sh \"What issue is being tested?\""
    exit 1
fi

# Resolve directory of this script
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Correct paths
SIMLOGS_DIR="$SCRIPT_DIR/data/log_example/simlogs"
DB_DIR="$SCRIPT_DIR/data"
SECRETS_DIR="$SCRIPT_DIR/secrets/sa.json"

echo "=== Loading logs into DuckDB ==="

docker run --rm -it \
  --env-file "$SCRIPT_DIR/.env" \
  -v "$SIMLOGS_DIR:/logs" \
  -v "$DB_DIR:/data" \
  -v "$SECRETS_DIR:/secrets/sa.json:ro" \
  -e GOOGLE_APPLICATION_CREDENTIALS=/secrets/sa.json \
  vanshikachaddha/fdb-log-analyzer \
  load /logs --all --db /data/fdb_logs.duckdb

echo "=== Running RAG-powered RCA ==="

docker run --rm -it \
  --env-file "$SCRIPT_DIR/.env" \
  -v "$DB_DIR:/data" \
  -v "$SECRETS_DIR:/secrets/sa.json:ro" \
  -e GOOGLE_APPLICATION_CREDENTIALS=/secrets/sa.json \
  --entrypoint "" \
  vanshikachaddha/fdb-log-analyzer \
  python tools/agentic_loop/query_test.py \
      /data/fdb_logs.duckdb \
      "$QUERY" \
      --confidence 0.9 \
      --max-iterations 25 \
      --use-rag \
      --rag-corpus "$RAG_CORPUS_RESOURCE"

echo "=== Done! ==="
