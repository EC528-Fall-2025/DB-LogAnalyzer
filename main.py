#!/usr/bin/env python3
# main.py
from __future__ import annotations
import argparse, sys
from pathlib import Path
from service.agentic_loop import AgenticLoop
from service.db import get_conn
from service.ingest import iter_events_from_file, insert_events
from service.repository import load_events_window
from service.forced_recovery_chunker import ForcedRecoveryChunker

def main():
    parser = argparse.ArgumentParser(description="FoundationDB Log Analyzer CLI")
    subparsers = parser.add_subparsers(dest="command")

    # ── Direct LLM analysis on a log file (no DB) ───────────────────────────────
    p_agentic = subparsers.add_parser("agentic", help="Run agentic loop directly on a log file")
    p_agentic.add_argument("log_path", help="Path to FDB trace log file (XML/JSON/plain)")
    p_agentic.add_argument("--limit", type=int, default=None, help="Max events to process")
    p_agentic.add_argument("--include-codecoverage", action="store_true", default=False)
    p_agentic.add_argument("--z", type=float, default=2.0, help="Z-score threshold for anomalies")
    p_agentic.add_argument("--api-key", default=None, help="Gemini API key (else uses GEMINI_API_KEY)")

    # ── Full pipeline: parse → DuckDB → LLM ────────────────────────────────────
    p_pipe = subparsers.add_parser("pipeline", help="Ingest into DuckDB then run agentic loop")
    p_pipe.add_argument("log_path", help="Path to FDB trace log file (XML/JSON/plain)")
    p_pipe.add_argument("--db", default="data/fdb_logs.duckdb", help="DuckDB path")
    p_pipe.add_argument("--limit", type=int, default=None, help="Max events to ingest/analyze")
    p_pipe.add_argument("--include-codecoverage", action="store_true", default=False)
    p_pipe.add_argument("--z", type=float, default=2.0)
    p_pipe.add_argument("--api-key", default=None)

    # ── Export forced recovery chunks to XML files ─────────────────────────────
    p_chunk = subparsers.add_parser("chunk-xml", help="Export forced recovery chunks to separate XML files")
    p_chunk.add_argument("log_path", help="Path to FDB trace log file (XML/JSON/plain)")
    p_chunk.add_argument("--output-dir", default="output/chunks", help="Output directory for chunk XML files")
    p_chunk.add_argument("--prefix", default="chunk", help="Filename prefix for chunk files")
    p_chunk.add_argument("--kb", default=None, help="Path to knowledge base JSONL (default: samples/fdb_recovery_knowledgebase.jsonl)")

    args = parser.parse_args()

    if args.command == "agentic":
        run_agentic(args)
    elif args.command == "pipeline":
        run_pipeline(args)
    elif args.command == "chunk-xml":
        run_chunk_xml(args)
    else:
        parser.print_help()

def run_agentic(args):
    print(f"Agentic mode on: {args.log_path}")
    loop = AgenticLoop(
        z_score_threshold=args.z,
        recovery_lookback=5.0,
        auto_filter=True,
        use_ai=True,
        api_key=args.api_key
    )
    result = loop.run(args.log_path, limit=args.limit, include_codecoverage=args.include_codecoverage)
    loop.print_results(result)

def run_pipeline(args):
    print(f"Pipeline mode - ingest to DuckDB then run agentic loop")
    print(f"   Log: {args.log_path}")
    print(f"   DB : {args.db}")

    # 1) Ensure DB exists + DDL loaded
    con = get_conn(args.db)

    # 2) Ingest from file into DuckDB
    events_iter = iter_events_from_file(args.log_path, args.limit)
    inserted = insert_events(con, events_iter)
    print(f"   Inserted {inserted} events")

    # 3) Load back from DuckDB (optionally windowed)
    loaded = load_events_window(args.db, limit=args.limit)
    print(f"   Loaded {len(loaded)} events from DuckDB")

    # 4) Run agentic loop over loaded events
    loop = AgenticLoop(
        z_score_threshold=args.z,
        recovery_lookback=5.0,
        auto_filter=True,
        use_ai=True,
        api_key=args.api_key
    )
    result = loop.run_on_events(loaded, include_codecoverage=args.include_codecoverage)
    loop.print_results(result)

def run_chunk_xml(args):
    """Export forced recovery chunks to separate XML files."""
    print(f"Chunking log into XML files")
    print(f"   Input : {args.log_path}")
    print(f"   Output: {args.output_dir}")
    
    # Initialize chunker
    kb_path = Path(args.kb) if args.kb else None
    chunker = ForcedRecoveryChunker(knowledge_base_path=kb_path)
    
    # Export chunks to XML files
    try:
        created_files = chunker.export_chunks_to_xml(
            log_path=args.log_path,
            output_dir=args.output_dir,
            prefix=args.prefix,
        )
        
        print(f"   Created {len(created_files)} chunk files:")
        for filepath in created_files:
            # Get file size for display
            size = Path(filepath).stat().st_size
            size_kb = size / 1024
            print(f"      - {filepath} ({size_kb:.1f} KB)")
        
    except Exception as e:
        print(f"   Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    sys.exit(main())
