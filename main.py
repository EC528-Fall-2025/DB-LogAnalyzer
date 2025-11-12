#!/usr/bin/env python3
# main.py
from __future__ import annotations
import argparse, sys
from service.agentic_loop import AgenticLoop
from service.db import get_conn
from service.ingest import iter_events_from_file, insert_events
from service.repository import load_events_window
from cli.main import CLI

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

    args = parser.parse_args()

    if args.command == "agentic":
        run_agentic(args)
    elif args.command == "pipeline":
        run_pipeline(args)
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
    print(f"Pipeline mode — ingest → DuckDB → agentic")
    print(f"   Log: {args.log_path}")
    print(f"   DB : {args.db}")

    # 1) Ensure DB exists + DDL loaded
    
    cli = CLI()
    cli_args = argparse.Namespace(
        input=args.log_path,
        output=args.db,
        format="duckdb",
        schema="data/schema.sql" if hasattr(args, "schema") else None
    )

    cli.handle_pipeline(cli_args)

    loaded = load_events_window(args.db, limit=args.limit)
    print(f"   ✓ Loaded {len(loaded)} events from DuckDB")

    # 2) Run agentic loop over loaded events
    loop = AgenticLoop(
        z_score_threshold=args.z,
        recovery_lookback=5.0,
        auto_filter=True,
        use_ai=True,
        api_key=args.api_key
    )
    result = loop.run_on_events(loaded, include_codecoverage=args.include_codecoverage)
    loop.print_results(result)

if __name__ == "__main__":
    sys.exit(main())
