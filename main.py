#!/usr/bin/env python3
"""Main entry point for FDB Log Analyzer CLI."""
from __future__ import annotations

import argparse
import sys

from tools.agentic_loop.agentic_loop import AgenticLoop
from tools.agentic_loop.investigation_agent import load_events_window
from cli_wrapper.main import CLI

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="FoundationDB Log Analyzer CLI")
    subparsers = parser.add_subparsers(dest="command")

    # Direct LLM analysis on a DuckDB database
    p_agentic = subparsers.add_parser("agentic", help="Run agentic loop on a DuckDB database")
    p_agentic.add_argument("log_path", help="Path to DuckDB database file")
    p_agentic.add_argument("--limit", type=int, default=None, help="Max events to process")
    p_agentic.add_argument("--include-codecoverage", action="store_true", default=False)
    p_agentic.add_argument("--z", type=float, default=2.0, help="Z-score threshold for anomalies")
    p_agentic.add_argument("--api-key", default=None, help="Gemini API key (else uses GEMINI_API_KEY)")

    # Full pipeline: parse → DuckDB → LLM
    p_pipe = subparsers.add_parser("pipeline", help="Ingest into DuckDB then run agentic loop")
    p_pipe.add_argument("log_path", help="Path to FDB trace log file (XML/JSON/plain)")
    p_pipe.add_argument("--db", default="data/fdb_logs.duckdb", help="DuckDB path")
    p_pipe.add_argument("--limit", type=int, default=None, help="Max events to ingest/analyze")
    p_pipe.add_argument("--include-codecoverage", action="store_true", default=False)
    p_pipe.add_argument("--z", type=float, default=2.0, help="Z-score threshold for anomalies")
    p_pipe.add_argument("--api-key", default=None, help="Gemini API key (else uses GEMINI_API_KEY)")

    args = parser.parse_args()

    if args.command == "agentic":
        run_agentic(args)
    elif args.command == "pipeline":
        run_pipeline(args)
    else:
        parser.print_help()

def run_agentic(args):
    """Run agentic loop on DuckDB database."""
    print(f"Agentic mode on DuckDB: {args.log_path}")
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
    """Run full pipeline: ingest into DuckDB then run agentic loop."""
    print(f"Pipeline mode — ingest → DuckDB → agentic")
    print(f"   Log: {args.log_path}")
    print(f"   DB : {args.db}")

    # Ensure DB exists + DDL loaded
    cli = CLI()
    cli_args = argparse.Namespace(
        input=args.log_path,
        output=args.db,
        format="duckdb",
        schema="data_storage/schema.sql" if hasattr(args, "schema") else None
    )

    cli.handle_pipeline(cli_args)

    loaded = load_events_window(args.db, limit=args.limit)
    print(f"   ✓ Loaded {len(loaded)} events from DuckDB")

    # Run agentic loop over loaded events
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
