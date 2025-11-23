#!/usr/bin/env python3
"""
Run forced recovery chunking without the CLI.

Usage:
    python scripts/chunk_logs.py --log samples/sample1.xml --limit 3
    python scripts/chunk_logs.py --log samples/sample1.xml --output chunks.json --include-events
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import List, Dict, Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from service.forced_recovery_chunker import ForcedRecoveryChunker


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Chunk logs from forced recovery triggers to MasterRecoveryState StatusCode=14."
    )
    parser.add_argument(
        "--log",
        required=True,
        help="Path to the log file (XML/JSON/plaintext supported by LogParser).",
    )
    parser.add_argument(
        "--knowledge-base",
        help="Optional path to knowledge base JSONL file. Defaults to samples/fdb_recovery_knowledgebase.jsonl.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of chunk summaries to display (default: 5).",
    )
    parser.add_argument(
        "--output",
        help="Write chunk data to JSON file.",
    )
    parser.add_argument(
        "--include-events",
        action="store_true",
        help="Include individual events in the JSON output (default: off).",
    )
    return parser.parse_args()


def print_summaries(chunks: List[Dict[str, Any]], limit: int) -> None:
    """Pretty-print chunk summaries."""
    if not chunks:
        print("No forced recovery chunks detected.")
        return

    print(f"Detected {len(chunks)} forced recovery chunk(s).\n")
    for idx, chunk in enumerate(chunks[:limit], start=1):
        status = "complete" if chunk["complete"] else "incomplete"
        start_comment = chunk.get("start_comment") or "(no comment)"
        print(f" Chunk {idx}: {chunk['start_time']} → {chunk['end_time']}")
        print(f"    Status: {status}, Events: {chunk['event_count']}, Start Comment: {start_comment}")
        if not chunk["complete"]:
            print("      MasterRecoveryState StatusCode=14 not reached within this chunk.")
        print()


def main() -> None:
    """Entry point."""
    args = parse_args()
    log_path = Path(args.log).expanduser()
    if not log_path.exists():
        raise FileNotFoundError(f"Log file not found: {log_path}")

    kb_path = Path(args.knowledge_base).expanduser() if args.knowledge_base else None
    chunker = ForcedRecoveryChunker(knowledge_base_path=kb_path)
    chunk_objects = chunker.chunk_events(chunker.parser.parse_logs(str(log_path)))
    include_events = args.include_events or bool(args.output)
    chunk_dicts = [
        chunk.to_dict(include_events=include_events)
        for chunk in chunk_objects
    ]

    print_summaries(chunk_dicts, args.limit)

    if args.output:
        output_path = Path(args.output).expanduser()
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(chunk_dicts, handle, ensure_ascii=False, indent=2)
        print(f"Wrote chunk data to: {output_path}")


if __name__ == "__main__":
    main()


