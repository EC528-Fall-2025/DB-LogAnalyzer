# scripts/chunk_to_duckdb.py
from __future__ import annotations
import argparse
from pathlib import Path
import duckdb
from service.forced_recovery_chunker import ForcedRecoveryChunker

def ensure_tables(con):
    con.execute("""
        CREATE TABLE IF NOT EXISTS recovery_chunks (
            chunk_id BIGINT,
            chunk_index INTEGER,
            start_time TIMESTAMP,
            end_time TIMESTAMP,
            start_type VARCHAR,
            complete BOOLEAN,
            event_count INTEGER,
            start_comment VARCHAR
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS recovery_chunk_events (
            chunk_id BIGINT,
            event_index INTEGER,
            event_id VARCHAR,
            datetime TIMESTAMP,
            type VARCHAR,
            severity INTEGER,
            time DOUBLE,
            fields JSON
        )
    """)

def drop_tables(con):
    con.execute("DROP TABLE IF EXISTS recovery_chunk_events")
    con.execute("DROP TABLE IF EXISTS recovery_chunks")

def store_chunks(db_path: str, chunks: list, include_events: bool = True, reset: bool = False):
    con = duckdb.connect(db_path)
    if reset:
        drop_tables(con)
    ensure_tables(con)

    # next chunk_id
    (max_id,) = con.execute("SELECT COALESCE(MAX(chunk_id),0) FROM recovery_chunks").fetchone()
    next_id = max_id + 1

    for c in chunks:
        cid = next_id
        next_id += 1
        con.execute("""
            INSERT INTO recovery_chunks
            (chunk_id, chunk_index, start_time, end_time, start_type, complete, event_count, start_comment)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            cid,
            c["index"],
            c["start_time"],
            c["end_time"],
            c["start_type"],
            c["complete"],
            c["event_count"],
            c.get("start_comment",""),
        ])
        if include_events and "events" in c:
            rows = []
            for i, ev in enumerate(c["events"], start=1):
                rows.append([
                    cid,
                    i,
                    ev["event_id"],
                    ev["datetime"],
                    ev["type"],
                    ev["severity"] if ev["severity"] is not None else None,
                    ev["time"],
                    ev["fields"]
                ])
            con.execute("""
                INSERT INTO recovery_chunk_events
                (chunk_id, event_index, event_id, datetime, type, severity, time, fields)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, rows)

    con.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True, help="Path to FDB log (XML lines).")
    ap.add_argument("--db", required=True, help="DuckDB path (e.g., data/chunks.duckdb)")
    ap.add_argument("--reset", action="store_true", help="Drop & recreate tables")
    ap.add_argument("--include-events", action="store_true", help="Store per-event rows too")
    args = ap.parse_args()

    log_path = Path(args.log)
    if not log_path.exists():
        raise FileNotFoundError(f"Log not found: {log_path}")

    chunker = ForcedRecoveryChunker()
    events = chunker.parser.parse_logs(str(log_path))
    chunk_objs = chunker.chunk_events(events)
    chunk_dicts = [c.to_dict(include_events=args.include_events or True) for c in chunk_objs]

    print(f"Detected {len(chunk_dicts)} forced-recovery chunk(s).")
    if chunk_dicts[:1]:
        c0 = chunk_dicts[0]
        print(f"  First: {c0['start_time']} → {c0['end_time']}  ({c0['event_count']} ev) complete={c0['complete']}")

    store_chunks(args.db, chunk_dicts, include_events=True, reset=args.reset)
    print(f"✅ Stored {len(chunk_dicts)} chunk(s) into: {args.db}")

if __name__ == "__main__":
    main()
