# service/db.py
from __future__ import annotations
import duckdb
from pathlib import Path

DEFAULT_DB = "data/fdb_logs.duckdb"

DDL = """
PRAGMA enable_object_cache;
CREATE TABLE IF NOT EXISTS events (
  id           UUID PRIMARY KEY DEFAULT uuid(),
  event        TEXT,
  ts           TIMESTAMP,
  severity     INTEGER,
  role         TEXT,
  fields_json  JSON
);
CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts);
CREATE INDEX IF NOT EXISTS idx_events_event ON events(event);
CREATE INDEX IF NOT EXISTS idx_events_severity ON events(severity);
"""

def get_conn(db_path: str = DEFAULT_DB) -> duckdb.DuckDBPyConnection:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(db_path)
    con.execute(DDL)
    return con