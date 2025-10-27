# service/ingest.py
from __future__ import annotations
from typing import Iterable, Dict, Any, List
import json     
from service.db import get_conn
from service.parser import LogParser
from dto.event import EventModel

def iter_events_from_file(path: str, limit: int | None = None) -> Iterable[EventModel]:
    p = LogParser()
    for i, ev in enumerate(p.parse_logs(path)):
        if limit is not None and i >= limit:
            break
        yield ev

def insert_events(con, events: Iterable[EventModel]) -> int:
    # Buffer rows for vectorized insert
    buf: List[Dict[str, Any]] = []
    count = 0
    for e in events:
        buf.append({
            "event": e.event,
            "ts": e.ts,
            "severity": e.severity,
            "role": e.role,
            "fields_json": e.fields_json or {}
        })
        if len(buf) >= 10_000:
            _flush(con, buf); count += len(buf); buf.clear()
    if buf:
        _flush(con, buf); count += len(buf)
    return count

def _flush(con, rows: List[Dict[str, Any]]):
    # executemany with a JSON cast for fields_json
    sql = """
        INSERT INTO events (event, ts, severity, role, fields_json)
        VALUES (?, ?, ?, ?, CAST(? AS JSON))
    """
    params = [
        (r["event"], r["ts"], r["severity"], r["role"], json.dumps(r["fields_json"]))
        for r in rows
    ]
    con.executemany(sql, params)