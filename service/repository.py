# service/repository.py
from __future__ import annotations
from typing import List, Optional
import json, uuid
from dto.event import EventModel
from service.db import get_conn

def _ensure_dict(maybe_json) -> dict:
    if maybe_json is None:
        return {}
    if isinstance(maybe_json, dict):
        return maybe_json
    if isinstance(maybe_json, str):
        try:
            return json.loads(maybe_json)
        except Exception:
            return {"_raw": maybe_json}
    return {}

def _uuid_to_int(u) -> int:
    """
    Convert DuckDB UUID (or str UUID) to a stable 64-bit-ish int that fits typical int fields.
    """
    try:
        if isinstance(u, uuid.UUID):
            n = u.int
        else:
            n = uuid.UUID(str(u)).int
        # squeeze to signed 63-bit range to be safe for pydantic/int
        return n & ((1 << 63) - 1)
    except Exception:
        return 0  # fallback

def load_events_window(
    db_path: str,
    start_ts: Optional[str] = None,
    end_ts: Optional[str] = None,
    limit: Optional[int] = None,
) -> List[EventModel]:
    """
    Read events from DuckDB and return EventModel objects.
    We hydrate required fields and coerce JSON where needed.
    """
    con = get_conn(db_path)

    where = []
    params = []
    if start_ts:
        where.append("ts >= ?"); params.append(start_ts)
    if end_ts:
        where.append("ts < ?"); params.append(end_ts)
    where_clause = ("WHERE " + " AND ".join(where)) if where else ""
    lim = f"LIMIT {int(limit)}" if limit else ""

    q = f"""
      SELECT event_id, event, ts, severity, role, fields_json
      FROM events
      {where_clause}
      ORDER BY ts ASC
      {lim}
    """
    rows = con.execute(q, params).fetchall()

    out: List[EventModel] = []
    for rid, event, ts, sev, role, fields in rows:
        fdict = _ensure_dict(fields)

        raw_blob = {
            "id": str(rid) if rid is not None else None,
            "event": event,
            "ts": ts.isoformat() if ts else None,
            "severity": sev,
            "role": role,
            "fields_json": fdict,
        }

        em = EventModel(
            # core
            event=event,
            ts=ts,
            severity=sev,
            role=role,
            fields_json=fdict,

            # EventModel-required fields
            event_id=_uuid_to_int(rid),     # <- int, derived from UUID
            process=role or None,           # best-effort mapping
            pid=None,
            machine_id=None,
            address=None,
            trace_file=None,
            src_line=None,
            raw_json=raw_blob,              # <- dict, not string
        )
        out.append(em)

    return out
