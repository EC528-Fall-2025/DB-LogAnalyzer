"""Helper utilities for investigation tools."""

import json
import uuid
from datetime import datetime

from data_transfer_object.event_dto import EventModel


def _ensure_dict(maybe_json):
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


def _uuid_to_int(u):
    try:
        if isinstance(u, uuid.UUID):
            n = u.int
        else:
            n = uuid.UUID(str(u)).int
        return n & ((1 << 63) - 1)
    except Exception:
        return 0


def _percentile(values: list, pct: float):
    """Lightweight percentile without numpy."""
    if not values:
        return None
    pct = max(0.0, min(100.0, pct))
    vals = sorted(values)
    k = (len(vals) - 1) * (pct / 100.0)
    f = int(k)
    c = min(f + 1, len(vals) - 1)
    if f == c:
        return vals[int(k)]
    d0 = vals[f] * (c - k)
    d1 = vals[c] * (k - f)
    return d0 + d1


def _parse_event_row(row):
    rid, ts, sev, event, role, fields = row
    fdict = _ensure_dict(fields)

    if ts and not isinstance(ts, datetime):
        ts = datetime.fromtimestamp(ts)

    raw_blob = {
        "id": str(rid),
        "event": event,
        "ts": ts.isoformat() if ts else None,
        "severity": sev,
        "role": role,
        "fields_json": fdict,
    }

    return EventModel(
        event=event,
        ts=ts,
        severity=sev,
        role=role,
        fields_json=fdict,
        event_id=_uuid_to_int(rid),
        process=role,
        pid=None,
        machine_id=None,
        address=None,
        trace_file=None,
        src_line=None,
        raw_json=raw_blob,
    )


def _build_conditions(filters: dict):
    conditions = []
    params = []

    if filters.get("start_time"):
        conditions.append("ts >= ?")
        params.append(filters["start_time"])

    if filters.get("end_time"):
        conditions.append("ts <= ?")
        params.append(filters["end_time"])

    if filters.get("severity_min") is not None:
        conditions.append("severity >= ?")
        params.append(filters["severity_min"])

    if filters.get("severity_max") is not None:
        conditions.append("severity <= ?")
        params.append(filters["severity_max"])

    if filters.get("event_type"):
        et = filters["event_type"]
        if "%" in et:
            conditions.append("event LIKE ?")
            params.append(et)
        else:
            conditions.append("event = ?")
            params.append(et)

    if filters.get("role"):
        conditions.append("role = ?")
        params.append(filters["role"])

    where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""
    return where_clause, params
