"""
Investigation Tools - Database query utilities for FDB log analysis.

All tools take db_path as input and return structured data for LLM analysis.
"""

import json
import uuid
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter, defaultdict

from data_transfer_object.event_dto import EventModel
from tools.database import get_conn


def _ensure_dict(maybe_json) -> dict:
    """Ensure value is a dictionary."""
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
    """Convert DuckDB UUID (or str UUID) to a stable 64-bit-ish int."""
    try:
        if isinstance(u, uuid.UUID):
            n = u.int
        else:
            n = uuid.UUID(str(u)).int
        return n & ((1 << 63) - 1)
    except Exception:
        return 0


def _parse_event_row(row) -> EventModel:
    """Parse a database row into EventModel."""
    rid, ts, sev, event, role, fields = row
    fdict = _ensure_dict(fields)
    
    ts_datetime = None
    if ts:
        if isinstance(ts, datetime):
            ts_datetime = ts
        elif isinstance(ts, (int, float)):
            ts_datetime = datetime.fromtimestamp(ts)
    
    raw_blob = {
        "id": str(rid) if rid is not None else None,
        "event": event,
        "ts": ts_datetime.isoformat() if ts_datetime else None,
        "severity": sev,
        "role": role,
        "fields_json": fdict,
    }
    
    return EventModel(
        event=event,
        ts=ts_datetime,
        severity=sev,
        role=role,
        fields_json=fdict,
        event_id=_uuid_to_int(rid) if rid else 0,
        process=role or None,
        pid=None,
        machine_id=None,
        address=None,
        trace_file=None,
        src_line=None,
        raw_json=raw_blob,
    )


# ============================================================================
# DATA RETRIEVAL TOOLS
# ============================================================================

def get_events(
    db_path: str,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    severity_min: Optional[int] = None,
    severity_max: Optional[int] = None,
    event_type: Optional[str] = None,
    role: Optional[str] = None,
    limit: int = 1000
) -> List[EventModel]:
    """
    Retrieve filtered event slices from database.
    
    Args:
        db_path: Path to DuckDB database
        start_time: Optional start timestamp filter
        end_time: Optional end timestamp filter
        severity_min: Minimum severity (inclusive)
        severity_max: Maximum severity (inclusive)
        event_type: Optional event type filter (exact match or LIKE pattern)
        role: Optional role filter
        limit: Maximum number of events to return
        
    Returns:
        List of EventModel instances
    """
    con = get_conn(db_path)
    
    conditions = []
    params = []
    
    if start_time:
        conditions.append("ts >= ?")
        params.append(start_time)
    if end_time:
        conditions.append("ts <= ?")
        params.append(end_time)
    
    if severity_min is not None:
        conditions.append("severity >= ?")
        params.append(severity_min)
    
    if severity_max is not None:
        conditions.append("severity <= ?")
        params.append(severity_max)
    
    if event_type:
        if '%' in event_type:
            conditions.append("event LIKE ?")
            params.append(event_type)
        else:
            conditions.append("event = ?")
            params.append(event_type)
    
    if role:
        conditions.append("role = ?")
        params.append(role)
    
    where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""
    
    sql = f"""
        SELECT event_id, ts, severity, event, role, fields_json
        FROM events
        {where_clause}
        ORDER BY ts ASC
        LIMIT {limit}
    """
    
    rows = con.execute(sql, params).fetchall()
    return [_parse_event_row(row) for row in rows]


def sample_events(
    db_path: str,
    sample_size: int = 100,
    from_start: bool = True
) -> List[EventModel]:
    """
    Get small sample of events (head or tail).
    
    Args:
        db_path: Path to DuckDB database
        sample_size: Number of events to sample
        from_start: If True, sample from start; if False, from end
        
    Returns:
        List of EventModel instances
    """
    con = get_conn(db_path)
    
    if from_start:
        sql = f"""
            SELECT event_id, ts, severity, event, role, fields_json
            FROM events
            ORDER BY ts ASC
            LIMIT {sample_size}
        """
    else:
        sql = f"""
            SELECT event_id, ts, severity, event, role, fields_json
            FROM events
            ORDER BY ts DESC
            LIMIT {sample_size}
        """
    
    rows = con.execute(sql).fetchall()
    events = [_parse_event_row(row) for row in rows]
    
    if not from_start:
        events.reverse()  # Keep chronological order
    
    return events


def first_occurrence(
    db_path: str,
    event_type: str,
    severity_min: Optional[int] = None
) -> Optional[EventModel]:
    """
    Find earliest occurrence of an event type.
    
    Args:
        db_path: Path to DuckDB database
        event_type: Event type to search for (supports LIKE patterns with %)
        severity_min: Optional minimum severity filter
        
    Returns:
        EventModel of first occurrence, or None if not found
    """
    con = get_conn(db_path)
    
    conditions = []
    params = []
    
    if '%' in event_type:
        conditions.append("event LIKE ?")
        params.append(event_type)
    else:
        conditions.append("event = ?")
        params.append(event_type)
    
    if severity_min is not None:
        conditions.append("severity >= ?")
        params.append(severity_min)
    
    where_clause = "WHERE " + " AND ".join(conditions)
    
    sql = f"""
        SELECT event_id, ts, severity, event, role, fields_json
        FROM events
        {where_clause}
        ORDER BY ts ASC
        LIMIT 1
    """
    
    rows = con.execute(sql, params).fetchall()
    if rows:
        return _parse_event_row(rows[0])
    return None


def changes_after(
    db_path: str,
    after_time: datetime,
    event_types: Optional[List[str]] = None,
    limit: int = 500
) -> List[EventModel]:
    """
    See what events occurred after a specific time.
    
    Args:
        db_path: Path to DuckDB database
        after_time: Timestamp to look after
        event_types: Optional list of event types to filter (None = all)
        limit: Maximum number of events to return
        
    Returns:
        List of EventModel instances
    """
    con = get_conn(db_path)
    
    conditions = ["ts > ?"]
    params = [after_time]
    
    if event_types:
        event_conditions = []
        for et in event_types:
            if '%' in et:
                event_conditions.append("event LIKE ?")
                params.append(et)
            else:
                event_conditions.append("event = ?")
                params.append(et)
        if event_conditions:
            conditions.append(f"({' OR '.join(event_conditions)})")
    
    where_clause = "WHERE " + " AND ".join(conditions)
    
    sql = f"""
        SELECT event_id, ts, severity, event, role, fields_json
        FROM events
        {where_clause}
        ORDER BY ts ASC
        LIMIT {limit}
    """
    
    rows = con.execute(sql, params).fetchall()
    return [_parse_event_row(row) for row in rows]


def top_events_by_severity(
    db_path: str,
    top_n: int = 50,
    severity_min: int = 40
) -> List[EventModel]:
    """
    Find most severe events.
    
    Args:
        db_path: Path to DuckDB database
        top_n: Number of top events to return
        severity_min: Minimum severity threshold
        
    Returns:
        List of EventModel instances, sorted by severity DESC then time DESC
    """
    con = get_conn(db_path)
    
    sql = f"""
        SELECT event_id, ts, severity, event, role, fields_json
        FROM events
        WHERE severity >= ?
        ORDER BY severity DESC, ts DESC
        LIMIT {top_n}
    """
    
    rows = con.execute(sql, [severity_min]).fetchall()
    return [_parse_event_row(row) for row in rows]


# ============================================================================
# TIMELINE TOOLS
# ============================================================================

def get_recovery_timeline(
    db_path: str,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None
) -> List[Dict[str, Any]]:
    """
    Get recovery sequence timeline.
    
    Args:
        db_path: Path to DuckDB database
        start_time: Optional start time filter
        end_time: Optional end time filter
        
    Returns:
        List of recovery events with timestamps and states
    """
    con = get_conn(db_path)
    
    conditions = ["event = 'MasterRecoveryState'"]
    params = []
    
    if start_time:
        conditions.append("ts >= ?")
        params.append(start_time)
    if end_time:
        conditions.append("ts <= ?")
        params.append(end_time)
    
    where_clause = "WHERE " + " AND ".join(conditions)
    
    sql = f"""
        SELECT event_id, ts, severity, event, role, fields_json
        FROM events
        {where_clause}
        ORDER BY ts ASC
    """
    
    rows = con.execute(sql, params).fetchall()
    
    timeline = []
    for row in rows:
        rid, ts, sev, event, role, fields = row
        fdict = _ensure_dict(fields)
        
        timeline.append({
            "timestamp": ts.isoformat() if isinstance(ts, datetime) else str(ts),
            "recovery_state": fdict.get("State", fdict.get("state", "unknown")),
            "role": role,
            "fields": fdict
        })
    
    return timeline


def get_coordinator_changes(
    db_path: str,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None
) -> List[Dict[str, Any]]:
    """
    Get coordinator change events (detect flapping).
    
    Args:
        db_path: Path to DuckDB database
        start_time: Optional start time filter
        end_time: Optional end time filter
        
    Returns:
        List of coordinator change events
    """
    con = get_conn(db_path)
    
    conditions = ["event LIKE '%Coordinator%' OR event LIKE '%coordinator%'"]
    params = []
    
    if start_time:
        conditions.append("ts >= ?")
        params.append(start_time)
    if end_time:
        conditions.append("ts <= ?")
        params.append(end_time)
    
    where_clause = "WHERE " + " AND ".join(conditions)
    
    sql = f"""
        SELECT event_id, ts, severity, event, role, fields_json
        FROM events
        {where_clause}
        ORDER BY ts ASC
    """
    
    rows = con.execute(sql, params).fetchall()
    
    changes = []
    for row in rows:
        rid, ts, sev, event, role, fields = row
        fdict = _ensure_dict(fields)
        
        changes.append({
            "timestamp": ts.isoformat() if isinstance(ts, datetime) else str(ts),
            "event_type": event,
            "role": role,
            "fields": fdict
        })
    
    return changes


def get_role_recruitments(
    db_path: str,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None
) -> List[Dict[str, Any]]:
    """
    Get role recruitment events (detect churn/instability).
    
    Args:
        db_path: Path to DuckDB database
        start_time: Optional start time filter
        end_time: Optional end time filter
        
    Returns:
        List of role recruitment events
    """
    con = get_conn(db_path)
    
    conditions = ["event LIKE '%Recruit%' OR event LIKE '%recruit%'"]
    params = []
    
    if start_time:
        conditions.append("ts >= ?")
        params.append(start_time)
    if end_time:
        conditions.append("ts <= ?")
        params.append(end_time)
    
    where_clause = "WHERE " + " AND ".join(conditions)
    
    sql = f"""
        SELECT event_id, ts, severity, event, role, fields_json
        FROM events
        {where_clause}
        ORDER BY ts ASC
    """
    
    rows = con.execute(sql, params).fetchall()
    
    recruitments = []
    for row in rows:
        rid, ts, sev, event, role, fields = row
        fdict = _ensure_dict(fields)
        
        recruitments.append({
            "timestamp": ts.isoformat() if isinstance(ts, datetime) else str(ts),
            "event_type": event,
            "role": role,
            "fields": fdict
        })
    
    return recruitments


def get_event_histogram(
    db_path: str,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    event_types: Optional[List[str]] = None
) -> Dict[str, int]:
    """
    Get event type histogram (detect spikes/noise).
    
    Args:
        db_path: Path to DuckDB database
        start_time: Optional start time filter
        end_time: Optional end time filter
        event_types: Optional list of event types to include (None = all)
        
    Returns:
        Dictionary mapping event type to count
    """
    con = get_conn(db_path)
    
    conditions = []
    params = []
    
    if start_time:
        conditions.append("ts >= ?")
        params.append(start_time)
    if end_time:
        conditions.append("ts <= ?")
        params.append(end_time)
    
    if event_types:
        event_conditions = []
        for et in event_types:
            if '%' in et:
                event_conditions.append("event LIKE ?")
                params.append(et)
            else:
                event_conditions.append("event = ?")
                params.append(et)
        if event_conditions:
            conditions.append(f"({' OR '.join(event_conditions)})")
    
    where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""
    
    sql = f"""
        SELECT event, COUNT(*) as count
        FROM events
        {where_clause}
        GROUP BY event
        ORDER BY count DESC
    """
    
    rows = con.execute(sql, params).fetchall()
    return {row[0]: row[1] for row in rows}


def get_lag_series(
    db_path: str,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    limit: int = 1000
) -> List[Dict[str, Any]]:
    """
    Get lag time series from StorageMetrics (detect storage pressure).
    
    Args:
        db_path: Path to DuckDB database
        start_time: Optional start time filter
        end_time: Optional end time filter
        limit: Maximum number of data points
        
    Returns:
        List of lag measurements with timestamps
    """
    con = get_conn(db_path)
    
    conditions = ["event = 'StorageMetrics'"]
    params = []
    
    if start_time:
        conditions.append("ts >= ?")
        params.append(start_time)
    if end_time:
        conditions.append("ts <= ?")
        params.append(end_time)
    
    where_clause = "WHERE " + " AND ".join(conditions)
    
    sql = f"""
        SELECT event_id, ts, severity, event, role, fields_json
        FROM events
        {where_clause}
        ORDER BY ts ASC
        LIMIT {limit}
    """
    
    rows = con.execute(sql, params).fetchall()
    
    lag_series = []
    for row in rows:
        rid, ts, sev, event, role, fields = row
        fdict = _ensure_dict(fields)
        
        # Extract VersionLag or other lag metrics
        version_lag = fdict.get("VersionLag") or fdict.get("versionLag") or fdict.get("VersionLagValue")
        
        if version_lag is not None:
            try:
                lag_value = float(version_lag)
                lag_series.append({
                    "timestamp": ts.isoformat() if isinstance(ts, datetime) else str(ts),
                    "version_lag": lag_value,
                    "role": role,
                    "all_fields": fdict
                })
            except (ValueError, TypeError):
                continue
    
    return lag_series


# ============================================================================
# DETECTION TOOLS
# ============================================================================

def detect_storage_engine_pressure(
    db_path: str,
    lag_threshold: float = 50000.0
) -> Dict[str, Any]:
    """
    Check for lag spikes indicating storage engine pressure.
    
    Args:
        db_path: Path to DuckDB database
        lag_threshold: VersionLag threshold to consider as pressure
        
    Returns:
        Dictionary with detection results
    """
    lag_series = get_lag_series(db_path)
    
    if not lag_series:
        return {
            "detected": False,
            "reason": "No StorageMetrics events found",
            "max_lag": None,
            "high_lag_events": []
        }
    
    high_lag_events = [e for e in lag_series if e.get("version_lag", 0) > lag_threshold]
    max_lag = max((e.get("version_lag", 0) for e in lag_series), default=0)
    
    return {
        "detected": len(high_lag_events) > 0,
        "max_lag": max_lag,
        "threshold": lag_threshold,
        "high_lag_events": high_lag_events[:20],  # Top 20
        "total_high_lag_count": len(high_lag_events)
    }


def detect_ratekeeper_throttling(
    db_path: str
) -> Dict[str, Any]:
    """
    Check for Ratekeeper throttling events.
    
    Args:
        db_path: Path to DuckDB database
        
    Returns:
        Dictionary with throttling detection results
    """
    con = get_conn(db_path)
    
    sql = """
        SELECT event_id, ts, severity, event, role, fields_json
        FROM events
        WHERE event LIKE '%RkUpdate%' OR event LIKE '%Ratekeeper%' OR event LIKE '%Throttle%'
        ORDER BY ts ASC
    """
    
    rows = con.execute(sql).fetchall()
    
    throttling_events = []
    for row in rows:
        rid, ts, sev, event, role, fields = row
        fdict = _ensure_dict(fields)
        
        # Check for throttling indicators in fields
        throttled = False
        reason = None
        
        for key, value in fdict.items():
            key_lower = str(key).lower()
            if 'throttle' in key_lower or 'throttled' in key_lower:
                throttled = True
                reason = f"{key}: {value}"
                break
        
        if throttled or 'throttle' in str(event).lower():
            throttling_events.append({
                "timestamp": ts.isoformat() if isinstance(ts, datetime) else str(ts),
                "event_type": event,
                "role": role,
                "reason": reason,
                "fields": fdict
            })
    
    return {
        "detected": len(throttling_events) > 0,
        "throttling_events": throttling_events,
        "count": len(throttling_events)
    }


def detect_missing_tlogs(
    db_path: str
) -> Dict[str, Any]:
    """
    Find missing TLog events (critical failure indicator).
    
    Args:
        db_path: Path to DuckDB database
        
    Returns:
        Dictionary with missing TLog detection results
    """
    con = get_conn(db_path)
    
    sql = """
        SELECT event_id, ts, severity, event, role, fields_json
        FROM events
        WHERE (event LIKE '%TLog%' AND (event LIKE '%Missing%' OR event LIKE '%Failed%' OR event LIKE '%Error%'))
           OR (event LIKE '%Tlog%' AND (event LIKE '%Missing%' OR event LIKE '%Failed%' OR event LIKE '%Error%'))
        ORDER BY ts ASC
    """
    
    rows = con.execute(sql).fetchall()
    
    missing_tlog_events = []
    for row in rows:
        rid, ts, sev, event, role, fields = row
        fdict = _ensure_dict(fields)
        
        missing_tlog_events.append({
            "timestamp": ts.isoformat() if isinstance(ts, datetime) else str(ts),
            "event_type": event,
            "severity": sev,
            "role": role,
            "fields": fdict
        })
    
    return {
        "detected": len(missing_tlog_events) > 0,
        "missing_tlog_events": missing_tlog_events,
        "count": len(missing_tlog_events)
    }


def detect_recovery_loop(
    db_path: str,
    time_window_seconds: int = 60
) -> Dict[str, Any]:
    """
    Check for repeated recovery events (cluster instability).
    
    Args:
        db_path: Path to DuckDB database
        time_window_seconds: Time window to check for repeated recoveries
        
    Returns:
        Dictionary with recovery loop detection results
    """
    timeline = get_recovery_timeline(db_path)
    
    if len(timeline) < 3:
        return {
            "detected": False,
            "reason": "Insufficient recovery events",
            "recovery_count": len(timeline)
        }
    
    # Group recoveries by time window
    recovery_groups = []
    current_group = [timeline[0]]
    
    for event in timeline[1:]:
        prev_time = datetime.fromisoformat(current_group[-1]["timestamp"].replace('Z', '+00:00'))
        curr_time = datetime.fromisoformat(event["timestamp"].replace('Z', '+00:00'))
        
        if (curr_time - prev_time).total_seconds() <= time_window_seconds:
            current_group.append(event)
        else:
            if len(current_group) >= 3:
                recovery_groups.append(current_group)
            current_group = [event]
    
    if len(current_group) >= 3:
        recovery_groups.append(current_group)
    
    return {
        "detected": len(recovery_groups) > 0,
        "recovery_groups": recovery_groups,
        "total_recoveries": len(timeline),
        "loop_count": len(recovery_groups)
    }


def detect_coordination_loss(
    db_path: str
) -> Dict[str, Any]:
    """
    Detect coordinator flapping/loss events.
    
    Args:
        db_path: Path to DuckDB database
        
    Returns:
        Dictionary with coordination loss detection results
    """
    changes = get_coordinator_changes(db_path)
    
    # Look for patterns indicating loss
    loss_indicators = []
    for change in changes:
        event_type = change.get("event_type", "").lower()
        fields = change.get("fields", {})
        
        if any(keyword in event_type for keyword in ["fail", "error", "lost", "unreachable", "timeout"]):
            loss_indicators.append(change)
        elif any(keyword in str(fields).lower() for keyword in ["fail", "error", "lost", "unreachable"]):
            loss_indicators.append(change)
    
    return {
        "detected": len(loss_indicators) > 0,
        "coordination_changes": len(changes),
        "loss_indicators": loss_indicators,
        "count": len(loss_indicators)
    }


def detect_version_skew(
    db_path: str
) -> Dict[str, Any]:
    """
    Detect version mismatch events (bug indicator).
    
    Args:
        db_path: Path to DuckDB database
        
    Returns:
        Dictionary with version skew detection results
    """
    con = get_conn(db_path)
    
    sql = """
        SELECT event_id, ts, severity, event, role, fields_json
        FROM events
        WHERE event LIKE '%Version%' AND (event LIKE '%Mismatch%' OR event LIKE '%Skew%' OR event LIKE '%Conflict%')
        ORDER BY ts ASC
    """
    
    rows = con.execute(sql).fetchall()
    
    version_events = []
    for row in rows:
        rid, ts, sev, event, role, fields = row
        fdict = _ensure_dict(fields)
        
        version_events.append({
            "timestamp": ts.isoformat() if isinstance(ts, datetime) else str(ts),
            "event_type": event,
            "severity": sev,
            "role": role,
            "fields": fdict
        })
    
    return {
        "detected": len(version_events) > 0,
        "version_events": version_events,
        "count": len(version_events)
    }


def detect_process_class_mismatch(
    db_path: str
) -> Dict[str, Any]:
    """
    Detect role misconfiguration (load imbalance indicator).
    
    Args:
        db_path: Path to DuckDB database
        
    Returns:
        Dictionary with process class mismatch detection results
    """
    con = get_conn(db_path)
    
    sql = """
        SELECT event_id, ts, severity, event, role, fields_json
        FROM events
        WHERE event LIKE '%ProcessClass%' OR event LIKE '%ClassMismatch%' OR event LIKE '%Misconfig%'
        ORDER BY ts ASC
    """
    
    rows = con.execute(sql).fetchall()
    
    mismatch_events = []
    for row in rows:
        rid, ts, sev, event, role, fields = row
        fdict = _ensure_dict(fields)
        
        mismatch_events.append({
            "timestamp": ts.isoformat() if isinstance(ts, datetime) else str(ts),
            "event_type": event,
            "severity": sev,
            "role": role,
            "fields": fdict
        })
    
    return {
        "detected": len(mismatch_events) > 0,
        "mismatch_events": mismatch_events,
        "count": len(mismatch_events)
    }


# ============================================================================
# METRICS TOOLS
# ============================================================================

def get_commit_latency_stats(
    db_path: str,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None
) -> Dict[str, Any]:
    """
    Get commit latency distribution statistics.
    
    Args:
        db_path: Path to DuckDB database
        start_time: Optional start time filter
        end_time: Optional end time filter
        
    Returns:
        Dictionary with latency statistics
    """
    con = get_conn(db_path)
    
    conditions = ["event LIKE '%CommitLatency%' OR event LIKE '%LatencyMetrics%'"]
    params = []
    
    if start_time:
        conditions.append("ts >= ?")
        params.append(start_time)
    if end_time:
        conditions.append("ts <= ?")
        params.append(end_time)
    
    where_clause = "WHERE " + " AND ".join(conditions)
    
    sql = f"""
        SELECT event_id, ts, severity, event, role, fields_json
        FROM events
        {where_clause}
        ORDER BY ts ASC
    """
    
    rows = con.execute(sql, params).fetchall()
    
    latencies = []
    for row in rows:
        rid, ts, sev, event, role, fields = row
        fdict = _ensure_dict(fields)
        
        # Extract latency values
        for key, value in fdict.items():
            if 'latency' in str(key).lower() or 'commit' in str(key).lower():
                try:
                    lat_val = float(value)
                    latencies.append(lat_val)
                except (ValueError, TypeError):
                    continue
    
    if not latencies:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "p95": None,
            "p99": None,
            "max": None
        }
    
    latencies.sort()
    n = len(latencies)
    
    return {
        "count": n,
        "mean": sum(latencies) / n,
        "median": latencies[n // 2],
        "p95": latencies[int(n * 0.95)] if n > 0 else None,
        "p99": latencies[int(n * 0.99)] if n > 0 else None,
        "max": max(latencies)
    }


def get_queue_depth_stats(
    db_path: str,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None
) -> Dict[str, Any]:
    """
    Get queue depth statistics (backpressure indicator).
    
    Args:
        db_path: Path to DuckDB database
        start_time: Optional start time filter
        end_time: Optional end time filter
        
    Returns:
        Dictionary with queue depth statistics
    """
    con = get_conn(db_path)
    
    conditions = ["event LIKE '%Queue%' OR event LIKE '%Metrics%'"]
    params = []
    
    if start_time:
        conditions.append("ts >= ?")
        params.append(start_time)
    if end_time:
        conditions.append("ts <= ?")
        params.append(end_time)
    
    where_clause = "WHERE " + " AND ".join(conditions)
    
    sql = f"""
        SELECT event_id, ts, severity, event, role, fields_json
        FROM events
        {where_clause}
        ORDER BY ts ASC
    """
    
    rows = con.execute(sql, params).fetchall()
    
    queue_depths = []
    for row in rows:
        rid, ts, sev, event, role, fields = row
        fdict = _ensure_dict(fields)
        
        # Extract queue depth values
        for key, value in fdict.items():
            if 'queue' in str(key).lower() or 'depth' in str(key).lower():
                try:
                    depth_val = float(value)
                    queue_depths.append(depth_val)
                except (ValueError, TypeError):
                    continue
    
    if not queue_depths:
        return {
            "count": 0,
            "mean": None,
            "max": None,
            "high_queue_events": []
        }
    
    queue_depths.sort(reverse=True)
    high_threshold = 1000
    high_queues = [d for d in queue_depths if d > high_threshold]
    
    return {
        "count": len(queue_depths),
        "mean": sum(queue_depths) / len(queue_depths),
        "max": max(queue_depths),
        "high_queue_count": len(high_queues),
        "high_queue_threshold": high_threshold
    }


def get_durable_lag_stats(
    db_path: str,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None
) -> Dict[str, Any]:
    """
    Get durable lag statistics (durability risk indicator).
    
    Args:
        db_path: Path to DuckDB database
        start_time: Optional start time filter
        end_time: Optional end time filter
        
    Returns:
        Dictionary with durable lag statistics
    """
    con = get_conn(db_path)
    
    conditions = ["event LIKE '%Durable%' OR event LIKE '%StorageMetrics%'"]
    params = []
    
    if start_time:
        conditions.append("ts >= ?")
        params.append(start_time)
    if end_time:
        conditions.append("ts <= ?")
        params.append(end_time)
    
    where_clause = "WHERE " + " AND ".join(conditions)
    
    sql = f"""
        SELECT event_id, ts, severity, event, role, fields_json
        FROM events
        {where_clause}
        ORDER BY ts ASC
    """
    
    rows = con.execute(sql, params).fetchall()
    
    lag_values = []
    for row in rows:
        rid, ts, sev, event, role, fields = row
        fdict = _ensure_dict(fields)
        
        # Extract durable lag values
        for key, value in fdict.items():
            if 'lag' in str(key).lower() and 'durable' in str(key).lower():
                try:
                    lag_val = float(value)
                    lag_values.append(lag_val)
                except (ValueError, TypeError):
                    continue
    
    if not lag_values:
        return {
            "count": 0,
            "mean": None,
            "max": None
        }
    
    return {
        "count": len(lag_values),
        "mean": sum(lag_values) / len(lag_values),
        "max": max(lag_values)
    }


def get_tlog_pop_stats(
    db_path: str,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None
) -> Dict[str, Any]:
    """
    Get TLog pop statistics (TLog diagnosis).
    
    Args:
        db_path: Path to DuckDB database
        start_time: Optional start time filter
        end_time: Optional end time filter
        
    Returns:
        Dictionary with TLog pop statistics
    """
    con = get_conn(db_path)
    
    conditions = ["event LIKE '%TLog%' OR event LIKE '%Tlog%'"]
    params = []
    
    if start_time:
        conditions.append("ts >= ?")
        params.append(start_time)
    if end_time:
        conditions.append("ts <= ?")
        params.append(end_time)
    
    where_clause = "WHERE " + " AND ".join(conditions)
    
    sql = f"""
        SELECT event_id, ts, severity, event, role, fields_json
        FROM events
        {where_clause}
        ORDER BY ts ASC
    """
    
    rows = con.execute(sql, params).fetchall()
    
    pop_values = []
    for row in rows:
        rid, ts, sev, event, role, fields = row
        fdict = _ensure_dict(fields)
        
        # Extract TLog pop/version values
        for key, value in fdict.items():
            if 'pop' in str(key).lower() or 'version' in str(key).lower():
                try:
                    pop_val = float(value)
                    pop_values.append(pop_val)
                except (ValueError, TypeError):
                    continue
    
    if not pop_values:
        return {
            "count": 0,
            "mean": None,
            "min": None,
            "max": None
        }
    
    return {
        "count": len(pop_values),
        "mean": sum(pop_values) / len(pop_values),
        "min": min(pop_values),
        "max": max(pop_values)
    }


def get_hot_shard_signals(
    db_path: str,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None
) -> List[Dict[str, Any]]:
    """
    Get shard hotspot signals (partition imbalance indicator).
    
    Args:
        db_path: Path to DuckDB database
        start_time: Optional start time filter
        end_time: Optional end time filter
        
    Returns:
        List of hot shard signals
    """
    con = get_conn(db_path)
    
    conditions = ["event LIKE '%Shard%' OR event LIKE '%Hot%' OR event LIKE '%StorageMetrics%'"]
    params = []
    
    if start_time:
        conditions.append("ts >= ?")
        params.append(start_time)
    if end_time:
        conditions.append("ts <= ?")
        params.append(end_time)
    
    where_clause = "WHERE " + " AND ".join(conditions)
    
    sql = f"""
        SELECT event_id, ts, severity, event, role, fields_json
        FROM events
        {where_clause}
        ORDER BY ts DESC
        LIMIT 500
    """
    
    rows = con.execute(sql, params).fetchall()
    
    hot_shards = []
    for row in rows:
        rid, ts, sev, event, role, fields = row
        fdict = _ensure_dict(fields)
        
        # Look for indicators of hot shards
        for key, value in fdict.items():
            key_lower = str(key).lower()
            if any(indicator in key_lower for indicator in ['hot', 'spike', 'overload', 'saturated']):
                try:
                    val = float(value)
                    if val > 0:
                        hot_shards.append({
                            "timestamp": ts.isoformat() if isinstance(ts, datetime) else str(ts),
                            "event_type": event,
                            "role": role,
                            "indicator": key,
                            "value": val,
                            "fields": fdict
                        })
                        break
                except (ValueError, TypeError):
                    continue
    
    return hot_shards


# ============================================================================
# CONFIG/HEALTH TOOLS
# ============================================================================

def get_cluster_config_summary(
    db_path: str
) -> Dict[str, Any]:
    """
    Get cluster configuration context from events.
    
    Args:
        db_path: Path to DuckDB database
        
    Returns:
        Dictionary with cluster configuration summary
    """
    con = get_conn(db_path)
    
    # Look for configuration-related events
    sql = """
        SELECT event_id, ts, severity, event, role, fields_json
        FROM events
        WHERE event LIKE '%Config%' OR event LIKE '%Version%' OR event LIKE '%Coordinator%'
        ORDER BY ts DESC
        LIMIT 100
    """
    
    rows = con.execute(sql).fetchall()
    
    config_info = {
        "versions": set(),
        "coordinators": set(),
        "roles": set(),
        "config_events": []
    }
    
    for row in rows:
        rid, ts, sev, event, role, fields = row
        fdict = _ensure_dict(fields)
        
        if role:
            config_info["roles"].add(role)
        
        # Extract version info
        for key, value in fdict.items():
            if 'version' in str(key).lower():
                config_info["versions"].add(str(value))
        
        config_info["config_events"].append({
            "timestamp": ts.isoformat() if isinstance(ts, datetime) else str(ts),
            "event_type": event,
            "role": role,
            "fields": fdict
        })
    
    return {
        "versions": list(config_info["versions"]),
        "roles": list(config_info["roles"]),
        "config_events": config_info["config_events"][:20]  # Top 20
    }


def check_quorum_validity(
    db_path: str
) -> Dict[str, Any]:
    """
    Check coordinator quorum validity (availability check).
    
    Args:
        db_path: Path to DuckDB database
        
    Returns:
        Dictionary with quorum validity check results
    """
    changes = get_coordinator_changes(db_path)
    
    # Look for quorum-related events
    con = get_conn(db_path)
    
    sql = """
        SELECT event_id, ts, severity, event, role, fields_json
        FROM events
        WHERE event LIKE '%Quorum%' OR event LIKE '%quorum%'
        ORDER BY ts DESC
        LIMIT 50
    """
    
    rows = con.execute(sql).fetchall()
    
    quorum_events = []
    for row in rows:
        rid, ts, sev, event, role, fields = row
        fdict = _ensure_dict(fields)
        
        quorum_events.append({
            "timestamp": ts.isoformat() if isinstance(ts, datetime) else str(ts),
            "event_type": event,
            "severity": sev,
            "role": role,
            "fields": fdict
        })
    
    # Check for loss of quorum indicators
    loss_indicators = [e for e in quorum_events if 
                      any(keyword in str(e.get("event_type", "")).lower() 
                          for keyword in ["lost", "fail", "unreachable", "timeout"])]
    
    return {
        "valid": len(loss_indicators) == 0,
        "quorum_events": len(quorum_events),
        "loss_indicators": loss_indicators,
        "coordinator_changes": len(changes)
    }


def check_redundancy_mode(
    db_path: str
) -> Dict[str, Any]:
    """
    Check redundancy mode validation (correctness check).
    
    Args:
        db_path: Path to DuckDB database
        
    Returns:
        Dictionary with redundancy mode check results
    """
    con = get_conn(db_path)
    
    sql = """
        SELECT event_id, ts, severity, event, role, fields_json
        FROM events
        WHERE event LIKE '%Redundancy%' OR event LIKE '%redundancy%'
        ORDER BY ts DESC
        LIMIT 50
    """
    
    rows = con.execute(sql).fetchall()
    
    redundancy_events = []
    for row in rows:
        rid, ts, sev, event, role, fields = row
        fdict = _ensure_dict(fields)
        
        redundancy_events.append({
            "timestamp": ts.isoformat() if isinstance(ts, datetime) else str(ts),
            "event_type": event,
            "severity": sev,
            "role": role,
            "fields": fdict
        })
    
    # Extract redundancy mode from events
    modes = set()
    for event in redundancy_events:
        fields = event.get("fields", {})
        for key, value in fields.items():
            if 'redundancy' in str(key).lower() or 'mode' in str(key).lower():
                modes.add(str(value))
    
    return {
        "redundancy_events": len(redundancy_events),
        "modes_detected": list(modes),
        "events": redundancy_events[:10]  # Top 10
    }


# ============================================================================
# RAG/SIMILARITY TOOLS
# ============================================================================

def search_similar_incidents(
    db_path: str,
    reference_event_type: str,
    limit: int = 10
) -> List[Dict[str, Any]]:
    """
    Find similar incident cases (pattern reuse).
    
    Args:
        db_path: Path to DuckDB database
        reference_event_type: Event type to find similar cases for
        limit: Maximum number of similar incidents to return
        
    Returns:
        List of similar incident events
    """
    # Find events of similar type
    con = get_conn(db_path)
    
    # Use LIKE pattern matching for similarity
    pattern = f"%{reference_event_type}%"
    
    sql = f"""
        SELECT event_id, ts, severity, event, role, fields_json
        FROM events
        WHERE event LIKE ?
        ORDER BY ts DESC
        LIMIT {limit}
    """
    
    rows = con.execute(sql, [pattern]).fetchall()
    
    similar_incidents = []
    for row in rows:
        rid, ts, sev, event, role, fields = row
        fdict = _ensure_dict(fields)
        
        similar_incidents.append({
            "timestamp": ts.isoformat() if isinstance(ts, datetime) else str(ts),
            "event_type": event,
            "severity": sev,
            "role": role,
            "fields": fdict
        })
    
    return similar_incidents


def get_context_window(
    db_path: str,
    around_time: datetime,
    window_seconds: int = 30,
    limit: int = 200
) -> List[EventModel]:
    """
    Get raw event window around a specific time (deep inspection).
    
    Args:
        db_path: Path to DuckDB database
        around_time: Central timestamp
        window_seconds: Time window in seconds before and after
        limit: Maximum number of events to return
        
    Returns:
        List of EventModel instances in the time window
    """
    start_time = around_time - timedelta(seconds=window_seconds)
    end_time = around_time + timedelta(seconds=window_seconds)
    
    return get_events(
        db_path,
        start_time=start_time,
        end_time=end_time,
        limit=limit
    )


# ============================================================================
# GLOBAL SCAN TOOLS (Prevent Tunnel Vision)
# ============================================================================

def global_severity_scan(
    db_path: str,
    min_severity: int = 30,
    limit: int = 500
) -> List[EventModel]:
    """
    Global severity scan over the full table (mandatory first step).
    
    Args:
        db_path: Path to DuckDB database
        min_severity: Minimum severity threshold (default 30)
        limit: Maximum number of events to return
        
    Returns:
        List of EventModel instances with severity >= min_severity
    """
    return get_events(
        db_path,
        severity_min=min_severity,
        limit=limit
    )


def global_severity_scan_warnings(
    db_path: str,
    warning_types: List[str] = None,
    limit: int = 500
) -> List[EventModel]:
    """
    Scan for Severity 20 warnings of well-known types.
    
    Args:
        db_path: Path to DuckDB database
        warning_types: List of event types to check (default: common ones)
        limit: Maximum number of events to return
        
    Returns:
        List of EventModel instances with Severity 20 warnings
    """
    if warning_types is None:
        warning_types = ["SlowSSLoop", "Disk", "StorageServerDurabilityLag", "Ratekeeper"]
    
    events = []
    for event_type in warning_types:
        found = get_events(
            db_path,
            severity_min=20,
            severity_max=20,
            event_type=f"%{event_type}%",
            limit=limit
        )
        events.extend(found)
    
    return events[:limit]


def global_event_histogram(
    db_path: str,
    limit: int = 50
) -> Dict[str, int]:
    """
    Global histogram of event types across entire dataset.
    
    Args:
        db_path: Path to DuckDB database
        limit: Maximum number of event types to return (not used, kept for API compatibility)
        
    Returns:
        Dictionary mapping event type to count
    """
    histogram = get_event_histogram(db_path)
    # Limit the results if needed
    if limit and len(histogram) > limit:
        # Sort by count descending and take top N
        sorted_items = sorted(histogram.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_items[:limit])
    return histogram


def global_severity_counts(
    db_path: str
) -> Dict[int, int]:
    """
    Count events per severity level across entire dataset.
    
    Args:
        db_path: Path to DuckDB database
        
    Returns:
        Dictionary mapping severity level to count
    """
    con = get_conn(db_path)
    
    sql = """
        SELECT severity, COUNT(*) as count
        FROM events
        GROUP BY severity
        ORDER BY severity DESC
    """
    
    rows = con.execute(sql).fetchall()
    return {row[0]: row[1] for row in rows}


def bucket_max_severity(
    db_path: str,
    bucket_seconds: int = 300
) -> List[Dict[str, Any]]:
    """
    Max severity per time bucket across the whole span.
    
    Args:
        db_path: Path to DuckDB database
        bucket_seconds: Size of time bucket in seconds (default: 300 = 5 minutes)
        
    Returns:
        List of dictionaries with bucket, max_severity, and event_count
    """
    con = get_conn(db_path)
    
    sql = f"""
        SELECT 
            CAST(FLOOR(EXTRACT(EPOCH FROM ts) / {bucket_seconds}) * {bucket_seconds} AS BIGINT) AS bucket_start,
            MAX(severity) AS max_severity,
            COUNT(*) AS event_count
        FROM events
        GROUP BY bucket_start
        HAVING max_severity >= 20
        ORDER BY max_severity DESC, bucket_start ASC
        LIMIT 100
    """
    
    rows = con.execute(sql).fetchall()
    
    buckets = []
    for row in rows:
        bucket_start_epoch, max_sev, count = row
        bucket_start = datetime.fromtimestamp(bucket_start_epoch)
        
        buckets.append({
            "bucket_start": bucket_start.isoformat(),
            "bucket_start_epoch": bucket_start_epoch,
            "max_severity": max_sev,
            "event_count": count
        })
    
    return buckets


def bucket_event_summary(
    db_path: str,
    bucket_seconds: int = 600,
    min_count: int = 10
) -> List[Dict[str, Any]]:
    """
    Top event types per time bucket (coarse view).
    
    Args:
        db_path: Path to DuckDB database
        bucket_seconds: Size of time bucket in seconds (default: 600 = 10 minutes)
        min_count: Minimum count per bucket to include
        
    Returns:
        List of dictionaries with bucket, event_type, and count
    """
    con = get_conn(db_path)
    
    sql = f"""
        SELECT 
            CAST(FLOOR(EXTRACT(EPOCH FROM ts) / {bucket_seconds}) * {bucket_seconds} AS BIGINT) AS bucket_start,
            event AS event_type,
            COUNT(*) AS count
        FROM events
        GROUP BY bucket_start, event_type
        HAVING count >= {min_count}
        ORDER BY bucket_start ASC, count DESC
        LIMIT 500
    """
    
    rows = con.execute(sql).fetchall()
    
    summaries = []
    for row in rows:
        bucket_start_epoch, event_type, count = row
        bucket_start = datetime.fromtimestamp(bucket_start_epoch)
        
        summaries.append({
            "bucket_start": bucket_start.isoformat(),
            "event_type": event_type,
            "count": count
        })
    
    return summaries


def get_time_span(
    db_path: str
) -> Dict[str, Any]:
    """
    Get earliest and latest timestamps in the database.
    
    Args:
        db_path: Path to DuckDB database
        
    Returns:
        Dictionary with earliest, latest, and time_span_seconds
    """
    con = get_conn(db_path)
    
    sql = """
        SELECT 
            MIN(ts) AS earliest,
            MAX(ts) AS latest,
            EXTRACT(EPOCH FROM (MAX(ts) - MIN(ts))) AS span_seconds
        FROM events
    """
    
    row = con.execute(sql).fetchone()
    
    if row and row[0]:
        earliest, latest, span_seconds = row
        return {
            "earliest": earliest.isoformat() if isinstance(earliest, datetime) else str(earliest),
            "latest": latest.isoformat() if isinstance(latest, datetime) else str(latest),
            "span_seconds": float(span_seconds) if span_seconds else 0
        }
    
    return {
        "earliest": None,
        "latest": None,
        "span_seconds": 0
    }


def get_global_summary(
    db_path: str
) -> Dict[str, Any]:
    """
    Generate a global summary required before finalizing.
    
    Args:
        db_path: Path to DuckDB database
        
    Returns:
        Dictionary with max severity, time span, recovery count, and top event types
    """
    con = get_conn(db_path)
    
    # Get max severity
    max_severity_row = con.execute("SELECT MAX(severity) FROM events").fetchone()
    max_severity = max_severity_row[0] if max_severity_row[0] else 0
    
    # Get time span
    time_span = get_time_span(db_path)
    
    # Count recoveries
    recovery_count_row = con.execute(
        "SELECT COUNT(*) FROM events WHERE event LIKE '%Recovery%' OR event = 'MasterRecoveryState'"
    ).fetchone()
    recovery_count = recovery_count_row[0] if recovery_count_row else 0
    
    # Top 5 event types
    top_types = global_event_histogram(db_path, limit=5)
    
    return {
        "max_severity": max_severity,
        "time_span": time_span,
        "recovery_count": recovery_count,
        "top_5_event_types": top_types
    }


def get_uncovered_buckets(
    db_path: str,
    inspected_buckets: List[int],
    bucket_seconds: int = 600,
    min_severity: int = 20
) -> List[Dict[str, Any]]:
    """
    Find buckets with severity >= min_severity that haven't been inspected yet.
    
    Args:
        db_path: Path to DuckDB database
        inspected_buckets: List of bucket_start_epoch values already inspected
        bucket_seconds: Size of time bucket in seconds
        min_severity: Minimum severity threshold
        
    Returns:
        List of uncovered buckets that need inspection
    """
    con = get_conn(db_path)
    
    # Create a list of inspected bucket strings for SQL
    if not inspected_buckets:
        inspected_clause = "FALSE"  # No inspected buckets
    else:
        inspected_str = ",".join(str(b) for b in inspected_buckets)
        inspected_clause = f"bucket_start IN ({inspected_str})"
    
    sql = f"""
        SELECT 
            CAST(FLOOR(EXTRACT(EPOCH FROM ts) / {bucket_seconds}) * {bucket_seconds} AS BIGINT) AS bucket_start,
            MAX(severity) AS max_severity,
            COUNT(*) AS event_count
        FROM events
        GROUP BY bucket_start
        HAVING max_severity >= {min_severity} AND NOT ({inspected_clause})
        ORDER BY max_severity DESC
        LIMIT 20
    """
    
    rows = con.execute(sql).fetchall()
    
    uncovered = []
    for row in rows:
        bucket_start_epoch, max_sev, count = row
        bucket_start = datetime.fromtimestamp(bucket_start_epoch)
        
        uncovered.append({
            "bucket_start": bucket_start.isoformat(),
            "bucket_start_epoch": int(bucket_start_epoch),
            "max_severity": max_sev,
            "event_count": count
        })
    
    return uncovered

