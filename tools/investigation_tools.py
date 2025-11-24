"""
Unified Investigation Tools (Condensed Version)
-----------------------------------------------
Contains:
    - Helpers
    - GlobalScanner
    - HotspotSelector
    - ContextAnalyzer
    - Core Detectors
"""

import json
import uuid
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

from data_transfer_object.event_dto import EventModel
from tools.database import get_conn



# ============================================================
# HELPER FUNCTIONS
# ============================================================

def _ensure_dict(maybe_json):
    if maybe_json is None:
        return {}
    if isinstance(maybe_json, dict):
        return maybe_json
    if isinstance(maybe_json, str):
        try:
            return json.loads(maybe_json)
        except:
            return {"_raw": maybe_json}
    return {}


def _uuid_to_int(u):
    try:
        if isinstance(u, uuid.UUID):
            n = u.int
        else:
            n = uuid.UUID(str(u)).int
        return n & ((1 << 63) - 1)
    except:
        return 0


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


# ============================================================
# 1. GLOBAL SCANNER
# ============================================================

class GlobalScanner:
    """Cluster-wide stats & overview."""

    def __init__(self, db_path):
        self.db_path = db_path
        self.con = get_conn(db_path)

    # -------------------------------------------------------
    # Basic Global Stats
    # -------------------------------------------------------

    def severity_counts(self):
        sql = "SELECT severity, COUNT(*) FROM events GROUP BY severity ORDER BY severity DESC"
        rows = self.con.execute(sql).fetchall()
        return {sev: cnt for sev, cnt in rows}

    def event_histogram(self, limit=50):
        sql = """
            SELECT event, COUNT(*)
            FROM events
            GROUP BY event
            ORDER BY COUNT(*) DESC
        """
        rows = self.con.execute(sql).fetchall()
        return {ev: cnt for ev, cnt in rows[:limit]}

    def time_span(self):
        sql = """
            SELECT MIN(ts), MAX(ts),
                   EXTRACT(EPOCH FROM (MAX(ts) - MIN(ts)))
            FROM events
        """
        earliest, latest, span = self.con.execute(sql).fetchone()
        return {
            "earliest": earliest,
            "latest": latest,
            "duration_seconds": span
        }

    def global_summary(self):
        maxsev = self.con.execute("SELECT MAX(severity) FROM events").fetchone()[0]

        return {
            "max_severity": maxsev,
            "severity_counts": self.severity_counts(),
            "event_histogram": self.event_histogram(10),
            "time_span": self.time_span(),
        }

    def top_events(self, severity_min=40, limit=50):
        sql = """
            SELECT event_id, ts, severity, event, role, fields_json
            FROM events
            WHERE severity >= ?
            ORDER BY severity DESC, ts DESC
            LIMIT ?
        """
        rows = self.con.execute(sql, [severity_min, limit]).fetchall()
        return [_parse_event_row(r) for r in rows]

    def bucket_heatmap(self, bucket_seconds=300, limit=100):
        sql = f"""
            SELECT 
                (FLOOR(EXTRACT(EPOCH FROM ts) / {bucket_seconds}) * {bucket_seconds}) AS bucket,
                MAX(severity), COUNT(*)
            FROM events
            GROUP BY bucket
            ORDER BY MAX(severity) DESC
            LIMIT {limit}
        """
        rows = self.con.execute(sql).fetchall()

        return [
            {
                "bucket_start_epoch": b,
                "bucket_start": datetime.fromtimestamp(b),
                "max_severity": sev,
                "count": cnt,
            }
            for (b, sev, cnt) in rows
        ]

    # -------------------------------------------------------
    # 🔥 Rollback Detector (ClogWithRollbacks / Epoch Reset)
    # -------------------------------------------------------

    def rollback_analysis(self):
        """
        Detects rollback-like behavior:
        • Version drops (CommittedVersion or DurableVersion decreasing)
        • Version resets (large → small)
        • RecoveryVersion drop signals
        """

        return {
            "detected": False,
            "version_drops": self._detect_version_drops(),
            "version_resets": self._detect_version_resets(),
            "recovery_resets": self._detect_recovery_resets()
        } | self._rollback_status()

    # -------------------------------------------------------
    # Internal Rollback Helpers
    # -------------------------------------------------------

    def _detect_version_drops(self):
        sql = """
            SELECT 
                ts,
                CAST(json_extract(fields_json, '$.CommittedVersion') AS BIGINT) AS cv,
                CAST(json_extract(fields_json, '$.DurableVersion') AS BIGINT) AS dv,
                event_id,
                event
            FROM events
            WHERE 
                json_extract(fields_json, '$.CommittedVersion') IS NOT NULL
                OR json_extract(fields_json, '$.DurableVersion') IS NOT NULL
            ORDER BY ts
        """

        rows = self.con.execute(sql).fetchall()

        drops = []
        prev_cv = None
        prev_dv = None

        for ts, cv, dv, eid, ev in rows:

            if prev_cv is not None and cv is not None and cv < prev_cv:
                drops.append({
                    "ts": ts,
                    "event_id": eid,
                    "event": ev,
                    "type": "CommittedVersionDrop",
                    "drop_amount": int(prev_cv - cv),
                    "prev": int(prev_cv),
                    "now": int(cv),
                })

            if prev_dv is not None and dv is not None and dv < prev_dv:
                drops.append({
                    "ts": ts,
                    "event_id": eid,
                    "event": ev,
                    "type": "DurableVersionDrop",
                    "drop_amount": int(prev_dv - dv),
                    "prev": int(prev_dv),
                    "now": int(dv),
                })

            if cv is not None:
                prev_cv = cv
            if dv is not None:
                prev_dv = dv

        return drops

    def _detect_version_resets(self):
        sql = """
            SELECT 
                ts,
                CAST(json_extract(fields_json, '$.CommittedVersion') AS BIGINT) AS cv,
                event_id,
                event
            FROM events
            WHERE json_extract(fields_json, '$.CommittedVersion') IS NOT NULL
            ORDER BY ts
        """

        rows = self.con.execute(sql).fetchall()
        resets = []
        prev = None

        for ts, cv, eid, ev in rows:
            if cv is not None and prev is not None:
                if prev > 1_000_000 and cv < 1_000_000:
                    resets.append({
                        "ts": ts,
                        "event_id": eid,
                        "event": ev,
                        "prev_version": int(prev),
                        "new_version": int(cv),
                    })
            if cv is not None:
                prev = cv

        return resets

    def _detect_recovery_resets(self):
        sql = """
            SELECT 
                ts,
                CAST(json_extract(fields_json, '$.RecoveryVersion') AS BIGINT) AS rv,
                event_id,
                event
            FROM events
            WHERE event = 'RecoveryState'
            ORDER BY ts
        """

        rows = self.con.execute(sql).fetchall()
        resets = []
        prev = None

        for ts, rv, eid, ev in rows:
            if rv is not None and prev is not None and rv < prev:
                resets.append({
                    "ts": ts,
                    "event_id": eid,
                    "event": ev,
                    "prev_recovery_version": int(prev),
                    "new_recovery_version": int(rv),
                    "drop_amount": int(prev - rv),
                })
            if rv is not None:
                prev = rv

        return resets

    # Status aggregator — decides if rollback happened
    def _rollback_status(self):
        drops = self._detect_version_drops()
        resets = self._detect_version_resets()
        rec_resets = self._detect_recovery_resets()

        detected = any([drops, resets, rec_resets])

        return {
            "detected": detected,
            "num_drops": len(drops),
            "max_drop": max((d["drop_amount"] for d in drops), default=0),
            "num_resets": len(resets),
            "num_recovery_resets": len(rec_resets),
        }

# ============================================================
# 2. HOTSPOT SELECTOR
# ============================================================

class HotspotSelector:
    """Identify high-severity buckets and missing coverage."""

    def __init__(self, db_path):
        self.db_path = db_path
        self.con = get_conn(db_path)

    def high_severity_buckets(self, min_severity=20, bucket_seconds=600, limit=20):
        sql = f"""
            SELECT 
                (FLOOR(EXTRACT(EPOCH FROM ts) / {bucket_seconds}) * {bucket_seconds}) AS bucket,
                MAX(severity), COUNT(*)
            FROM events
            GROUP BY bucket
            HAVING MAX(severity) >= {min_severity}
            ORDER BY MAX(severity) DESC
            LIMIT {limit}
        """
        rows = self.con.execute(sql).fetchall()

        return [
            {
                "bucket_start_epoch": b,
                "bucket_start": datetime.fromtimestamp(b),
                "max_severity": sev,
                "count": cnt
            }
            for (b, sev, cnt) in rows
        ]

    def get_uncovered(self, inspected_buckets, min_severity=20, bucket_seconds=600):
        if inspected_buckets:
            clause = f"AND bucket NOT IN ({','.join(map(str, inspected_buckets))})"
        else:
            clause = ""

        sql = f"""
            SELECT 
                (FLOOR(EXTRACT(EPOCH FROM ts) / {bucket_seconds}) * {bucket_seconds}) AS bucket,
                MAX(severity), COUNT(*)
            FROM events
            GROUP BY bucket
            HAVING MAX(severity) >= {min_severity} {clause}
            ORDER BY MAX(severity) DESC
            LIMIT 20
        """
        rows = self.con.execute(sql).fetchall()

        return [
            {
                "bucket_start_epoch": b,
                "bucket_start": datetime.fromtimestamp(b),
                "max_severity": sev,
                "count": cnt
            }
            for (b, sev, cnt) in rows
        ]


# ============================================================
# 3. CONTEXT ANALYZER
# ============================================================

class ContextAnalyzer:
    """Local deep-dive investigation tools."""

    def __init__(self, db_path):
        self.db_path = db_path
        self.con = get_conn(db_path)

    def context_window(self, around_time: datetime, window_seconds=30, limit=200):
        start = around_time - timedelta(seconds=window_seconds)
        end = around_time + timedelta(seconds=window_seconds)

        sql = """
            SELECT event_id, ts, severity, event, role, fields_json
            FROM events
            WHERE ts BETWEEN ? AND ?
            ORDER BY ts ASC
            LIMIT ?
        """
        rows = self.con.execute(sql, [start, end, limit]).fetchall()
        return [_parse_event_row(r) for r in rows]

    def similar_events(self, event_type: str, limit=10):
        sql = """
            SELECT event_id, ts, severity, event, role, fields_json
            FROM events
            WHERE event LIKE ?
            ORDER BY ts DESC
            LIMIT ?
        """
        rows = self.con.execute(sql, [f"%{event_type}%", limit]).fetchall()

        return [
            {
                "timestamp": r[1].isoformat() if isinstance(r[1], datetime) else str(r[1]),
                "event_type": r[3],
                "severity": r[2],
                "role": r[4],
                "fields": _ensure_dict(r[5])
            }
            for r in rows
        ]


# ============================================================
# 4. CORE DETECTORS
# ============================================================

class Detectors:
    """Core root-cause signals."""

    def __init__(self, db_path):
        self.db_path = db_path
        self.con = get_conn(db_path)
        self.ctx = ContextAnalyzer(db_path)

    # STORAGE PRESSURE --------------------------------------
    def storage_engine_pressure(self, lag_threshold=50000):
        sql = """
            SELECT ts, fields_json
            FROM events
            WHERE event = 'StorageMetrics'
        """
        rows = self.con.execute(sql).fetchall()

        lags = []
        for ts, f in rows:
            d = _ensure_dict(f)
            lag = d.get("VersionLag") or d.get("versionLag")
            if lag:
                try:
                    lags.append(float(lag))
                except:
                    continue

        if not lags:
            return {"detected": False, "reason": "no storage metrics"}

        high = [l for l in lags if l > lag_threshold]
        return {
            "detected": len(high) > 0,
            "max_lag": max(lags),
            "count_high": len(high),
            "threshold": lag_threshold
        }

    # RATEKEEPER -------------------------------------------
    def ratekeeper_throttling(self):
        sql = """
            SELECT ts, event, fields_json
            FROM events
            WHERE event LIKE '%Ratekeeper%' OR event LIKE '%Throttle%'
        """
        rows = self.con.execute(sql).fetchall()

        out = []
        for ts, ev, f in rows:
            f = _ensure_dict(f)
            if "throttle" in str(ev).lower() or any("throttle" in k.lower() for k in f):
                out.append({"timestamp": ts, "event": ev, "fields": f})

        return {"detected": len(out) > 0, "events": out}

    # MISSING TLOGS ----------------------------------------
    def missing_tlogs(self):
        sql = """
            SELECT ts, event, severity, fields_json
            FROM events
            WHERE event LIKE '%TLog%' AND 
                  (event LIKE '%Missing%' OR event LIKE '%Failed%' OR event LIKE '%Error%')
        """
        rows = self.con.execute(sql).fetchall()
        return {"detected": len(rows) > 0, "events": rows}

    # RECOVERY LOOP ----------------------------------------
    def recovery_loop(self, threshold=3, window_seconds=60):
        timeline = self.ctx.similar_events("MasterRecoveryState", limit=500)
        if len(timeline) < threshold:
            return {"detected": False}

        times = [datetime.fromisoformat(e["timestamp"]) for e in timeline]

        loop_count = 0
        for i in range(len(times) - threshold):
            if (times[i + threshold - 1] - times[i]).total_seconds() <= window_seconds:
                loop_count += 1

        return {"detected": loop_count > 0, "loop_count": loop_count}

    # COORDINATION LOSS ------------------------------------
    def coordination_loss(self):
        sql = """
            SELECT ts, event, fields_json
            FROM events
            WHERE event LIKE '%Coordinator%'
        """
        rows = self.con.execute(sql).fetchall()

        loss = []
        for ts, ev, f in rows:
            s = str(ev).lower()
            f = str(f).lower()
            if any(x in s for x in ["fail", "lost"]) or any(x in f for x in ["fail", "lost"]):
                loss.append({"timestamp": ts, "event": ev})

        return {"detected": len(loss) > 0, "events": loss}
    
