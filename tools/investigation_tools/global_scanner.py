"""GlobalScanner: cluster-wide stats, baselines, and recovery episodes."""

from datetime import datetime, timedelta

from tools.database import get_conn
from .helpers import _parse_event_row, _percentile


class GlobalScanner:
    """Cluster-wide stats & overview."""

    def __init__(self, db_path):
        self.db_path = db_path
        self.con = get_conn(db_path)

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

    # -------------------------------------------------------
    # Metric baselines (mean/std) per role/metric
    # -------------------------------------------------------
    def compute_metric_baselines(self, min_count: int = 5, per_role: bool = True):
        """
        Compute mean/stddev/count for each metric (optionally per role).
        Useful for setting z-score thresholds and knowing which metrics are stable/noisy.
        """
        role_select = "e.role" if per_role else "NULL as role"
        sql = f"""
            SELECT
                m.metric_name,
                {role_select} AS role,
                AVG(CAST(m.metric_value AS DOUBLE)) AS mean,
                STDDEV_SAMP(CAST(m.metric_value AS DOUBLE)) AS stddev,
                COUNT(*) AS cnt,
                MIN(CAST(m.metric_value AS DOUBLE)) AS min,
                MAX(CAST(m.metric_value AS DOUBLE)) AS max
            FROM event_metrics m
            JOIN events e ON m.event_id = e.event_id
            WHERE isfinite(CAST(m.metric_value AS DOUBLE)) AND ABS(CAST(m.metric_value AS DOUBLE)) < 1e308
            GROUP BY m.metric_name, role
            HAVING COUNT(*) >= ?
            ORDER BY m.metric_name, role
        """
        rows = self.con.execute(sql, [min_count]).fetchall()
        baselines = []
        for metric_name, role, mean, stddev, cnt, mn, mx in rows:
            baselines.append({
                "metric": metric_name,
                "role": role,
                "mean": mean,
                "stddev": stddev,
                "count": cnt,
                "min": mn,
                "max": mx,
            })
        return baselines

    def upsert_metric_baselines(self, min_count: int = 20, top_n: int = 500, per_role: bool = True):
        """
        Compute baselines and upsert the top N by count into metric_baselines table.
        """
        try:
            self.con.execute("DROP TABLE IF EXISTS metric_baselines")
            self.con.execute("""
                CREATE TABLE metric_baselines (
                    metric_name VARCHAR,
                    role VARCHAR,
                    mean DOUBLE,
                    stddev DOUBLE,
                    p95 DOUBLE,
                    min DOUBLE,
                    max DOUBLE,
                    count BIGINT,
                    updated_at TIMESTAMP,
                    PRIMARY KEY (metric_name, role)
                )
            """)
        except Exception:
            pass

        baselines = self.compute_metric_baselines(min_count=min_count, per_role=per_role)
        # Exclude non-metric fields from baselines
        excluded = {
            "ThreadID", "ID", "Machine", "Address", "ProcessID", "PID",
            "TraceFile", "TraceFileExtended", "SourceLine"
        }
        baselines = [b for b in baselines if b["metric"] not in excluded]
        # sort by count desc and take top_n
        baselines = sorted(baselines, key=lambda x: x["count"], reverse=True)[:top_n]

        upsert_sql = """
            INSERT INTO metric_baselines (metric_name, role, mean, stddev, p95, min, max, count, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, now())
            ON CONFLICT (metric_name, role) DO UPDATE SET
                mean=excluded.mean,
                stddev=excluded.stddev,
                p95=excluded.p95,
                min=excluded.min,
                max=excluded.max,
                count=excluded.count,
                updated_at=now()
        """

        for b in baselines:
            role_value = b["role"] if b["role"] is not None else "ALL"
            p95 = None
            # compute p95 quickly from event_metrics if desired (optional)
            try:
                p95_row = self.con.execute(
                    """
                    SELECT percentile_cont(0.95) WITHIN GROUP (ORDER BY CAST(metric_value AS DOUBLE))
                    FROM event_metrics m
                    JOIN events e ON m.event_id = e.event_id
                    WHERE m.metric_name = ? AND ( ? IS NULL OR e.role = ? )
                    AND isfinite(CAST(metric_value AS DOUBLE)) AND ABS(CAST(metric_value AS DOUBLE)) < 1e308
                    """,
                    [b["metric"], role_value if role_value != "ALL" else None, role_value if role_value != "ALL" else None],
                ).fetchone()
                p95 = p95_row[0] if p95_row else None
            except Exception:
                p95 = None

            self.con.execute(
                upsert_sql,
                [
                    b["metric"],
                    role_value,
                    b["mean"],
                    b["stddev"],
                    p95,
                    b["min"],
                    b["max"],
                    b["count"],
                ],
            )

        return {"upserted": len(baselines), "min_count": min_count, "top_n": top_n}

    # -------------------------------------------------------
    # Recovery episode detection
    # -------------------------------------------------------
    def recovery_episodes(self, gap_seconds: int = 60, severity_window: int = 30):
        """
        Cluster RecoveryState events into episodes separated by >gap_seconds.
        Returns episode counts/durations and severity spikes around each episode.
        """
        rows = self.con.execute(
            "SELECT ts FROM events WHERE event = 'MasterRecoveryState' ORDER BY ts"
        ).fetchall()
        times = [r[0] for r in rows if r[0]]
        if not times:
            return {"detected": False, "episodes": []}

        episodes = []
        current = [times[0], times[0]]
        for ts in times[1:]:
            if (ts - current[1]).total_seconds() > gap_seconds:
                episodes.append(tuple(current))
                current = [ts, ts]
            else:
                current[1] = ts
        episodes.append(tuple(current))

        detailed = []
        for start, end in episodes:
            window_start = start - timedelta(seconds=severity_window)
            window_end = end + timedelta(seconds=severity_window)
            sev_row = self.con.execute(
                "SELECT MAX(severity) FROM events WHERE ts BETWEEN ? AND ?",
                [window_start, window_end],
            ).fetchone()
            max_sev = sev_row[0] if sev_row else None
            detailed.append({
                "start": start,
                "end": end,
                "duration_seconds": (end - start).total_seconds(),
                "max_severity_nearby": max_sev,
            })

        return {
            "detected": len(detailed) > 0,
            "count": len(detailed),
            "episodes": detailed,
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
    # Rollback Detector (ClogWithRollbacks / Epoch Reset)
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
