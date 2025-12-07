"""Core detectors with numeric grounding."""

from datetime import datetime

from tools.database import get_conn
from tools.anomaly_detector import MetricAnomalyDetector
from .context_analyzer import ContextAnalyzer
from .helpers import _ensure_dict, _parse_event_row, _percentile


class Detectors:
    """Core root-cause signals."""

    def __init__(self, db_path):
        self.db_path = db_path
        self.con = get_conn(db_path)
        self.ctx = ContextAnalyzer(db_path)
        self._baseline_cache = {}

    def _get_baseline(self, metric_name: str, role: str):
        """
        Fetch baseline stats from metric_baselines table, cached by (metric, role).
        Falls back to role='ALL' if role-specific is not found.
        """
        key = (metric_name, role)
        if key in self._baseline_cache:
            return self._baseline_cache[key]

        row = self.con.execute(
            """
            SELECT mean, stddev, p95, count
            FROM metric_baselines
            WHERE metric_name = ? AND role = ?
            """,
            [metric_name, role],
        ).fetchone()

        if not row and role != "ALL":
            row = self.con.execute(
                """
                SELECT mean, stddev, p95, count
                FROM metric_baselines
                WHERE metric_name = ? AND role = 'ALL'
                """,
                [metric_name],
            ).fetchone()

        baseline = None
        if row:
            baseline = {
                "mean": row[0],
                "stddev": row[1],
                "p95": row[2],
                "count": row[3],
            }
        self._baseline_cache[key] = baseline
        return baseline

    # STORAGE PRESSURE --------------------------------------
    def storage_engine_pressure(self, lag_threshold=50000, z_score_threshold: float = 3.0,
                                start_time=None, end_time=None):
        sql = """
            SELECT ts, role, fields_json
            FROM events
            WHERE event = 'StorageMetrics'
        """
        params = []
        where = ""
        if start_time and end_time:
            where = " AND ts BETWEEN ? AND ?"
            params = [start_time, end_time]

        rows = self.con.execute(sql + where, params).fetchall()

        lag_points = []
        for ts, role, f in rows:
            d = _ensure_dict(f)
            lag = d.get("VersionLag") or d.get("versionLag")
            if lag is None:
                continue
            try:
                lag_val = float(lag)
            except Exception:
                continue
            lag_points.append((ts, role or "ALL", lag_val))

        if not lag_points:
            return {"detected": False, "reason": "no storage metrics"}

        lags = [lv for _, _, lv in lag_points]
        anomalies = []
        for ts, role, lv in lag_points:
            baseline = self._get_baseline("VersionLag", role or "ALL")
            z = None
            if baseline and baseline.get("stddev") and baseline["stddev"] > 0:
                try:
                    z = abs((lv - baseline["mean"]) / baseline["stddev"])
                except Exception:
                    z = None
            # flag if z-score exceeds threshold or if raw value exceeds explicit threshold
            if (z is not None and z >= z_score_threshold) or (lv > lag_threshold):
                anomalies.append({"ts": ts, "role": role, "value": lv, "zscore": z})

        high = anomalies
        first_high_ts = high[0]["ts"] if high else None
        last_high_ts = high[-1]["ts"] if high else None
        max_z = max((a["zscore"] for a in anomalies if a["zscore"] is not None), default=None)

        return {
            "detected": len(high) > 0,
            "max_lag": max(lags),
            "p95_lag": _percentile(lags, 95),
            "mean_lag": sum(lags) / len(lags),
            "count_high": len(high),
            "total": len(lags),
            "threshold": lag_threshold,
            "z_score_threshold": z_score_threshold,
            "max_zscore": max_z,
            "first_high_ts": first_high_ts,
            "last_high_ts": last_high_ts,
            "anomalies_sample": high[:5],
        }

    # RATEKEEPER -------------------------------------------
    def ratekeeper_throttling(self, start_time=None, end_time=None):
        sql = """
            SELECT ts, event, fields_json
            FROM events
            WHERE event LIKE '%Ratekeeper%' OR event LIKE '%Throttle%'
        """
        params = []
        if start_time and end_time:
            sql += " AND ts BETWEEN ? AND ?"
            params = [start_time, end_time]
        rows = self.con.execute(sql, params).fetchall()

        out = []
        for ts, ev, f in rows:
            f = _ensure_dict(f)
            if "throttle" in str(ev).lower() or any("throttle" in k.lower() for k in f):
                out.append({"timestamp": ts, "event": ev, "fields": f})

        return {
            "detected": len(out) > 0,
            "count": len(out),
            "first_ts": out[0]["timestamp"] if out else None,
            "last_ts": out[-1]["timestamp"] if out else None,
            "events": out[:10],  # keep sample small
        }

    # MISSING TLOGS ----------------------------------------
    def missing_tlogs(self, start_time=None, end_time=None):
        sql = """
            SELECT ts, event, severity, fields_json
            FROM events
            WHERE event LIKE '%TLog%' AND 
                  (event LIKE '%Missing%' OR event LIKE '%Failed%' OR event LIKE '%Error%')
        """
        params = []
        if start_time and end_time:
            sql += " AND ts BETWEEN ? AND ?"
            params = [start_time, end_time]
        rows = self.con.execute(sql, params).fetchall()
        sample = []
        for ts, ev, sev, f in rows[:10]:
            sample.append({"timestamp": ts, "event": ev, "severity": sev, "fields": _ensure_dict(f)})
        return {
            "detected": len(rows) > 0,
            "count": len(rows),
            "first_ts": rows[0][0] if rows else None,
            "last_ts": rows[-1][0] if rows else None,
            "events": sample,
        }

    # RECOVERY LOOP ----------------------------------------
    def recovery_loop(self, threshold=3, window_seconds=60, start_time=None, end_time=None):
        # use similar_events with optional time filtering
        if start_time and end_time:
            timeline = self.ctx.similar_events("MasterRecoveryState", limit=500)
            timeline = [e for e in timeline if start_time <= datetime.fromisoformat(e["timestamp"]) <= end_time]
        else:
            timeline = self.ctx.similar_events("MasterRecoveryState", limit=500)
        if len(timeline) < threshold:
            return {"detected": False}

        times = [datetime.fromisoformat(e["timestamp"]) for e in timeline]

        loop_count = 0
        for i in range(len(times) - threshold):
            if (times[i + threshold - 1] - times[i]).total_seconds() <= window_seconds:
                loop_count += 1

        first_ts = min(times) if times else None
        last_ts = max(times) if times else None
        duration = (last_ts - first_ts).total_seconds() if (first_ts and last_ts) else None

        return {
            "detected": loop_count > 0,
            "loop_count": loop_count,
            "first_ts": first_ts,
            "last_ts": last_ts,
            "duration_seconds": duration,
        }

    # COORDINATION LOSS ------------------------------------
    def coordination_loss(self, start_time=None, end_time=None):
        sql = """
            SELECT ts, event, fields_json
            FROM events
            WHERE event LIKE '%Coordinator%'
        """
        params = []
        if start_time and end_time:
            sql += " AND ts BETWEEN ? AND ?"
            params = [start_time, end_time]
        rows = self.con.execute(sql, params).fetchall()

        loss = []
        for ts, ev, f in rows:
            s = str(ev).lower()
            f = str(f).lower()
            if any(x in s for x in ["fail", "lost"]) or any(x in f for x in ["fail", "lost"]):
                loss.append({"timestamp": ts, "event": ev})

        return {
            "detected": len(loss) > 0,
            "count": len(loss),
            "first_ts": loss[0]["timestamp"] if loss else None,
            "last_ts": loss[-1]["timestamp"] if loss else None,
            "events": loss[:10],
        }

    # Z-SCORE HOTSPOTS -------------------------------------
    def zscore_hotspots(self, bucket_seconds=300, min_z=2.0, limit=20):
        """
        Find time buckets with unusually high event counts using z-score over bucket counts.
        Uses DuckDB for fast aggregation.
        """
        sql = f"""
        WITH bucketed AS (
            SELECT
                (FLOOR(EXTRACT(EPOCH FROM ts) / {bucket_seconds}) * {bucket_seconds}) AS bucket,
                COUNT(*) AS cnt,
                MAX(severity) AS max_sev
            FROM events
            GROUP BY 1
        ),
        stats AS (
            SELECT AVG(cnt) AS mean_cnt, STDDEV_SAMP(cnt) AS std_cnt FROM bucketed
        )
        SELECT
            b.bucket,
            b.cnt,
            b.max_sev,
            (b.cnt - s.mean_cnt) / NULLIF(s.std_cnt, 0) AS zscore
        FROM bucketed b, stats s
        WHERE s.std_cnt IS NOT NULL AND s.std_cnt > 0
        ORDER BY zscore DESC
        LIMIT ?
        """
        try:
            rows = self.con.execute(sql, [limit]).fetchall()
        except Exception as e:
            return {"detected": False, "error": str(e)}

        hotspots = []
        for bucket_epoch, cnt, max_sev, z in rows:
            if z is None:
                continue
            if z < min_z:
                continue
            hotspots.append({
                "bucket_start_epoch": bucket_epoch,
                "bucket_start": datetime.fromtimestamp(bucket_epoch),
                "count": cnt,
                "max_severity": max_sev,
                "zscore": z,
            })

        return {
            "detected": len(hotspots) > 0,
            "hotspots": hotspots,
            "bucket_seconds": bucket_seconds,
            "min_z": min_z,
        }

    # BASELINE WINDOW ANOMALIES -----------------------------
    def baseline_window_anomalies(self, bucket_seconds=30, z_score_threshold=3.0, min_samples=3, metrics=None):
        """
        Flag time buckets whose metric means deviate from baselines (per role) using z-score.
        metrics: optional list of metric names to consider; if None, use common high-signal metrics.
        """
        metrics = metrics or ["VersionLag", "DurabilityLag", "BytesInput", "WorstStorageServerQueue", "WorstStorageServerDurabilityLag"]
        metric_list = ",".join(f"'{m}'" for m in metrics)

        sql = f"""
        WITH bucketed AS (
            SELECT
                (FLOOR(EXTRACT(EPOCH FROM e.ts) / {bucket_seconds}) * {bucket_seconds}) AS bucket,
                e.role,
                m.metric_name,
                AVG(CAST(m.metric_value AS DOUBLE)) AS mean_val,
                COUNT(*) AS cnt
            FROM event_metrics m
            JOIN events e ON m.event_id = e.event_id
            WHERE m.metric_name IN ({metric_list})
              AND isfinite(CAST(m.metric_value AS DOUBLE))
            GROUP BY 1,2,3
        )
        SELECT bucket, role, metric_name, mean_val, cnt
        FROM bucketed
        WHERE cnt >= ?
        ORDER BY bucket, role, metric_name
        """

        try:
            rows = self.con.execute(sql, [min_samples]).fetchall()
        except Exception as e:
            return {"detected": False, "error": str(e)}

        anomalies = []
        for bucket_epoch, role, metric_name, mean_val, cnt in rows:
            base = self._get_baseline(metric_name, role or "ALL")
            z = None
            if base and base.get("stddev") and base["stddev"] > 0:
                try:
                    z = abs((mean_val - base["mean"]) / base["stddev"])
                except Exception:
                    z = None
            if z is not None and z >= z_score_threshold:
                anomalies.append({
                    "bucket_start_epoch": bucket_epoch,
                    "bucket_start": datetime.fromtimestamp(bucket_epoch),
                    "role": role,
                    "metric": metric_name,
                    "mean_val": mean_val,
                    "baseline_mean": base.get("mean") if base else None,
                    "baseline_std": base.get("stddev") if base else None,
                    "zscore": z,
                    "count": cnt,
                })

        anomalies.sort(key=lambda a: a["bucket_start_epoch"])
        first_anomaly = anomalies[0] if anomalies else None

        return {
            "detected": len(anomalies) > 0,
            "anomalies": anomalies[:20],
            "first_anomaly": first_anomaly,
            "bucket_seconds": bucket_seconds,
            "z_score_threshold": z_score_threshold,
        }

    # METRIC ANOMALIES (per-event z-score using MetricAnomalyDetector) -----
    def metric_anomalies(self, limit: int = 500, z_score_threshold: float = 2.5):
        """
        Run MetricAnomalyDetector on the most recent events to flag per-event anomalies.
        Returns a summary with counts and sample anomalies.
        """
        try:
            rows = self.con.execute(
                """
                SELECT event_id, ts, severity, event, role, fields_json
                FROM events
                ORDER BY ts DESC
                LIMIT ?
                """,
                [limit],
            ).fetchall()
        except Exception as e:
            return {"detected": False, "error": str(e)}

        events = [_parse_event_row(r) for r in rows]
        detector = MetricAnomalyDetector(z_score_threshold=z_score_threshold)
        anomalies = detector.detect_anomalies(events)

        flagged = [(e, reasons) for e, reasons in anomalies if reasons and "insufficient_data" not in reasons]
        sample = []
        for e, reasons in flagged[:10]:
            sample.append({
                "ts": e.ts,
                "event": e.event,
                "severity": e.severity,
                "role": e.role,
                "reasons": reasons,
            })

        return {
            "detected": len(flagged) > 0,
            "total_events": len(events),
            "anomalies_detected": detector.stats.get("anomalies_detected"),
            "by_method": detector.stats.get("by_method"),
            "sample": sample,
        }
