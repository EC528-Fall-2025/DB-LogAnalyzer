"""Timeline builder for agentic loop.

Constructs a chronological narrative the LLM can consume directly
instead of inferring order itself.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from data_transfer_object.event_dto import EventModel


class TimelineBuilder:
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path

    def build(
        self,
        all_events: List[EventModel],
        detector_outputs: Optional[Dict[str, Any]] = None,
        hotspots: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Build a structured, chronological timeline for the LLM.

        Returns a dict with:
        - first_anomaly: earliest notable/severe event
        - timeline: ordered list with relative timestamps
        - root_cause_signal: simple causal hint (e.g., storage lag precedes recovery)
        - hot_buckets: top hotspot buckets (if any)
        - detector_signals: detector outputs echoed back
        """
        if not all_events:
            return {}

        events = [e for e in all_events if e.ts]
        if not events:
            return {}

        events.sort(key=lambda e: e.ts)
        start_ts = events[0].ts

        def rel(ts: Optional[datetime]) -> str:
            if not ts:
                return "N/A"
            return f"+{(ts - start_ts).total_seconds():.1f}s"

        first_severe = next((e for e in events if (e.severity or 0) >= 30), events[0])

        # Metric-based markers
        versionlag_events = []
        for e in events:
            if not e.fields_json:
                continue
            for key in ["VersionLag", "versionLag", "VersionLagValue", "Lag", "lag"]:
                if key in e.fields_json:
                    try:
                        v = float(e.fields_json[key])
                        if v > 100000:
                            versionlag_events.append((e, v))
                    except Exception:
                        pass
                    break

        first_lag_100k = versionlag_events[0] if versionlag_events else None
        first_lag_1m = next(((e, v) for e, v in versionlag_events if v > 1_000_000), None)

        recovery_events = [e for e in events if "Recovery" in str(e.event)]
        first_recovery = recovery_events[0] if recovery_events else None

        timeline_items: List[Dict[str, Any]] = []
        timeline_items.append({
            "t": rel(first_severe.ts),
            "event": first_severe.event,
            "note": "Earliest notable/severe event",
        })
        if first_lag_100k:
            timeline_items.append({
                "t": rel(first_lag_100k[0].ts),
                "event": f"VersionLag={int(first_lag_100k[1])}",
                "note": "Lag exceeds 100k (storage pressure signal)",
            })
        if first_lag_1m:
            timeline_items.append({
                "t": rel(first_lag_1m[0].ts),
                "event": f"VersionLag={int(first_lag_1m[1])}",
                "note": "Lag exceeds 1M (critical storage pressure)",
            })
        if first_recovery:
            timeline_items.append({
                "t": rel(first_recovery.ts),
                "event": first_recovery.event,
                "note": "Recovery activity begins",
            })

        # Detector marks (if they expose timestamps)
        detector_marks = []
        detector_outputs = detector_outputs or {}
        for name, result in detector_outputs.items():
            ts = None
            if isinstance(result, dict):
                ts = result.get("first_ts") or result.get("timestamp")
            if ts:
                detector_marks.append({
                    "t": rel(ts) if hasattr(ts, "isoformat") else ts,
                    "event": name,
                    "note": "Detector triggered",
                })
        timeline_items.extend(detector_marks)

        hot_buckets = []
        if hotspots:
            for b in hotspots[:3]:
                hot_buckets.append({
                    "bucket_start": b.get("bucket_start") or b.get("bucket_start_epoch"),
                    "max_severity": b.get("max_severity"),
                    "count": b.get("count"),
                })

        root_signal = None
        if first_lag_100k and first_recovery and first_lag_100k[0].ts <= first_recovery.ts:
            root_signal = "storage_pressure_precedes_recovery"
        elif first_recovery:
            root_signal = "recovery_precedes_storage_pressure"

        return {
            "first_anomaly": {
                "timestamp": first_severe.ts.isoformat() if first_severe.ts else "N/A",
                "event": first_severe.event,
                "meaning": "Earliest notable/severe event",
            },
            "timeline": timeline_items,
            "root_cause_signal": root_signal,
            "hot_buckets": hot_buckets,
            "detector_signals": detector_outputs,
        }
