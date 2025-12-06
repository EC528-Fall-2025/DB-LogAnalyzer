"""ContextAnalyzer: local deep-dive investigation tools."""

from datetime import datetime, timedelta

from tools.database import get_conn
from .helpers import _parse_event_row, _ensure_dict


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
