"""Hotspot selector: find high-severity buckets and gaps."""

from datetime import datetime

from tools.database import get_conn


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
                "bucket_start": datetime.utcfromtimestamp(b),
                "max_severity": sev,
                "count": cnt
            }
            for (b, sev, cnt) in rows
        ]

    def get_uncovered(self, inspected_buckets, min_severity=20, bucket_seconds=600, limit=None):
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
        """
        if limit is not None:
            sql += f" LIMIT {limit}"
        rows = self.con.execute(sql).fetchall()

        return [
            {
                "bucket_start_epoch": b,
                "bucket_start": datetime.utcfromtimestamp(b),
                "max_severity": sev,
                "count": cnt
            }
            for (b, sev, cnt) in rows
        ]
