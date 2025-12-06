"""
Storage service for managing FDB logs in DuckDB.
"""
import json
import os
from datetime import datetime
from typing import Optional

import duckdb

from data_transfer_object.event_dto import EventModel
from tools.parser import LogParser


class StorageService:
    """Data storage service"""
    
    def __init__(self, db_path: str = "fdb_logs.duckdb"):
        """
        Initialize storage service
        
        Args:
            db_path: Database file path
        """
        self.db_path = db_path
        self.db = None
        self.parser = LogParser()
    
    def init_db(self, schema_path: str = None) -> duckdb.DuckDBPyConnection:
        """
        Initialize database connection and create table structure
        
        Args:
            schema_path: SQL schema file path
            
        Returns:
            Database connection object
        """
        self.db = duckdb.connect(self.db_path)
        
        # If schema file is provided, execute SQL
        if schema_path:
            if os.path.exists(schema_path):
                # Drop tables in reverse dependency order to handle foreign keys
                # This ensures we can recreate tables with correct schema
                try:
                    self.db.execute("DROP TABLE IF EXISTS process_roles")
                    self.db.execute("DROP TABLE IF EXISTS events_wide")
                    self.db.execute("DROP TABLE IF EXISTS event_metrics")
                    self.db.execute("DROP TABLE IF EXISTS events")
                    self.db.execute("DROP TABLE IF EXISTS processes")
                except Exception:
                    # Ignore errors if tables don't exist
                    pass
                
                # Now execute the schema
                with open(schema_path, 'r') as f:
                    self.db.execute(f.read())
            else:
                # Use default schema
                self._create_default_schema()
        else:
            self._create_default_schema()
        
        return self.db
    
    def _create_default_schema(self):
        """Create default database schema"""
        # This can include default table structure
        pass
    
    def load_logs_from_file(self, log_path: str, event_id_offset: int = 0) -> int:
        """
        Load logs from file to database
        
        Args:
            log_path: Log file path
            event_id_offset: Event ID offset to ensure uniqueness
            
        Returns:
            Number of loaded events
        """
        if not self.db:
            raise RuntimeError("Database not initialized, please call init_db() first")
        
        count = 0
        for event in self.parser.parse_logs(log_path):
            new_event_id = event.event_id + event_id_offset

            self.insert_event(event, new_event_id)
            self.insert_process(event)
            self.insert_process_role(event)
            self.insert_metrics(new_event_id, event.event, event.fields_json)
            self.insert_events_wide(new_event_id, event.fields_json)

            count += 1
    
        return count
    
    def insert_event(self, event: EventModel, event_id_override: Optional[int] = None):
        """Insert event into events table"""
        row = event.model_dump()
        event_id_value = event_id_override if event_id_override is not None else row["event_id"]

        self.db.execute("""
            INSERT INTO events (
                event_id, ts, severity, event, process, role,
                pid, machine_id, address, trace_file, src_line,
                raw_json, fields_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            str(event_id_value),
            row["ts"],
            row["severity"],
            row["event"],
            row["process"],
            row["role"],
            row["pid"],
            row["machine_id"],
            row["address"],
            row["trace_file"],
            row["src_line"],
            json.dumps(row["raw_json"]),
            json.dumps(row["fields_json"])
        ])
    
    def insert_metrics(self, event_id: int, event_name: str, fields_json: dict):
        """Insert event metrics into metrics table"""
        for k, v in fields_json.items():
            try:
                val = float(v)
            except (ValueError, TypeError):
                continue  # Skip non-numeric types
            
            self.db.execute("""
                INSERT INTO event_metrics (event_id, event, metric_name, metric_value)
                VALUES (?, ?, ?, ?)
            """, [str(event_id), event_name, k, val])
    
    def insert_events_wide(self, event_id: int, fields_json: dict):
            """Insert metrics into events_wide table using REAL FDB fields, safely."""

            def safe_float(v):
                """Parse numeric-ish values; if space-separated, take the max of numeric tokens ignoring -1."""
                if v is None:
                    return None
                if isinstance(v, (int, float)):
                    try:
                        return float(v)
                    except Exception:
                        return None
                if isinstance(v, str):
                    parts = v.split()
                    nums = []
                    for p in parts:
                        if p.lower() in {"inf", "nan"}:
                            continue
                        try:
                            num = float(p)
                            nums.append(num)
                        except Exception:
                            continue
                    if nums:
                        # drop sentinel -1 if there are other values
                        cleaned = [n for n in nums if n != -1]
                        if cleaned:
                            nums = cleaned
                        return max(nums)
                    try:
                        return float(v)
                    except Exception:
                        return None
                return None

            event_id = str(event_id)

            # -----------------------------
            # GRVLatencyMetrics
            # -----------------------------
            if "Mean" in fields_json and "P95" in fields_json:
                # GRVLatencyMetrics fields are always single floats
                grv_latency_ms = safe_float(fields_json["Mean"]) * 1000.0 if safe_float(fields_json["Mean"]) else None
            elif "GRVLatency" in fields_json:
                grv_latency_ms = safe_float(fields_json["GRVLatency"])
            elif "grvLatency" in fields_json:
                grv_latency_ms = safe_float(fields_json["grvLatency"])
            else:
                grv_latency_ms = None

            # -----------------------------
            # Txn volume (CommitProxy / TransactionMetrics)
            # -----------------------------
            txn_volume = None
            if "Committed" in fields_json:
                txn_volume = safe_float(fields_json["Committed"])
            elif "Mutations" in fields_json:
                txn_volume = safe_float(fields_json["Mutations"])
            elif "TxnCommitIn" in fields_json:
                txn_volume = safe_float(fields_json["TxnCommitIn"])
            elif "TxnRequestIn" in fields_json:
                txn_volume = safe_float(fields_json["TxnRequestIn"])

            # -----------------------------
            # Queue metrics
            # -----------------------------
            queue_bytes = None
            if "BytesInput" in fields_json:
                queue_bytes = safe_float(fields_json["BytesInput"])
            elif "QueueSize" in fields_json:
                queue_bytes = safe_float(fields_json["QueueSize"])
            elif "WorstStorageServerQueue" in fields_json:
                queue_bytes = safe_float(fields_json["WorstStorageServerQueue"])
            elif "WorstTLogQueue" in fields_json:
                queue_bytes = safe_float(fields_json["WorstTLogQueue"])

            # -----------------------------
            # Durability lag
            # -----------------------------
            durability_lag_s = None
            if "DurableLag" in fields_json:
                durability_lag_s = safe_float(fields_json["DurableLag"])
            elif "DurabilityLag" in fields_json:
                durability_lag_s = safe_float(fields_json["DurabilityLag"])
            elif "WorstStorageServerDurabilityLag" in fields_json:
                durability_lag_s = safe_float(fields_json["WorstStorageServerDurabilityLag"])
            elif "DurableVersion" in fields_json and "Version" in fields_json:
                v = safe_float(fields_json["Version"])
                dv = safe_float(fields_json["DurableVersion"])
                if v is not None and dv is not None:
                    durability_lag_s = (v - dv) / 1e5

            # -----------------------------
            # Data movement
            # -----------------------------
            data_move_in_flight = safe_float(fields_json.get("InFlightBytes"))

            # -----------------------------
            # Disk queue
            # -----------------------------
            disk_queue_bytes = safe_float(fields_json.get("DiskQueue"))

            # -----------------------------
            # TLog KV operations
            # -----------------------------
            kv_ops = safe_float(fields_json.get("Ops"))

            # -----------------------------
            # Insert safely
            # -----------------------------
            self.db.execute("""
                INSERT INTO events_wide (
                    event_id, grv_latency_ms, txn_volume, queue_bytes,
                    durability_lag_s, data_move_in_flight, disk_queue_bytes, kv_ops
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (event_id) DO UPDATE SET
                    grv_latency_ms = excluded.grv_latency_ms,
                    txn_volume = excluded.txn_volume,
                    queue_bytes = excluded.queue_bytes,
                    durability_lag_s = excluded.durability_lag_s,
                    data_move_in_flight = excluded.data_move_in_flight,
                    disk_queue_bytes = excluded.disk_queue_bytes,
                    kv_ops = excluded.kv_ops
            """, [
                event_id, grv_latency_ms, txn_volume, queue_bytes,
                durability_lag_s, data_move_in_flight, disk_queue_bytes, kv_ops
            ])


    def insert_process(self, event: EventModel):
        """
        Ensure process exists even if no Worker/ProcessStart events exist.
        Many simulation logs (CloggedSideband, etc.) never emit process spawn events.
        """

        # Process key: use event.address if present; otherwise derive from Machine field.
        pk = None

        # 1. Preferred: address
        if event.address:
            pk = event.address

        # 2. If missing, try fields_json["Machine"]
        elif "Machine" in event.fields_json:
            pk = event.fields_json["Machine"]

        # 3. If still missing, nothing we can do
        if not pk:
            return

        # Insert or update process row
        self.db.execute("""
            INSERT INTO processes (process_key, first_seen_ts, last_seen_ts, address)
            VALUES (?, ?, ?, ?)
            ON CONFLICT (process_key) DO UPDATE SET
                last_seen_ts = excluded.last_seen_ts
        """, [
            pk,
            event.ts,
            event.ts,
            pk
        ])

    def insert_process_role(self, event: EventModel):
        """Attach roles to processes safely."""
        
        pk = None

        if event.address:
            pk = event.address
        elif "Machine" in event.fields_json:
            pk = event.fields_json["Machine"]
        else:
            return

        # Skip if no role
        if not event.role:
            return

        self.db.execute("""
            INSERT INTO process_roles (process_key, role, start_ts)
            VALUES (?, ?, ?)
            ON CONFLICT DO NOTHING
        """, [pk, event.role, event.ts])

    
    def create_rollups(self, interval_seconds: int = 60):
        """
        Create rollup materializations for metrics.

        Args:
            interval_seconds: Window size in seconds (default 60)
        """
        if not self.db:
            raise RuntimeError("Database not initialized, please call init_db() first")
        
        self.db.execute(f"""
            CREATE TABLE IF NOT EXISTS rollups_{interval_seconds}s AS
            SELECT
                date_trunc('second', ts) - (extract(epoch from ts)::int % {interval_seconds}) * INTERVAL '1 second' AS window_start,
                role,
                metric_name,
                COUNT(*) AS n,
                AVG(metric_value) AS avg,
                MAX(metric_value) AS max,
                quantile_cont(metric_value, 0.95) AS p95
            FROM events
            JOIN event_metrics USING (event_id)
            GROUP BY 1, 2, 3
            ORDER BY 1, 2, 3;
        """)

        # Create the window view
        # self.db.execute(f"""
        #     create or replace view _win_{interval_seconds}s as
        #     select range as window_start, window_start + interval {interval_seconds} second as window_end
        #     from range((select min(ts) from events),
        #                (select max(ts) from events),
        #                interval {interval_seconds} second);
        # """)

        # Create the rollups table
        # self.db.execute(f"""
        #     create table if not exists rollups_{interval_seconds}s as
        #     select
        #       w.window_start,
        #       e.role,
        #       m.metric_name,
        #       count(*)                            as n,
        #       avg(m.metric_value)                 as avg,
        #       max(m.metric_value)                 as max,
        #       quantile_cont(m.metric_value, 0.95) as p95
        #     from _win_{interval_seconds}s w
        #     join events e
        #       on e.ts >= w.window_start and e.ts < w.window_end
        #     join event_metrics m using (event_id)
        #     group by 1,2,3;
        # """)
        print(f"✅ Created rollups_{interval_seconds}s table with windowed aggregates")

    def check_events_loaded(self) -> bool:
        """Check if events have been loaded"""
        if not self.db:
            raise RuntimeError("Database not initialized, please call init_db() first")
        
        rows = self.db.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        return rows > 0
    
    def get_event_count(self) -> int:
        """Get total event count"""
        if not self.db:
            raise RuntimeError("Database not initialized, please call init_db() first")
        
        return self.db.execute("SELECT COUNT(*) FROM events").fetchone()[0]
    
    def query(self, sql: str) -> duckdb.DuckDBPyRelation:
        """
        Execute custom SQL query
        
        Args:
            sql: SQL query statement
            
        Returns:
            Query results
        """
        if not self.db:
            raise RuntimeError("Database not initialized, please call init_db() first")
        
        return self.db.execute(sql)
    
    def close(self):
        """Close database connection"""
        if self.db:
            self.db.close()
            self.db = None
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()
