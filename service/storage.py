"""
Storage service for managing FDB logs in DuckDB.
"""
import duckdb
import json
import os
from typing import Optional
from datetime import datetime
from dto.event import EventModel
from .parser import LogParser


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
            self.insert_event(event, event_id_override=new_event_id)
            self.insert_metrics(new_event_id, event.fields_json)
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
    
    def insert_metrics(self, event_id: int, fields_json: dict):
        """Insert event metrics into metrics table"""
        for k, v in fields_json.items():
            try:
                val = float(v)
            except (ValueError, TypeError):
                continue  # Skip non-numeric types
            
            self.db.execute("""
                INSERT INTO event_metrics (event_id, metric_name, metric_value)
                VALUES (?, ?, ?)
            """, [str(event_id), k, val])
    
    def insert_events_wide(self, event_id: int, fields_json: dict):
        """Insert event into wide table (includes common metrics)"""
        # Extract known metrics
        grv_latency = float(fields_json.get("GRVLatency", "nan")) if "GRVLatency" in fields_json else None
        txn_volume = float(fields_json.get("TxnVolume", "nan")) if "TxnVolume" in fields_json else None
        queue_bytes = float(fields_json.get("QueueBytes", "nan")) if "QueueBytes" in fields_json else None
        
        self.db.execute("""
            INSERT INTO events_wide (
                event_id, grv_latency_ms, txn_volume, queue_bytes
            ) VALUES (?, ?, ?, ?)
        """, [str(event_id), grv_latency, txn_volume, queue_bytes])
    
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
