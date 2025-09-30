import os
import duckdb

# -----------------------------
# DB INIT
# -----------------------------

def init_db(path="data/fdb_logs.duckdb"):
    """
    Initialize the DuckDB database, create all tables,
    and automatically load Parquet outputs (events, metrics, processes, etc.)
    when the database is first initialized.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    db = duckdb.connect(path)
    create_all_tables(db)

    # ✅ Added per Mira’s request:
    # Automatically load logs (events, metrics, processes, process_roles)
    load_logs_from_file(db, dir_path="data")

    return db


def create_all_tables(db):
    """
    Create all tables: events, event_metrics, events_wide, processes, and process_roles.
    NOTE: event_id is BIGINT across tables to keep FK types consistent.
    """

    # === EVENTS TABLE ===
    db.execute("""
    CREATE TABLE IF NOT EXISTS events (
        event_id          BIGINT PRIMARY KEY,
        ts                TIMESTAMP NOT NULL,
        severity          SMALLINT,
        event             VARCHAR NOT NULL,
        process           VARCHAR,
        role              VARCHAR,
        pid               INTEGER,
        machine_id        VARCHAR,
        address           VARCHAR,
        trace_file        VARCHAR,
        src_line          INTEGER,
        raw_json          JSON,
        fields_json       JSON
    );
    """)

    # === EVENT METRICS TABLE ===
    db.execute("""
    CREATE TABLE IF NOT EXISTS event_metrics (
        event_id     BIGINT NOT NULL,
        metric_name  VARCHAR NOT NULL,
        metric_value DOUBLE,
        unit         VARCHAR,
        is_counter   BOOLEAN DEFAULT FALSE,
        PRIMARY KEY (event_id, metric_name),
        FOREIGN KEY (event_id) REFERENCES events(event_id)
    );
    """)

    # === EVENTS WIDE TABLE ===
    db.execute("""
    CREATE TABLE IF NOT EXISTS events_wide (
        event_id            BIGINT PRIMARY KEY REFERENCES events(event_id),
        grv_latency_ms      DOUBLE,
        txn_volume          DOUBLE,
        queue_bytes         DOUBLE,
        durability_lag_s    DOUBLE,
        data_move_in_flight DOUBLE,
        disk_queue_bytes    DOUBLE,
        kv_ops              DOUBLE
    );
    """)

    # === PROCESSES TABLE ===
    db.execute("""
    CREATE TABLE IF NOT EXISTS processes (
        process_key   VARCHAR PRIMARY KEY,
        first_seen_ts TIMESTAMP,
        last_seen_ts  TIMESTAMP,
        address       VARCHAR,
        pid           INTEGER,
        class         VARCHAR,
        version       VARCHAR,
        command_line  VARCHAR
    );
    """)

    # === PROCESS ROLES TABLE ===
    db.execute("""
    CREATE TABLE IF NOT EXISTS process_roles (
        process_key   VARCHAR REFERENCES processes(process_key),
        role          VARCHAR,
        start_ts      TIMESTAMP,
        end_ts        TIMESTAMP,
        PRIMARY KEY (process_key, role, start_ts)
    );
    """)


# -----------------------------
# LOADERS
# -----------------------------

def load_logs_from_file(db, dir_path="data"):
    """
    Load all parsed Parquet files (produced by Sprint v1.x parsers) into DuckDB tables.
    This function is automatically called during init_db().
    """
    print("📥 Loading Parquet outputs into DuckDB…")
    loaded_any = False

    def _file(p):
        return os.path.join(dir_path, p)

    # events.parquet
    if os.path.exists(_file("events.parquet")):
        db.execute("""
            INSERT OR REPLACE INTO events
            SELECT * FROM read_parquet(?)
        """, [_file("events.parquet")])
        print("  • events ✅")
        loaded_any = True
    else:
        print("  • events.parquet not found (skipped)")

    # event_metrics.parquet
    if os.path.exists(_file("event_metrics.parquet")):
        db.execute("""
            INSERT OR REPLACE INTO event_metrics (event_id, metric_name, metric_value, unit, is_counter)
            SELECT event_id, metric_name, metric_value, unit, COALESCE(is_counter, FALSE)
            FROM read_parquet(?)
        """, [_file("event_metrics.parquet")])
        print("  • event_metrics ✅")
        loaded_any = True
    else:
        print("  • event_metrics.parquet not found (skipped)")

    # events_wide.parquet
    if os.path.exists(_file("events_wide.parquet")):
        db.execute("""
            INSERT OR REPLACE INTO events_wide
            SELECT * FROM read_parquet(?)
        """, [_file("events_wide.parquet")])
        print("  • events_wide ✅")
        loaded_any = True
    else:
        print("  • events_wide.parquet not found (skipped)")

    # processes.parquet
    if os.path.exists(_file("processes.parquet")):
        db.execute("""
            INSERT OR REPLACE INTO processes
            SELECT * FROM read_parquet(?)
        """, [_file("processes.parquet")])
        print("  • processes ✅")
        loaded_any = True
    else:
        print("  • processes.parquet not found (skipped)")

    # process_roles.parquet
    if os.path.exists(_file("process_roles.parquet")):
        db.execute("""
            INSERT OR REPLACE INTO process_roles
            SELECT * FROM read_parquet(?)
        """, [_file("process_roles.parquet")])
        print("  • process_roles ✅")
        loaded_any = True
    else:
        print("  • process_roles.parquet not found (skipped)")

    if loaded_any:
        print("✅ All available tables loaded.")
    else:
        print("⚠️ No Parquet files found in:", dir_path)


def load_logs_from_json(db, json_path):
    """
    Optional: Load NDJSON logs directly (if you ever want to ingest raw lines).
    Keeps the staging behavior. Not required for Mira’s ask, but handy.
    """
    db.execute("""
        CREATE OR REPLACE TABLE _staging_raw AS
        SELECT
            row_number() OVER () AS event_id,
            COALESCE(
              TRY_CAST(json_extract_string(j, '$.DateTime') AS TIMESTAMP),
              TRY_CAST(json_extract_string(j, '$.Time') AS TIMESTAMP)
            ) AS ts,
            TRY_CAST(json_extract_string(j, '$.Severity') AS SMALLINT) AS severity,
            json_extract_string(j, '$.Type') AS event,
            json_extract_string(j, '$.Process') AS process,
            json_extract_string(j, '$.Role') AS role,
            TRY_CAST(json_extract_string(j, '$.PID') AS INTEGER) AS pid,
            json_extract_string(j, '$.Machine') AS machine_id,
            json_extract_string(j, '$.Address') AS address,
            json_extract_string(j, '$.File') AS trace_file,
            TRY_CAST(json_extract_string(j, '$.Line') AS INTEGER) AS src_line,
            j AS raw_json,
            j AS fields_json
        FROM read_json_auto(?, format='newline_delimited') AS t(j)
    """, [json_path])

    db.execute("""
        INSERT OR REPLACE INTO events
        SELECT event_id, ts, severity, event, process, role, pid, machine_id, address, trace_file, src_line, raw_json, fields_json
        FROM _staging_raw
        ORDER BY event_id
    """)
    print(f"✅ Staged and inserted events from {json_path}")


# -----------------------------
# SCRIPT ENTRYPOINT
# -----------------------------
if __name__ == "__main__":
    db = init_db()
    print("✅ Database initialized and tables created.")
