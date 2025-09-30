import duckdb
import json
import datetime
from datetime import datetime
from sprint_v1_1.parser import parse_logs, EventModel

def load_into_db(db, log_path: str):
    for e in parse_logs(log_path):
        insert_event(db, e)
        insert_metrics(db, e.event_id, e.fields_json)
        insert_events_wide(db, e.event_id, e.fields_json)
        
def preprocess_json(in_path, out_path):
    with open(in_path) as f:
        lines = f.readlines()

    out_lines = []
    for line in lines:
        obj = json.loads(line)

        # Normalize DateTime if present
        if "DateTime" in obj:
            try:
                # Parse the ISO format with trailing Z (UTC)
                dt =  datetime.strptime(obj['DateTime'], "%Y-%m-%dT%H:%M:%SZ")
                # Store in normalized form without Z (DuckDB friendly)
                obj["DateTimeParsed"] = dt.strftime("%Y-%m-%d %H:%M:%S")
            except Exception as e:
                # if parsing fails, keep original
                obj["DateTimeParsed"] = obj["DateTime"]
        print(obj['DateTimeParsed'])
        out_lines.append(json.dumps(obj))

    with open(out_path, "w") as f:
        f.write("\n".join(out_lines))

# Example usage:
def init_db(path="fdb_logs.duckdb"):
    db = duckdb.connect(path)
    db.execute(open("./data/schema.sql").read())
    return db

    
def insert_event(db, e: EventModel):
    row = e.model_dump()

    db.execute("""
        INSERT INTO events (
            event_id, ts, severity, event, process, role,
            pid, machine_id, address, trace_file, src_line,
            raw_json, fields_json
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, [
        row["event_id"],
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

def insert_metrics(db, event_id: int, fields_json: dict):
    for k, v in fields_json.items():
        try:
            val = float(v)
        except (ValueError, TypeError):
            continue  # skip non-numeric values

        db.execute("""
            INSERT INTO event_metrics (event_id, metric_name, metric_value)
            VALUES (?, ?, ?)
        """, [event_id, k, val])

def insert_events_wide(db, event_id: int, fields_json: dict):
    # pick out known metrics if present
    grv_latency = float(fields_json.get("GRVLatency", "nan")) if "GRVLatency" in fields_json else None
    txn_volume = float(fields_json.get("TxnVolume", "nan")) if "TxnVolume" in fields_json else None
    queue_bytes = float(fields_json.get("QueueBytes", "nan")) if "QueueBytes" in fields_json else None

    db.execute("""
        INSERT INTO events_wide (
            event_id, grv_latency_ms, txn_volume, queue_bytes
        ) VALUES (?, ?, ?, ?)
    """, [event_id, grv_latency, txn_volume, queue_bytes])



