import json
from datetime import datetime
import pandas as pd

mandatory_fields = {
    "Severity", "Time", "DateTime", "Type", "Process", "Role",
    "PID", "Machine", "MachineId", "Address", "LogGroup"
}

def parse_log(json_string: str, event_id: int):
    json_obj = json.loads(json_string)

    # Normalize DateTime
    time_stamp = None
    if "DateTime" in json_obj:
        try:
            format_data = "%Y-%m-%dT%H:%M:%SZ"
            time_stamp = datetime.strptime(json_obj["DateTime"], format_data)
        except Exception:
            pass

    # Events table row
    event_row = {
        "event_id": event_id,
        "ts": time_stamp,
        "severity": json_obj.get("Severity"),
        "event": json_obj.get("Type"),
        "process": json_obj.get("Process"),
        "role": json_obj.get("Role"),
        "pid": json_obj.get("PID"),
        "machine_id": json_obj.get("Machine") or json_obj.get("MachineId"),
        "address": json_obj.get("Address"),
        "trace_file": json_obj.get("File"),
        "src_line": json_obj.get("Line"),
        "raw_json": json_obj,
        "fields_json": {k: v for k, v in json_obj.items() if k not in mandatory_fields}
    }

    # Event metrics table rows
    event_metrics = []
    for k, v in event_row["fields_json"].items():
        try:
            event_metrics.append({
                "event_id": event_id,
                "metric_name": k,
                "metric_value": float(v)
            })
        except (ValueError, TypeError):
            continue

    # Wide table row
    wide_row = {
        "event_id": event_id,
        "grv_latency_ms": float(json_obj["GrvLatency"]) if "GrvLatency" in json_obj else None,
        "txn_volume": float(json_obj["TxnVolume"]) if "TxnVolume" in json_obj else None,
        "queue_bytes": float(json_obj["QueueBytes"]) if "QueueBytes" in json_obj else None,
        "durability_lag_s": float(json_obj["DurabilityLag"]) if "DurabilityLag" in json_obj else None,
        "data_move_in_flight": float(json_obj["DataMoveInFlight"]) if "DataMoveInFlight" in json_obj else None,
        "disk_queue_bytes": float(json_obj["DiskQueueBytes"]) if "DiskQueueBytes" in json_obj else None,
        "kv_ops": float(json_obj["KVOps"]) if "KVOps" in json_obj else None,
    }

    # Process table row
    process_key = f"{event_row['machine_id']}|{event_row['pid']}"
    process_row = {
        "process_key": process_key,
        "first_seen_ts": time_stamp,
        "last_seen_ts": time_stamp,
        "address": event_row["machine_id"],
        "pid": event_row["pid"],
        "class": event_row["process"],
        "version": json_obj.get("Version"),
        "command_line": json_obj.get("CommandLine")
    }

    # Process roles rows
    role_rows = []
    if event_row["role"]:
        role_rows.append({
            "process_key": process_key,
            "role": event_row["role"],
            "start_ts": time_stamp,
            "end_ts": None
        })

    return event_row, event_metrics, wide_row, process_row, role_rows


def parse_log_file(path):
    events, metrics, wides, processes, roles = [], [], [], {}, {}

    with open(path) as f:
        for i, line in enumerate(f, start=1):
            if not line.strip():
                continue
            e, m, w, p, r = parse_log(line, i)
            events.append(e)
            metrics.extend(m)
            wides.append(w)

            if p["process_key"] not in processes:
                processes[p["process_key"]] = p

            for rr in r:
                roles[(rr["process_key"], rr["role"], rr["start_ts"])] = rr

    return events, metrics, wides, list(processes.values()), list(roles.values())


def main():
    path = #insert path
    events, metrics, wides, processes, roles = parse_log_file(path)  # <-- fix here

    # Convert to DataFrames
    df_events = pd.DataFrame(events)
    df_metrics = pd.DataFrame(metrics)
    df_wides = pd.DataFrame(wides)
    df_processes = pd.DataFrame(processes)
    df_roles = pd.DataFrame(roles)

    # Save to Parquet
    df_events.to_parquet("./data/events.parquet", index=False)
    df_metrics.to_parquet("./data/event_metrics.parquet", index=False)
    df_wides.to_parquet("./data/events_wide.parquet", index=False)
    df_processes.to_parquet("./data/processes.parquet", index=False)
    df_roles.to_parquet("./data/process_roles.parquet", index=False)

    print(f"  Events: {len(df_events)} rows → ./data/events.parquet")
    print(f"  Metrics: {len(df_metrics)} rows → ./data/event_metrics.parquet")
    print(f"  Wide: {len(df_wides)} rows → ./data/events_wide.parquet")
    print(f"  Processes: {len(df_processes)} rows → ./data/processes.parquet")
    print(f"  Roles: {len(df_roles)} rows → ./data/process_roles.parquet")

if __name__ == "__main__":
    main()
