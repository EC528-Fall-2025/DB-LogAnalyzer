# Sprint v1.1/run_parser.py
from pathlib import Path
import pandas as pd
from typing import Dict, Any

# ✅ direct import (no leading dot) so this runs as a script
from log_parser import parse_logs

IN_PATH = "data/sample_log.json"   # change if needed
OUT_DIR = Path("data")             # parquet output folder


def is_number(x) -> bool:
    try:
        if isinstance(x, (bool, type(None))):
            return False
        float(x)
        return True
    except Exception:
        return False


def main():
    events = []
    metrics = []
    wides = []
    processes: Dict[str, Dict[str, Any]] = {}
    roles = []

    for e in parse_logs(IN_PATH):
        # events row
        events.append({
            "event_id": e.event_id,
            "ts": e.ts,
            "severity": e.severity,
            "event": e.event,
            "process": e.process,
            "role": e.role,
            "pid": e.pid,
            "machine_id": e.machine_id,
            "address": e.address,
            "trace_file": e.trace_file,
            "src_line": e.src_line,
            "raw_json": e.raw_json,
            "fields_json": e.fields_json,
        })

        # event_metrics rows: numeric fields from fields_json
        for k, v in e.fields_json.items():
            if is_number(v):
                metrics.append({
                    "event_id": e.event_id,
                    "metric_name": k,
                    "metric_value": float(v),
                    "unit": None,
                    "is_counter": False,
                })

        # optional wide table values
        def f(key):
            v = e.raw_json.get(key)
            return float(v) if is_number(v) else None

        wides.append({
            "event_id": e.event_id,
            "grv_latency_ms": f("GrvLatency"),
            "txn_volume": f("TxnVolume"),
            "queue_bytes": f("QueueBytes"),
            "durability_lag_s": f("DurabilityLag"),
            "data_move_in_flight": f("DataMoveInFlight"),
            "disk_queue_bytes": f("DiskQueueBytes"),
            "kv_ops": f("KVOps"),
        })

        # processes & roles
        pkey = f"{e.machine_id}|{e.pid}"
        if pkey not in processes:
            processes[pkey] = {
                "process_key": pkey,
                "first_seen_ts": e.ts,
                "last_seen_ts": e.ts,
                "address": e.machine_id,
                "pid": e.pid,
                "class": e.process,
                "version": e.raw_json.get("Version"),
                "command_line": e.raw_json.get("CommandLine"),
            }
        else:
            if e.ts and (processes[pkey]["last_seen_ts"] is None or e.ts > processes[pkey]["last_seen_ts"]):
                processes[pkey]["last_seen_ts"] = e.ts

        if e.role:
            roles.append({
                "process_key": pkey,
                "role": e.role,
                "start_ts": e.ts,
                "end_ts": None,
            })

    # write parquet
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(events).to_parquet(OUT_DIR / "events.parquet", index=False)
    pd.DataFrame(metrics).to_parquet(OUT_DIR / "event_metrics.parquet", index=False)
    pd.DataFrame(wides).to_parquet(OUT_DIR / "events_wide.parquet", index=False)
    pd.DataFrame(list(processes.values())).to_parquet(OUT_DIR / "processes.parquet", index=False)
    pd.DataFrame(roles).to_parquet(OUT_DIR / "process_roles.parquet", index=False)

    print(f"✅ Parsed {len(events)} events and {len(metrics)} metrics.")
    print(f"📦 Parquet files saved in: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
