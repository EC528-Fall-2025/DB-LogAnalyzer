import duckdb
import numpy as np
import pandas as pd
from pathlib import Path

def _lower_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]
    return df

def _standardize_events(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize events to guaranteed columns:
    event_id, ts, event, severity, process, role, pid, machine_id, address,
    trace_file, src_line, raw_json_str, fields_json_str
    """
    df = _lower_cols(df).copy()

    # ts
    ts_col = next((c for c in ["ts", "timestamp", "time"] if c in df.columns), None)
    if ts_col is None:
        df["ts"] = pd.RangeIndex(start=0, stop=len(df))
    elif ts_col != "ts":
        df["ts"] = df[ts_col]

    # event/type
    ev_col = next((c for c in ["event", "type", "eventtype", "name"] if c in df.columns), None)
    if ev_col is None:
        df["event"] = "UnknownEvent"
    elif ev_col != "event":
        df["event"] = df[ev_col]

    # optional passthroughs
    for k in ["severity", "process", "role", "pid", "machine_id", "address", "trace_file", "src_line"]:
        if k not in df.columns:
            df[k] = None

    # JSON to string if present
    df["raw_json_str"] = df["raw_json"].astype(str) if "raw_json" in df.columns else None
    df["fields_json_str"] = df["fields_json"].astype(str) if "fields_json" in df.columns else None

    # event_id (synthesize if missing)
    if "event_id" not in df.columns:
        df = df.reset_index(drop=True)
        df["event_id"] = df.index.map(lambda i: f"row_{i}")

    want = [
        "event_id","ts","severity","event","process","role","pid","machine_id",
        "address","trace_file","src_line","raw_json_str","fields_json_str"
    ]
    df = df[want]
    return df

def _normalize_ts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure df['ts'] is a timezone-naive pandas datetime.
    Handles string and numeric epoch (s/ms/us/ns).
    """
    s = df["ts"]
    # Already datetime?
    if np.issubdtype(s.dtype, np.datetime64):
        return df

    # Try parse as string
    try:
        parsed = pd.to_datetime(s, errors="raise", utc=False, infer_datetime_format=True)
        df["ts"] = parsed
        return df
    except Exception:
        pass

    # Numeric epochs
    s_num = pd.to_numeric(s, errors="coerce")
    if s_num.notna().any():
        mx = float(np.nanmax(s_num))
    else:
        mx = 0.0
    if mx < 1e11:
        unit = "s"    # seconds
    elif mx < 1e14:
        unit = "ms"   # milliseconds
    elif mx < 1e17:
        unit = "us"   # microseconds
    else:
        unit = "ns"   # nanoseconds

    df["ts"] = pd.to_datetime(s_num, unit=unit, errors="coerce")

    # Fallback to monotonic if all NaT
    if df["ts"].isna().all():
        df = df.reset_index(drop=True)
        df["ts"] = pd.to_datetime(df.index, unit="s")

    return df

def _table_exists(con: duckdb.DuckDBPyConnection, name: str) -> bool:
    return con.execute(
        "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema='main' AND table_name=?",
        [name],
    ).fetchone()[0] > 0

def load_events_df(db_path: str) -> pd.DataFrame:
    """
    Load from DuckDB and robustly left-join optional tables (events_wide, event_metrics).
    Works even if events.event_id is missing (synthesizes one).
    """
    if not Path(db_path).exists():
        raise FileNotFoundError(f"DuckDB not found: {db_path}")

    con = duckdb.connect(db_path, read_only=True)

    # events (required)
    if not _table_exists(con, "events"):
        raise RuntimeError("Required table 'events' not found.")
    ev = con.sql("SELECT * FROM events").df()
    ev = _standardize_events(ev)
    ev = _normalize_ts(ev)
    ev = ev.sort_values("ts", kind="stable").reset_index(drop=True)

    # events_wide (optional)
    if _table_exists(con, "events_wide"):
        ew = _lower_cols(con.sql("SELECT * FROM events_wide").df())
        if "event_id" in ew.columns:
            ev = ev.merge(ew, on="event_id", how="left", suffixes=("", "_w"))

    # event_metrics (optional) → collapse to list of "name=value"
    if _table_exists(con, "event_metrics"):
        em = _lower_cols(con.sql("SELECT * FROM event_metrics").df())
        if {"event_id", "metric_name", "metric_value"}.issubset(em.columns):
            em["kv"] = em.apply(
                lambda r: f"{r['metric_name']}={r['metric_value'] if pd.notna(r['metric_value']) else 'NULL'}",
                axis=1,
            )
            emg = em.groupby("event_id")["kv"].apply(list).reset_index().rename(columns={"kv": "metrics_kv"})
            ev = ev.merge(emg, on="event_id", how="left")

    if "metrics_kv" not in ev.columns:
        ev["metrics_kv"] = None

    return ev
