from typing import List, Dict, Any
import pandas as pd
from langchain_core.documents import Document

# key numeric columns we’ll surface in the text summary
_FIELDS_TO_SUMMARIZE = [
    "grv_latency_ms","txn_volume","queue_bytes","durability_lag_s",
    "data_move_in_flight","disk_queue_bytes","kv_ops"
]

def _summarize_metrics(row: Dict[str, Any]) -> str:
    bits = []
    for k in _FIELDS_TO_SUMMARIZE:
        if k in row and pd.notna(row[k]):
            bits.append(f"{k}={row[k]}")
    mkv = row.get("metrics_kv")
    if isinstance(mkv, list):
        bits.extend(mkv[:8])  # cap verbosity
    elif mkv:
        bits.append(str(mkv))
    return " | ".join(bits)

def row_to_text(r: Dict[str, Any]) -> str:
    ts   = r.get("ts")
    sev  = r.get("severity")
    evt  = r.get("event")
    role = r.get("role")
    proc = r.get("process")
    addr = r.get("address")
    base = f"[{ts}] {evt} (sev={sev}, role={role}, proc={proc}, addr={addr})"
    metrics = _summarize_metrics(r)
    return base + (f" :: {metrics}" if metrics else "")

def logs_df_to_documents(df: pd.DataFrame, source_file: str) -> List[Document]:
    docs: List[Document] = []
    cols = df.columns.tolist()
    for _, row in df.iterrows():
        d = row.to_dict()
        text = row_to_text(d)
        meta = {c: (None if pd.isna(d[c]) else d[c]) for c in cols}
        meta["source_file"] = source_file
        docs.append(Document(page_content=text, metadata=meta))
    return docs
