from typing import List, Tuple
from datetime import timedelta
from langchain_core.documents import Document

# Recovery markers we care about
RECOVERY_MARKERS = {
    "MasterRecoveryState",
    "RecruitingTransactionServers",
    "RecoveryTransactionVersion",
    "RecoveryState",
}

# Startup/spammy events to drop BEFORE chunking
# (We intentionally DO NOT drop FileOpenError now.)
NOISY_EVENTS = {
    "BuggifySection",
    "ProgramStart",
    "Simulation",
    "Knob",
    "TraceEventMetrics",  # keep if present
}

# Keep events with severity >= 10 (relaxed from 20)
MIN_SEVERITY = 10


def _filter_noisy(docs: List[Document]) -> List[Document]:
    """Drop low-signal events and low severity before chunking."""
    out: List[Document] = []
    for d in docs:
        ev = (d.metadata.get("event") or "")
        if ev in NOISY_EVENTS:
            continue
        sev = d.metadata.get("severity")
        if sev is not None:
            try:
                if int(sev) < MIN_SEVERITY:
                    continue
            except Exception:
                # if severity is non-numeric, keep it
                pass
        out.append(d)
    return out


def collapse_consecutive_duplicates(docs: List[Document], max_annotate: int = 999) -> List[Document]:
    """
    Merge runs of identical page_content into one with a suffix [xN].
    Keeps metadata from the first doc in the run.
    """
    if not docs:
        return docs
    out: List[Document] = []
    i, n = 0, len(docs)
    while i < n:
        j = i + 1
        line = docs[i].page_content
        while j < n and docs[j].page_content == line:
            j += 1
        count = j - i
        base = docs[i]
        if count > 1:
            c = min(count, max_annotate)
            out.append(Document(page_content=f"{line}  [x{c}]", metadata=base.metadata))
        else:
            out.append(base)
        i = j
    return out


def chunk_by_recovery_markers(docs: List[Document], max_chunk_size: int = 80) -> List[Document]:
    """Start a new chunk at recovery markers; label chunk type accurately."""
    chunks: List[Document] = []
    buf: List[Document] = []

    def is_marker(doc: Document) -> bool:
        return (doc.metadata.get("event") or "").strip() in RECOVERY_MARKERS

    def flush():
        if not buf:
            return
        text = "\n".join(d.page_content for d in buf)
        meta = dict(buf[0].metadata)
        meta["chunk_size"] = len(buf)
        meta["start_ts"] = buf[0].metadata.get("ts")
        meta["end_ts"] = buf[-1].metadata.get("ts")
        meta["chunk_type"] = "recovery_segment" if is_marker(buf[0]) else "stream_window"
        chunks.append(Document(page_content=text, metadata=meta))

    for d in docs:
        evt = (d.metadata.get("event") or "").strip()
        if evt in RECOVERY_MARKERS and buf:
            flush()
            buf = [d]  # start new chunk with marker line
        else:
            buf.append(d)

        if len(buf) >= max_chunk_size:
            flush()
            buf = []

    flush()
    return chunks


def chunk_by_time_window(docs: List[Document], window_seconds: float = 3.0, max_chunk_size: int = 60) -> List[Document]:
    """
    Group consecutive docs whose timestamps are within a moving time window.
    Prevents giant runs at identical timestamps.
    """
    if not docs:
        return []

    try:
        from pandas import Timestamp
    except Exception:
        Timestamp = None  # type: ignore

    chunks: List[Document] = []
    buf: List[Document] = []
    buf_start_ts = None

    def to_ts(val):
        if Timestamp is None:
            return None
        try:
            return val if isinstance(val, Timestamp) else Timestamp(val)
        except Exception:
            return None

    def flush():
        nonlocal buf
        if not buf:
            return
        text = "\n".join(d.page_content for d in buf)
        meta = dict(buf[0].metadata)
        meta["chunk_type"] = "time_window"
        meta["chunk_size"] = len(buf)
        meta["start_ts"] = buf[0].metadata.get("ts")
        meta["end_ts"] = buf[-1].metadata.get("ts")
        chunks.append(Document(page_content=text, metadata=meta))
        buf = []

    for d in docs:
        ts = to_ts(d.metadata.get("ts"))
        if not buf:
            buf = [d]
            buf_start_ts = ts
            continue

        within_window = False
        if ts is not None and buf_start_ts is not None:
            try:
                within_window = (ts - buf_start_ts) <= timedelta(seconds=window_seconds)
            except Exception:
                within_window = False

        if (not within_window) or (len(buf) >= max_chunk_size):
            flush()
            buf = [d]
            buf_start_ts = ts
        else:
            buf.append(d)

    flush()
    return chunks


def chunk_fixed_window(docs: List[Document], window: int = 30) -> List[Document]:
    """Simple fixed-size chunking as a last resort."""
    out: List[Document] = []
    for i in range(0, len(docs), window):
        block = docs[i:i + window]
        if not block:
            continue
        text = "\n".join(d.page_content for d in block)
        meta = dict(block[0].metadata)
        meta["chunk_type"] = "fixed_window"
        meta["chunk_size"] = len(block)
        meta["start_ts"] = block[0].metadata.get("ts")
        meta["end_ts"] = block[-1].metadata.get("ts")
        out.append(Document(page_content=text, metadata=meta))
    return out


def choose_log_chunks(docs: List[Document]) -> Tuple[List[Document], str]:
    """
    Strategy:
      0) collapse duplicate lines
      1) recovery markers
      2) 3-second time window
      3) fixed window
    """
    # 0) compact repeated lines first
    docs = collapse_consecutive_duplicates(docs)

    # 1) try recovery-aware
    by_recovery = chunk_by_recovery_markers(docs, max_chunk_size=80)
    if len(by_recovery) > 1:
        return by_recovery, "recovery_segment"

    # 2) time-window fallback (wider window, bigger chunks)
    by_time = chunk_by_time_window(docs, window_seconds=3.0, max_chunk_size=60)
    if len(by_time) > 1:
        return by_time, "time_window"

    # 3) last resort
    return chunk_fixed_window(docs, window=30), "fixed_window"
