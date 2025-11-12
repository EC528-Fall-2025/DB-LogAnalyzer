# service/forced_recovery_chunker.py
from __future__ import annotations
import re
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
from datetime import datetime

ISO_FMT = "%Y-%m-%dT%H:%M:%SZ"

# ---- Tiny event model -------------------------------------------------
@dataclass
class ParsedEvent:
    event_id: str
    time: float
    dt: datetime
    type: str
    severity: Optional[int]
    fields: Dict[str, Any]

# ---- Tiny parser that handles one-Event-per-line XML ------------------
class SimpleLogParser:
    _attr_re = re.compile(r'(\w+)="([^"]*)"')
    _event_tag = re.compile(r"<Event\b[^>]*>")

    def parse_logs(self, path: str) -> List[ParsedEvent]:
        out: List[ParsedEvent] = []
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if "<Event" not in line:
                    continue
                if not self._event_tag.search(line):
                    continue
                attrs = dict(self._attr_re.findall(line))
                evt_type = attrs.get("Type") or attrs.get("type") or "Unknown"
                dt_str = attrs.get("DateTime")
                t = float(attrs.get("Time", "0")) if "Time" in attrs else 0.0
                sev = int(attrs["Severity"]) if "Severity" in attrs and attrs["Severity"].isdigit() else None
                # basic normalization
                try:
                    dt = datetime.strptime(dt_str, ISO_FMT) if dt_str else datetime.utcfromtimestamp(0)
                except Exception:
                    dt = datetime.utcfromtimestamp(0)
                ev_id = attrs.get("ID") or attrs.get("Id") or f"{len(out)}"
                # keep everything else as fields
                fields = {k: v for k, v in attrs.items() if k not in ["Type","DateTime","Time","Severity","ID","Id"]}
                out.append(ParsedEvent(event_id=ev_id, time=t, dt=dt, type=evt_type, severity=sev, fields=fields))
        return out

# ---- Chunk representation ---------------------------------------------
@dataclass
class RecoveryChunk:
    index: int
    start_time: str
    end_time: str
    start_type: str
    complete: bool
    event_count: int
    start_comment: Optional[str] = None
    events: Optional[List[ParsedEvent]] = None

    def to_dict(self, include_events: bool = False) -> Dict[str, Any]:
        d = {
            "index": self.index,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "start_type": self.start_type,
            "complete": self.complete,
            "event_count": self.event_count,
            "start_comment": self.start_comment or "",
        }
        if include_events and self.events:
            d["events"] = [self._ev_to_dict(e) for e in self.events]
        return d

    @staticmethod
    def _ev_to_dict(e: ParsedEvent) -> Dict[str, Any]:
        return {
            "event_id": e.event_id,
            "datetime": e.dt.strftime(ISO_FMT),
            "type": e.type,
            "severity": e.severity,
            "time": e.time,
            "fields": e.fields,
        }

# ---- The chunker ------------------------------------------------------
class ForcedRecoveryChunker:
    """
    Start a chunk when we see a 'forced-recovery-ish' trigger.
    End a chunk when we see MasterRecoveryState StatusCode=14 (recovery complete).
    If no explicit end is found, close the chunk at the last seen event so far (incomplete=False).
    """

    # Heuristics for starts and ends (adjust as you learn more)
    START_TYPES = {
        "ForceRecovery", "RecoveryTriggered", "MasterTerminated",
        "CommitProxyFailure", "CommitProxyFailed", "ResolverFailure",
        "TLogDegraded", "ExcludeFailedTLog", "CoordinatedConflict",
        "CoordinatedRecovery", "RecoveringFrom"
    }
    END_TYPE = "MasterRecoveryState"   # with fields StatusCode="14"
    END_STATUS = "14"

    def __init__(self, knowledge_base_path: Optional[str] = None):
        self.parser = SimpleLogParser()
        self.knowledge_base_path = knowledge_base_path  # not used here but kept for API parity

    def _is_start(self, e: ParsedEvent) -> bool:
        if e.type in self.START_TYPES:
            return True
        # soft signals that often precede a forced recovery
        if e.type in {"TLogMetrics"} and e.fields.get("Degraded","").lower() in {"1","true"}:
            return True
        return False

    def _is_end(self, e: ParsedEvent) -> bool:
        if e.type == self.END_TYPE:
            sc = e.fields.get("StatusCode") or e.fields.get("status") or ""
            return sc == self.END_STATUS
        # sometimes MasterServer reappears + MasterMetrics after crash — treat as end-ish
        if e.type in {"MasterMetrics", "MasterServer"}:
            return True
        return False

    def chunk_events(self, events: List[ParsedEvent]) -> List[RecoveryChunk]:
        chunks: List[RecoveryChunk] = []
        cur_buf: List[ParsedEvent] = []
        cur_start: Optional[ParsedEvent] = None
        idx = 0

        for e in sorted(events, key=lambda x: (x.dt, x.time)):
            # open new chunk on start
            if cur_start is None and self._is_start(e):
                cur_start = e
                cur_buf = [e]
                continue

            # buffer while in chunk
            if cur_start is not None:
                cur_buf.append(e)
                if self._is_end(e):
                    idx += 1
                    chunks.append(
                        RecoveryChunk(
                            index=idx,
                            start_time=cur_start.dt.strftime(ISO_FMT),
                            end_time=e.dt.strftime(ISO_FMT),
                            start_type=cur_start.type,
                            complete=True,
                            event_count=len(cur_buf),
                            start_comment=cur_start.fields.get("Comment") if cur_start.fields else None,
                            events=cur_buf,
                        )
                    )
                    cur_start = None
                    cur_buf = []
                continue

            # not in a chunk: keep scanning

        # trailing incomplete chunk
        if cur_start is not None and cur_buf:
            idx += 1
            last = cur_buf[-1]
            chunks.append(
                RecoveryChunk(
                    index=idx,
                    start_time=cur_start.dt.strftime(ISO_FMT),
                    end_time=last.dt.strftime(ISO_FMT),
                    start_type=cur_start.type,
                    complete=False,
                    event_count=len(cur_buf),
                    start_comment=cur_start.fields.get("Comment") if cur_start.fields else None,
                    events=cur_buf,
                )
            )
        return chunks
