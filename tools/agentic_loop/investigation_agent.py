"""
Investigation Agent - LLM-based log analysis with tool access.

Purpose: Analyze DuckDB events using LLM and investigation tools to identify root causes.
"""


import json
import os
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any

import google.generativeai as genai

from data_transfer_object.event_dto import EventModel
from tools.database import get_conn
from tools.investigation_tools import (
    GlobalScanner,
    HotspotSelector,
    ContextAnalyzer,
    Detectors,
    TimelineBuilder,
)
from tools.rag.query_formatter import build_rag_query
from tools.rag.rag_client import RAGClient
from tools.agentic_loop.llm_input_logger import write_llm_input, write_llm_output

# FoundationDB recovery cluster knowledge base (used as static context alongside RAG).
FDB_KNOWLEDGE_BASE = """
# FoundationDB Recovery Cluster Knowledge Base

## CLUSTER 0: recovery_restart_cascade
Problem: Recovery loops, repeatedly restarting before completion due to coordinator state conflicts.
Causes: Concurrent recovery attempts, coordinator generation mismatches, commit_unknown_result ambiguity, incomplete previous recovery leaving stale metadata, clock skew causing epoch conflicts.
Key Indicators: Repeated "reading_coordinated_state" retries, coordinator mismatch errors, generation conflicts, "Aborting current recovery" loops.

## CLUSTER 4: transaction_tag_throttling
Problem: Tag throttling active during recovery blocks GRV requests, stalling recovery progress.
Causes: Tag budget exhausted (especially 'system' tag), rate limits too restrictive for recovery burst, leftover throttle state from previous load, GRV queue buildup (>10K entries).
Key Indicators: TAG_THROTTLED responses, tag budget exhausted, high GRV queue lengths, GRV timeouts (>5s), recovery slowed by throttling.

## CLUSTER 5: tlog_failure_recovery
Problem: TLog failure/degradation forces recovery due to inability to persist commits safely.
Causes: TLog crash, disk full, severe I/O degradation (>500ms latency), memory overflow, network partition isolating TLog, partial commits (only some TLogs acked).
Key Indicators: TLog failed/degraded messages, high disk latency, partial commit warnings, TLog excluded, "stopped by CC recovery", acknowledge timeouts.

## CLUSTER 6: storage_engine_pressure
Problem: Post-recovery memory/disk pressure reduces throughput significantly.
Causes: Recovery-triggered read surge, memory exceeding limits, disk write buffer overflow, DiskQueue durability lag, switching to large transaction mode, catch-up phase overload, I/O saturation (>95%).
Key Indicators: "KeyValueStoreMemory switching to large txn", memory >100%, DiskQueue not durable, high fsync latency (>1000ms), throughput reduction (50-70%), occurring AFTER recovery completion.

## CLUSTER 7: commit_proxy_pipeline_crash
Problem: Commit proxy or resolver crash breaks pipeline; master terminates, forcing recovery.
Causes: Process crash (segfault, OOM), deadlock/hang, connection loss between proxy-resolver, progress timeout, state machine error, bug in commit logic.
Key Indicators: Proxy/resolver crash or unresponsive (>30s), "failed to progress", pipeline broken, master terminated due to proxy failure, ClusterController detects master failure.

## CLUSTER 8: network_partition_recovery
Problem: Network partition/connectivity loss isolates components, prevents quorum, causes split-brain.
Causes: Switch/router failure, sustained packet loss (>10%), firewall misconfiguration, DNS failures, network congestion, asymmetric failures, coordinator quorum lost.
Key Indicators: Coordinator unreachable, connection timeouts/failures, quorum lost, network unreachable, split-brain detected, multiple components reporting communication failures simultaneously.

## CLUSTER 9: configuration_change_recovery
Problem: Configuration change (add/remove nodes, replication changes) triggers instability requiring recovery.
Causes: Exclude/include operations during recovery, replication factor changes, partial config updates, version mismatches during rolling upgrade, locality conflicts, coordinator set changes.
Key Indicators: Exclude/include operations logged, configuration change in progress, redundancy mode change, version mismatch, data movement during recovery, coordinator set updates.

## CLUSTER 10: clock_skew_recovery
Problem: Clock differences between nodes cause version ordering violations, lease confusion, timeout inconsistencies.
Causes: NTP failure/misconfiguration, leap second handling differences, VM time sync issues, manual clock adjustments, backward time jumps, clock drift exceeding tolerance (>1s).
Key Indicators: Clock skew/drift detected, NTP sync failures, "time jumped backward", version ordering violations, lease expiration anomalies, timestamps out of order, timing inconsistencies.

## CLUSTER 11: data_corruption_recovery
Problem: Data corruption detected in storage/logs/metadata forces recovery to restore from replicas.
Causes: Bit rot (silent data corruption), filesystem corruption, memory errors (bad RAM), storage engine bugs, incomplete writes from power failure, checksum mismatches, TLog/coordinator state corruption.
Key Indicators: Checksum mismatch, CRC errors, corruption detected, validation failures, filesystem errors, TLog replay errors, rebuilding/discarding corrupted data messages.

## CLUSTER 12: resource_exhaustion_recovery
Problem: OS-level resource exhaustion (file descriptors, ports, threads) prevents normal operation.
Causes: File descriptor limit hit (ulimit), ephemeral port exhaustion (all 64K ports used), thread creation failures, connection pool exhausted, cgroup limits, kernel OOM, socket buffer exhaustion.
Key Indicators: "Too many open files", "Cannot allocate memory" (system level), port exhaustion, thread creation failed, connection refused (resource limit), ENOMEM/EMFILE/EAGAIN errors.

## CLUSTER 13: workload_spike_recovery
Problem: Sudden workload surge overwhelms cluster, causing queue overflows and timeout cascades.
Causes: Retry storm (thundering herd), bulk data operations, badly behaving client, cache warming read amplification, seasonal traffic spike, background job overload, cascading timeouts.
Key Indicators: Traffic surge/spike, high load warnings, queue overflow, commit/read rate spikes, retry storm detected, timeout cascades, rejecting requests due to overload, sudden metric degradation.

## CLUSTER 14: upgrade_rollback_recovery
Problem: Software upgrade fails, introduces bugs, or creates version incompatibilities requiring rollback.
Causes: Protocol version incompatibility, data format changes, new version bugs, incomplete/mixed version upgrade, rollback incompatibility, feature flag conflicts, performance regression, strict validation rejecting old data.
Key Indicators: Version mismatch, protocol incompatibility, upgrade/rollback in progress, mixed versions detected, data format migration errors, validation failures on legacy data.

## CLUSTER 15: cascading_failure_recovery
Problem: One component failure triggers chain reaction of failures across cluster, causing widespread instability.
Causes: Storage server failure redistributing load overwhelming others, coordinator failure causing reconnection storm, shared dependency failure (DNS, NTP), bug affecting all processes, resource exhaustion spreading, network affecting multiple machines.
Key Indicators: Multiple different components failing rapidly in succession, load redistribution failures, timeout cascades across components, simultaneous failures on different machines, avalanche pattern (1→2→4 failures).

## CLUSTER 16: lease_expiration_recovery
Problem: Master lease expires or renewal fails, requiring new master election and recovery.
Causes: Network issues preventing renewal, clock skew making lease appear expired, master paused too long (GC, scheduling), coordinator quorum unavailable, master too busy to renew, deadlock preventing renewal thread.
Key Indicators: Lease expiration warnings, failed to renew lease, master lease loss, coordinator unavailable during renewal, master election due to lease loss, pause/GC events exceeding lease timeout.

## QUICK REFERENCE
Temporal Patterns:
- Before recovery: Clusters 5,7,8,13,15
- During recovery: Clusters 0,4,9,10
- After recovery: Cluster 6
- During operations: Clusters 9,14,16

Component Focus:
- Coordinator: 0,8,10,16
- TLog: 5,11
- Proxy: 4,7
- Storage: 6,11
- System-wide: 8,12,13,15
"""


# Suppress FutureWarning from google.api_core (dependency of google-genai)
warnings.filterwarnings('ignore', category=FutureWarning, module='google.api_core')

# Workaround for importlib.metadata.packages_distributions compatibility
try:
    import importlib.metadata as _metadata
    if not hasattr(_metadata, 'packages_distributions'):
        def _packages_distributions():
            """Fallback for packages_distributions if not available."""
            return {}
        _metadata.packages_distributions = _packages_distributions
except (ImportError, AttributeError):
    pass


def _ensure_dict(maybe_json) -> dict:
    """Small local helper for load_events_window."""
    if maybe_json is None:
        return {}
    if isinstance(maybe_json, dict):
        return maybe_json
    if isinstance(maybe_json, str):
        try:
            return json.loads(maybe_json)
        except Exception:
            return {"_raw": maybe_json}
    return {}


def _parse_event_row_local(row) -> EventModel:
    """Parse a database row into EventModel (local copy for load_events_window)."""
    rid, ts, sev, event, role, fields = row
    fdict = _ensure_dict(fields)

    if ts and not isinstance(ts, datetime):
        ts = datetime.fromtimestamp(ts)

    raw_blob = {
        "id": str(rid),
        "event": event,
        "ts": ts.isoformat() if ts else None,
        "severity": sev,
        "role": role,
        "fields_json": fdict,
    }

    # event_id can be any stable int; just hash the id string for load_events_window
    try:
        import uuid
        u = uuid.UUID(str(rid))
        event_id = u.int & ((1 << 63) - 1)
    except Exception:
        event_id = 0

    return EventModel(
        event=event,
        ts=ts,
        severity=sev,
        role=role,
        fields_json=fdict,
        event_id=event_id,
        process=role,
        pid=None,
        machine_id=None,
        address=None,
        trace_file=None,
        src_line=None,
        raw_json=raw_blob,
    )


def load_events_window(
    db_path: str,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    limit: Optional[int] = None
) -> List[EventModel]:
    """
    Load events from database within a time window.

    This no longer uses get_events (removed in new tools). It directly queries DuckDB.
    """
    con = get_conn(db_path)
    limit = limit or 1000

    conditions = []
    params: List[Any] = []

    if start_time:
        conditions.append("ts >= ?")
        params.append(start_time)
    if end_time:
        conditions.append("ts <= ?")
        params.append(end_time)

    where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""
    sql = f"""
        SELECT event_id, ts, severity, event, role, fields_json
        FROM events
        {where_clause}
        ORDER BY ts ASC
        LIMIT ?
    """
    params.append(limit)

    rows = con.execute(sql, params).fetchall()
    return [_parse_event_row_local(r) for r in rows]


@dataclass
class InvestigationResult:
    """Result from investigation."""
    hypothesis: str
    confidence: float
    reasoning: str
    tools_used: List[str]
    evidence_events: List[EventModel]


LLM_CONTEXT_CHAR_LIMIT = 120000
ADDITIONAL_DATA_MAX_ITEMS = 20


class InvestigationAgent:
    """Agent that analyzes log events using LLM and unified investigation tools."""

    def __init__(
        self,
        db_path: str,
        max_iterations: int = 10,
        confidence_threshold: float = 0.8,
        max_llm_calls: int = 4,
        use_rag: Optional[bool] = None,
        rag_corpus: Optional[str] = None
    ):
        """
        Initialize investigation agent.

        Args:
            db_path: Path to DuckDB database
            max_iterations: Maximum number of investigation iterations
            confidence_threshold: Confidence level required to stop investigation
            max_llm_calls: Max Gemini invocations allowed per investigation
            use_rag: Enable RAG retrieval (defaults to True if RAG_CORPUS_RESOURCE env is set)
            rag_corpus: Override RAG corpus resource path
        """
        self.db_path = db_path
        self.max_iterations = max_iterations
        self.confidence_threshold = confidence_threshold
        self.max_llm_calls = max(1, max_llm_calls)
        self.use_rag = use_rag if use_rag is not None else bool(
            os.getenv("RAG_CORPUS_RESOURCE") or os.getenv("RAG_CORPUS")
        )
        self.rag_corpus = rag_corpus
        self.rag_client: Optional[RAGClient] = None
        self.timeline_summary: Optional[Dict[str, Any]] = None
        self.timeline_builder = TimelineBuilder(db_path)
        self._last_det_results: Dict[str, Any] = {}

        # Unified tools
        self.scanner = GlobalScanner(db_path)
        self.hotspots = HotspotSelector(db_path)
        self.ctx = ContextAnalyzer(db_path)
        self.detectors = Detectors(db_path)

    # ======================================================
    # TOOLING / LLM HELPERS
    # ======================================================

    def _get_tools_documentation(self) -> str:
        """Describe only the NEW unified tools to the LLM."""
        return """
AVAILABLE INVESTIGATION TOOLS (unified):

GLOBAL VIEW (GlobalScanner):
- global_summary()
  → High-level cluster state: max severity, severity distribution, histograms, time span.
- severity_counts()
  → Count of events by severity level.
- event_histogram(limit=50)
  → Top event types with counts.
- time_span()
  → Earliest / latest timestamps and duration.
- top_events(severity_min=40, limit=50)
  → Most severe events, used as anchors.
- bucket_heatmap(bucket_seconds=300, limit=100)
  → Time buckets with max severity and event counts.
- rollback()
  → Detect CommittedVersion/DurableVersion drops, version resets, and recovery version regressions.

HOTSPOT SELECTION (HotspotSelector):
- high_severity_buckets(min_severity=20, bucket_seconds=600, limit=20)
  → Find time buckets with high severity activity.
- get_uncovered(inspected_buckets, min_severity=20, bucket_seconds=600)
  → Find remaining buckets that haven't been inspected yet.

LOCAL CONTEXT (ContextAnalyzer):
- context_window(around_time, window_seconds=30, limit=200)
  → Get events around a specific timestamp.
- similar_events(event_type, limit=10)
  → Find recent events of a similar type.

CORE DETECTORS (Detectors):
- storage_engine_pressure(lag_threshold=50000)
  → Detect VersionLag spikes (storage pressure).
- ratekeeper_throttling()
  → Detect Ratekeeper/Throttle related events.
- missing_tlogs()
  → Detect missing/failed/error TLogs.
- recovery_loop(threshold=3, window_seconds=60)
  → Detect repeated recovery loops.
- coordination_loss()
  → Detect coordinator-related failures/loss.
Important: DO NOT assume that TLog errors, injected faults, or io_error
(1510) automatically indicate a TLogFailure root cause.

Write your analysis as a narrative Root Cause Analysis (RCA) story.

- Focus on the unfolding sequence of events, not just pattern matching.
- Build a causal chain: what happened first, what that caused next, and how the system reacted.
- Explain the interaction between components (TLogs, storage servers, recovery, ratekeeper).
- Treat detectors, metrics, and trace events as clues in the story, not as absolute ground truth.
- Avoid anchoring on any single error type; instead, reason about the broader system behavior.
- Identify primary causes vs. cascading secondary effects.
- If injected faults appear (`ErrorIsInjectedFault`), describe them as part of the simulation mechanism, unless they clearly precede all other failures.
- Use phrases like “Initially…”, “This triggered…”, “As a result…”, “Eventually…”, “This led to…” to maintain narrative flow.
- The goal is to reconstruct what the cluster experienced from its own perspective.

When generating explanations, do not anchor on injected faults 
(`ErrorIsInjectedFault: 1`, `io_error`, `SharedTLogFailed`) unless 
they occur BEFORE any signs of storage pressure or recovery churn.

Injected I/O faults in clog/rollback workloads are mechanisms, not 
primary causes. Explanations must reflect the correct causal order.

Explanations must describe the timeline and causal order rather 
than over-interpreting isolated injected-fault events.
"""

    def _append_events(
        self,
        destination: List[EventModel],
        events: Optional[List[EventModel]]
    ) -> bool:
        """Append events and indicate whether new context exists."""
        if not events:
            return False
        destination.extend(events)
        return True

    def _summarize_data_for_llm(self, data: Any, max_items: int = ADDITIONAL_DATA_MAX_ITEMS) -> Any:
        """Reduce large tool outputs to compact summaries for the LLM."""
        if isinstance(data, list):
            length = len(data)
            if length > max_items:
                return {
                    "type": "list",
                    "length": length,
                    "sample": data[:max_items],
                }
            return data
        if isinstance(data, dict):
            items = list(data.items())
            if len(items) > max_items:
                trimmed = dict(items[:max_items])
                trimmed["__summary__"] = f"Trimmed to {max_items} of {len(items)} entries"
                return trimmed
            return data
        return data

    def _record_additional_data(
        self,
        destination: List[tuple],
        tool_name: str,
        data: Any
    ) -> bool:
        """Store auxiliary tool output and mark context as updated."""
        summarized = self._summarize_data_for_llm(data)
        destination.append((tool_name, summarized))
        return True

    def _enforce_context_limit(self, text: str, limit: int = LLM_CONTEXT_CHAR_LIMIT) -> str:
        """Ensure the prompt stays within the allowable token/char budget."""
        if len(text) <= limit:
            return text
        truncated = text[:limit]
        note = f"\n\n[Context truncated to {limit} characters to satisfy token limits.]"
        return truncated + note

    def _summarize_for_rag(self, query_text: str, api_key: Optional[str]) -> Optional[str]:
        """
        Use a lightweight LLM pass to compress the detector/timeline text for RAG retrieval.
        Returns a summarized query or None if summarization failed.
        """
        if not query_text:
            return None
        try:
            genai.configure(api_key=api_key or os.getenv("GEMINI_API_KEY"))
            model = genai.GenerativeModel("gemini-2.5-flash")
            prompt = (
                "Summarize the following detector/timeline evidence into a concise retrieval query (<=800 chars). "
                "Keep explicit problem names, metrics, and any cluster IDs. Output plain text, bullet-style."
            )
            resp = model.generate_content([prompt, query_text])
            text = (resp.text or "").strip()
            if text:
                return text
        except Exception as e:
            print(f"   RAG summarizer failed, using raw query. Error: {e}")
        return None

    def _run_rag_retrieval(
        self,
        det_results: Dict[str, Any],
        timeline: Dict[str, Any],
        additional_data: List[tuple],
        api_key: Optional[str],
        tools_used: Optional[List[str]] = None
    ) -> bool:
        """Query the RAG corpus with detector/timeline summary."""
        if not self.use_rag:
            return False

        if not self.rag_client:
            try:
                self.rag_client = RAGClient(
                    api_key=api_key,
                    corpus_resource=self.rag_corpus
                )
            except Exception as e:
                print(f"   RAG disabled: {e}")
                self._record_additional_data(additional_data, "rag_error", str(e))
                self.use_rag = False
                return False

        query_text_raw = build_rag_query(det_results or {}, timeline or {}, self.timeline_summary)
        summarized_query = self._summarize_for_rag(query_text_raw, api_key)
        if not summarized_query:
            msg = "RAG summarizer unavailable; skipping RAG retrieval to avoid sending raw detector/timeline text."
            print(f"   {msg}")
            self._record_additional_data(additional_data, "rag_error", msg)
            return False
        query_text = summarized_query
        print("\nQuerying RAG corpus with detector summary...")
        print("   Using LLM-summarized RAG query.")
        try:
            print(f"   RAG query (summarized): {query_text}")
        except Exception:
            pass

        try:
            rag_response = self.rag_client.retrieve(query_text)

            rag_text_full = rag_response.get("text", "") if isinstance(rag_response, dict) else str(rag_response)
            rag_text = rag_text_full
            if len(rag_text) > 3000:
                rag_text = rag_text[:3000] + "... [truncated]"

            raw_chunks = []
            if isinstance(rag_response, dict):
                raw_chunks = rag_response.get("chunks", [])
                if len(raw_chunks) > 3:
                    raw_chunks = raw_chunks[:3] + ["... truncated chunks ..."]

            # Console preview so operators can see the fetched text (full) and chunk markers
            if rag_text_full:
                preview_full = rag_text_full.replace("\n", " ")
                print(f"   RAG text (full): {preview_full}")
            if raw_chunks:
                try:
                    labels = []
                    for ch in raw_chunks:
                        if isinstance(ch, dict):
                            # try to extract any obvious id/name markers
                            name = ch.get("name") or ch.get("id") or ch.get("displayName")
                            if not name:
                                if "candidates" in ch and ch["candidates"]:
                                    cand = ch["candidates"][0]
                                    name = cand.get("name") or cand.get("id")
                            labels.append(name or "chunk")
                        else:
                            labels.append(str(ch)[:80])
                    print(f"   RAG chunks: {labels}")
                except Exception:
                    print(f"   RAG chunks: {len(raw_chunks)} (unparsed)")

            self._record_additional_data(
                additional_data,
                "rag_retrieval",
                {
                    "query_summarized": summarized_query,
                    "response_text": rag_text,
                    "raw_chunks": raw_chunks,
                }
            )
            if tools_used is not None:
                tools_used.append("RAG.retrieve")
            print("   ✓ RAG response captured")
            return True
        except Exception as e:
            print(f"   RAG retrieval failed: {e}")
            self._record_additional_data(additional_data, "rag_error", str(e))
            return False

    # ======================================================
    # METRIC EXTRACTION & EVENT FORMATTING
    # ======================================================

    def _extract_metrics_from_events(self, events: List[EventModel]) -> Dict[str, Any]:
        """Extract critical metrics from events to prioritize over event severity."""
        metrics = {
            "version_lag_spikes": [],
            "negative_latencies": [],
            "slow_ss_loops": [],
            "throttling_reasons": [],
            "high_lag_timestamps": []
        }

        for event in events:
            if not event.fields_json:
                continue

            fields = event.fields_json

            # VersionLag and similar metrics
            version_lag = None
            for key in ["VersionLag", "versionLag", "VersionLagValue", "Lag", "lag"]:
                if key in fields:
                    try:
                        version_lag = float(fields[key])
                        break
                    except (ValueError, TypeError):
                        continue

            if version_lag is not None:
                if version_lag > 100000:
                    metrics["version_lag_spikes"].append({
                        "timestamp": event.ts.isoformat() if event.ts else "N/A",
                        "event_type": event.event,
                        "version_lag": version_lag,
                        "role": event.role,
                        "severity": event.severity
                    })
                    if version_lag > 1_000_000:
                        metrics["high_lag_timestamps"].append(event.ts)

            # Negative latencies
            for key in fields:
                key_lower = str(key).lower()
                if "latency" in key_lower or "min" in key_lower or "max" in key_lower:
                    try:
                        val = float(fields[key])
                        if val < 0:
                            metrics["negative_latencies"].append({
                                "timestamp": event.ts.isoformat() if event.ts else "N/A",
                                "event_type": event.event,
                                "metric": key,
                                "value": val,
                                "role": event.role
                            })
                    except (ValueError, TypeError):
                        continue

            # SlowSSLoop
            if "SlowSSLoop" in str(event.event) or "SlowSS" in str(event.event):
                metrics["slow_ss_loops"].append({
                    "timestamp": event.ts.isoformat() if event.ts else "N/A",
                    "event_type": event.event,
                    "severity": event.severity,
                    "fields": fields
                })

            # Throttling (Ratekeeper / RkUpdate)
            if "RkUpdate" in str(event.event) or "Ratekeeper" in str(event.event):
                for key, value in fields.items():
                    key_lower = str(key).lower()
                    if "throttle" in key_lower or "reason" in key_lower:
                        metrics["throttling_reasons"].append({
                            "timestamp": event.ts.isoformat() if event.ts else "N/A",
                            "event_type": event.event,
                            "reason": f"{key}: {value}",
                            "fields": fields
                        })

        return metrics

    def _format_events_for_llm(self, events: List[EventModel]) -> str:
        """Format events for LLM analysis, prioritizing metrics over severity."""
        if not events:
            return "No events found."

        metrics = self._extract_metrics_from_events(events)

        # Sort by severity 40+ first, then by time
        sorted_events = sorted(events, key=lambda e: (
            -(e.severity or 0) if (e.severity or 0) >= 40 else -1000,
            e.ts if e.ts else datetime.max
        ))

        severity_40_plus = [e for e in sorted_events if (e.severity or 0) >= 40]
        severity_30_info = [e for e in sorted_events if (e.severity or 0) == 30]
        severity_20_warnings = [e for e in sorted_events if (e.severity or 0) == 20]

        display_events = sorted_events[:20]
        total_count = len(events)

        lines: List[str] = []

        lines.append(f"Found {total_count} events:")
        lines.append(f"  - Severity 40+ (Errors): {len(severity_40_plus)}")
        lines.append(f"  - Severity 30 (Info): {len(severity_30_info)}")
        lines.append(f"  - Severity 20 (Warnings): {len(severity_20_warnings)}")
        lines.append("")

        lines.append("=" * 70)
        lines.append("🔴 CRITICAL: PRIORITIZE METRICS OVER EVENT SEVERITY")
        lines.append("=" * 70)
        lines.append("")
        lines.append("METRICS ARE MORE IMPORTANT THAN EVENT SEVERITY!")
        lines.append("   - VersionLag spikes (>100k, especially >1M) indicate storage pressure")
        lines.append("   - Negative latencies indicate timing bugs/overflows")
        lines.append("   - Throttling reasons show performance degradation")
        lines.append("   - SlowSSLoop indicates storage server performance issues")
        lines.append("")
        lines.append("❌ Do NOT treat Severity 20/30 events as root cause by themselves")
        lines.append("   Focus on the METRIC anomalies behind them.\n")

        # Metric anomalies first
        if metrics["version_lag_spikes"]:
            lines.append("🚨 VERSIONLAG SPIKES (Storage Engine Pressure):")
            lines.append(f"   Found {len(metrics['version_lag_spikes'])} events with VersionLag > 100k")
            critical_lags = [m for m in metrics["version_lag_spikes"] if m["version_lag"] > 1_000_000]
            if critical_lags:
                lines.append(f"   {len(critical_lags)} events with VersionLag > 1M (CRITICAL)")
                for lag in critical_lags[:5]:
                    lines.append(
                        f"      - {lag['timestamp']}: VersionLag={lag['version_lag']:.0f} "
                        f"(event: {lag['event_type']})"
                    )
            else:
                for lag in metrics["version_lag_spikes"][:5]:
                    lines.append(
                        f"      - {lag['timestamp']}: VersionLag={lag['version_lag']:.0f} "
                        f"(event: {lag['event_type']})"
                    )
            lines.append("")

        if metrics["negative_latencies"]:
            lines.append("🚨 NEGATIVE LATENCIES (Timing Bug/Overflow):")
            lines.append(f"   Found {len(metrics['negative_latencies'])} negative latency values")
            for neg in metrics["negative_latencies"][:5]:
                lines.append(
                    f"      - {neg['timestamp']}: {neg['metric']}={neg['value']} "
                    f"(event: {neg['event_type']})"
                )
            lines.append("")

        if metrics["slow_ss_loops"]:
            lines.append("🚨 SLOW SS LOOPS (Storage Server Performance):")
            lines.append(f"   Found {len(metrics['slow_ss_loops'])} SlowSSLoop events")
            for slow in metrics["slow_ss_loops"][:3]:
                lines.append(f"      - {slow['timestamp']}: {slow['event_type']}")
            lines.append("")

        if metrics["throttling_reasons"]:
            lines.append("🚨 THROTTLING DETECTED (Performance Degradation):")
            lines.append(f"   Found {len(metrics['throttling_reasons'])} throttling events")
            for throttle in metrics["throttling_reasons"][:5]:
                lines.append(f"      - {throttle['timestamp']}: {throttle['reason']}")
            lines.append("")

        lines.append("=" * 70)
        lines.append("EVENT DETAILS (context; metrics above are higher-signal)")
        lines.append("=" * 70)
        lines.append("")

        if events and any(e.ts for e in events):
            timestamps = [e.ts for e in events if e.ts]
            if timestamps:
                earliest = min(timestamps)
                latest = max(timestamps)
                span = (latest - earliest).total_seconds()
                lines.append(
                    f"Time range: {earliest.isoformat()} to {latest.isoformat()} "
                    f"({span:.1f} seconds)\n"
                )

        lines.append(f"Top {len(display_events)} events:\n")

        for i, event in enumerate(display_events, 1):
            timestamp_str = event.ts.isoformat() if event.ts else "N/A"
            role_str = event.role or "N/A"

            severity_indicator = ""
            if (event.severity or 0) >= 40:
                severity_indicator = " CRITICAL ERROR"
            elif (event.severity or 0) == 20:
                severity_indicator = " WARNING"

            fields_str = "N/A"
            if event.fields_json:
                if len(event.fields_json) <= 5:
                    fields_str = json.dumps(event.fields_json, indent=2)
                else:
                    top_fields = dict(list(event.fields_json.items())[:5])
                    fields_str = json.dumps(top_fields, indent=2) + "\n    ... (truncated)"

            event_block = f"""
Event {i}{severity_indicator}:
  Timestamp: {timestamp_str}
  Event Type: {event.event}
  Severity: {event.severity} ({'ERROR' if (event.severity or 0) >= 40 else 'WARNING' if (event.severity or 0) == 20 else 'INFO'})
  Role: {role_str}
  Fields:
{fields_str}
"""
            lines.append(event_block)

        if total_count > len(display_events):
            lines.append(f"\n... and {total_count - len(display_events)} more events")

        return "\n".join(lines)

        # ======================================================
        # LLM CALL
        # ======================================================

    def _analyze_with_llm(
        self,
        events_text: str,
        question: str,
        tools_doc: str,
        hypothesis: Optional[str] = None,
        confidence: float = 0.0,
        tools_used: Optional[List[str]] = None,
        api_key: Optional[str] = None
    ) -> Dict[str, Any]:
        """Analyze events with LLM."""
        api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not set. Provide api_key or set GEMINI_API_KEY env var.")

        try:
            genai.configure(api_key=api_key)
        except Exception:
            pass

        model = genai.GenerativeModel('gemini-2.5-flash')

        tools_used_str = f"\nTools already used: {', '.join(tools_used) if tools_used else 'None'}"

        prompt = f"""
You are an expert at analyzing FoundationDB simulation logs.

You are given:
- A description of available investigation tools:
{tools_doc}

- Cluster knowledge base:
{FDB_KNOWLEDGE_BASE}

- A formatted view of events and extracted metrics:
{events_text}

Your job is to identify the most likely issue/scenario being tested and explain it.
Always include the best-matching CLUSTER ID(s) in the hypothesis text (e.g., "CLUSTER 6: ..."). You may list multiple clusters if the evidence supports more than one.
If you cannot map to a cluster, say "cluster unknown" explicitly.

QUESTION: {question}

IMPORTANT:
The JSON format MUST remain exactly:

{{
  "hypothesis": "Single most likely scenario (must mention the relevant CLUSTER number(s) inline; may list multiple clusters if appropriate)",
  "confidence": 0.0,
  "reasoning": "Grounded explanation referencing only REAL events and metrics",
  "suggested_tools": [],
  "next_steps": ""
}}

- Do NOT add fields like `cluster_id` or `cluster_name`.
- Hypothesis MUST cite the cluster number(s) if identifiable; otherwise say 'cluster unknown'.
- Base your confidence on how well metrics + events align with a specific failure scenario.
{tools_used_str}
"""

        try:
            import time
            from google.api_core import exceptions as api_exceptions

            max_retries = 3
            retry_delay = 10

            result: Dict[str, Any] = {}
            for attempt in range(max_retries):
                try:
                    response = model.generate_content(prompt)
                    response_text = response.text.strip()

                    if "```json" in response_text:
                        start = response_text.find("```json") + 7
                        end = response_text.find("```", start)
                        response_text = response_text[start:end].strip()
                    elif "```" in response_text:
                        start = response_text.find("```") + 3
                        end = response_text.find("```", start)
                        response_text = response_text[start:end].strip()

                    result = json.loads(response_text)
                    break
                except Exception as api_error:
                    error_str = str(api_error).lower()
                    is_quota_error = (
                        "quota" in error_str or
                        "rate limit" in error_str or
                        "429" in error_str or
                        "resource has been exhausted" in error_str or
                        ("exceeded" in error_str and "quota" in error_str)
                    )

                    if hasattr(api_exceptions, 'ResourceExhausted'):
                        if isinstance(api_error, api_exceptions.ResourceExhausted):
                            is_quota_error = True

                    if is_quota_error and attempt < max_retries - 1:
                        wait_time = retry_delay * (2 ** attempt)
                        print(f"\nAPI Quota Exceeded. Waiting {wait_time}s before retry {attempt + 2}/{max_retries}...")
                        time.sleep(wait_time)
                        continue
                    elif is_quota_error:
                        print(f"\n❌ API Quota Exceeded after {max_retries} retries.")
                        return {
                            "hypothesis": "API Quota Exceeded: Unable to complete LLM analysis due to quota limits.",
                            "confidence": 0.0,
                            "reasoning": str(api_error)[:200],
                            "suggested_tools": [],
                            "next_steps": "Check quota/billing and retry later."
                        }
                    else:
                        raise

            hypothesis_out = result.get("hypothesis", "")
            confidence_out = float(result.get("confidence", 0.0))
            reasoning_out = result.get("reasoning", "")

            # Metric focus vs event-name focus adjustment
            has_metric_focus = any(keyword in (hypothesis_out + reasoning_out).lower()
                                   for keyword in [
                                       "versionlag", "version_lag", "lag", "latency",
                                       "throttle", "throttl", "slowss", "metric",
                                       "storage pressure", "performance", "degradation"
                                   ])

            event_name_focus = any(keyword in hypothesis_out.lower()
                                   for keyword in [
                                       "fkreenablelb", "file not found", "severity 30",
                                       "severity 20", "informational"
                                   ])

            has_metric_anomalies = any(keyword in events_text.lower()
                                       for keyword in [
                                           "versionlag spike", "negative latenc",
                                           "slowssloop", "throttling", ">100k", ">1m"
                                       ])

            if event_name_focus and not has_metric_focus and has_metric_anomalies:
                confidence_out = min(confidence_out, 0.4)
                reasoning_out = (
                    "[Confidence reduced: Metrics detected but hypothesis focuses on event names. "
                    "Metrics are more important than event severity.] " + reasoning_out
                )
            elif event_name_focus and not has_metric_focus:
                confidence_out = min(confidence_out, 0.5)
                reasoning_out = (
                    "[Confidence adjusted: Hypothesis focuses on event names rather than metrics] "
                    + reasoning_out
                )
            elif has_metric_focus and has_metric_anomalies:
                confidence_out = min(confidence_out + 0.1, 1.0)
                reasoning_out = (
                    "[✓ Confidence boosted: Hypothesis correctly focuses on metrics] " +
                    reasoning_out
                )

            return {
                "hypothesis": hypothesis_out,
                "confidence": confidence_out,
                "reasoning": reasoning_out,
                "suggested_tools": result.get("suggested_tools", []),
                "next_steps": result.get("next_steps", "")
            }

        except Exception as e:
            return {
                "hypothesis": f"Error during LLM analysis: {e}",
                "confidence": 0.0,
                "reasoning": str(e),
                "suggested_tools": [],
                "next_steps": ""
            }

    # ======================================================
    # MAIN INVESTIGATION LOOP (using unified tools)
    # ======================================================

    def investigate(
        self,
        initial_question: str,
        api_key: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> InvestigationResult:
        """
        Investigate issue by querying database, using unified tools, and analyzing with LLM.

        Phases:
          A) Global sweep via GlobalScanner
          B) Focused drill-down via HotspotSelector + ContextAnalyzer
          C) Coverage / detector check via Detectors + HotspotSelector
        """
        api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not set. Provide api_key or set GEMINI_API_KEY env var.")

        genai.configure(api_key=api_key)

        hypothesis: Optional[str] = None
        confidence: float = 0.0
        reasoning: str = ""
        tools_used: List[str] = []
        all_events: List[EventModel] = []
        all_additional_data: List[tuple] = []
        timeline_highlights: Dict[str, Any] = {}
        inspected_buckets: List[int] = []
        bucket_data: List[Dict[str, Any]] = []
        global_summary_done = False
        coverage_complete = False
        phase = "A"
        tools_doc = self._get_tools_documentation()
        iteration = 0
        context_dirty = True
        llm_calls = 0
        hotspot_inspected = False

        print(f"\nInvestigating: {initial_question}")
        print(f"   Target confidence: {self.confidence_threshold:.2f}")
        print(f"   Max iterations: {self.max_iterations}\n")
        if self.use_rag:
            corpus_msg = self.rag_corpus or os.getenv("RAG_CORPUS_RESOURCE") or os.getenv("RAG_CORPUS") or "default corpus"
            print(f"   RAG: enabled (corpus={corpus_msg})")
        else:
            print("   RAG: disabled (set use_rag=True or RAG_CORPUS_RESOURCE env to enable)")

        while iteration < self.max_iterations:
            iteration += 1
            print(f"\n{'='*70}")
            print(f"Iteration {iteration}/{self.max_iterations} - Phase {phase}")
            print(f"{'='*70}")

            # -------------------------
            # Phase A: Global Sweep
            # -------------------------
            if phase == "A":
                print("\nPhase A: Global Sweep (GlobalScanner)\n")

                # 1) Top severe events (>=30)
                print("\n1)  Top events (severity >= 30)...")
                top_events = self.scanner.top_events(severity_min=30, limit=500)
                tools_used.append("GlobalScanner.top_events")
                if self._append_events(all_events, top_events):
                    context_dirty = True
                print(f"   Found {len(top_events)} top events")

                # 2) Severity counts
                print("\n2)  Severity counts...")
                sev_counts = self.scanner.severity_counts()
                tools_used.append("GlobalScanner.severity_counts")
                self._record_additional_data(all_additional_data, "severity_counts", sev_counts)
                print(f"   Severity distribution: {sev_counts}")

                # 3) Event histogram
                print("\n3) Event histogram (top 10)...")
                histogram = self.scanner.event_histogram(limit=10)
                tools_used.append("GlobalScanner.event_histogram")
                self._record_additional_data(all_additional_data, "event_histogram", histogram)
                print(f"   Top event types: {list(histogram.items())[:5]}")

                # 4) Time span
                print("\n4) Time span...")
                span = self.scanner.time_span()
                tools_used.append("GlobalScanner.time_span")
                self._record_additional_data(all_additional_data, "time_span", span)
                print(f"   Earliest: {span.get('earliest')}, Latest: {span.get('latest')}")
                print(f"   Duration (s): {span.get('duration_seconds')}")

                # 5) Bucket heatmap
                print("\n5)  Bucket heatmap (max severity per bucket)...")
                buckets = self.scanner.bucket_heatmap(bucket_seconds=300, limit=100)
                tools_used.append("GlobalScanner.bucket_heatmap")
                self._record_additional_data(all_additional_data, "bucket_heatmap", buckets)
                print(f"   Found {len(buckets)} buckets")

                for b in buckets[:10]:
                    if "bucket_start_epoch" in b:
                        inspected_buckets.append(int(b["bucket_start_epoch"]))

                # 6) Global summary
                print("\n6)  Global summary...")
                summary = self.scanner.global_summary()
                tools_used.append("GlobalScanner.global_summary")
                self._record_additional_data(all_additional_data, "global_summary", summary)
                global_summary_done = True
                print(f"   Max severity: {summary.get('max_severity')}")
                print(f"   Top event types: {summary.get('event_histogram', {})}")

                # 7) Rollback Analysis
                print("\n7)  Rollback analysis (version drops/resets/recovery-regressions)...")
                rollback_info = self.scanner.rollback_analysis()
                tools_used.append("GlobalScanner.rollback_analysis")
                self._record_additional_data(all_additional_data, "rollback_analysis", rollback_info)
                print(f"   Rollback detected: {rollback_info.get('detected')}, "
                    f"drops={rollback_info.get('num_drops')}, "
                    f"resets={rollback_info.get('num_resets')}, "
                    f"rv_resets={rollback_info.get('num_recovery_resets')}")


                # 8) Metric Baselines
                print("\n8)  Metric baselines (persist + summarize)...")
                try:
                    baseline_info = self.scanner.upsert_metric_baselines(min_count=20, top_n=500, per_role=True)
                    tools_used.append("GlobalScanner.upsert_metric_baselines")
                    self._record_additional_data(all_additional_data, "metric_baselines_upsert", baseline_info)
                    print(f"   Baselines upserted: {baseline_info.get('upserted')}")
                except Exception as e:
                    print(f"   metric_baselines error: {e}")
                
                #9) Recovery Episodes
                print("\n9)  Recovery episodes (grouped recoveries)...")
                try:
                    recovery_eps = self.scanner.recovery_episodes()
                    tools_used.append("GlobalScanner.recovery_episodes")
                    self._record_additional_data(all_additional_data, "recovery_episodes", recovery_eps)
                    print(f"   Recovery episodes: {recovery_eps.get('count')}")
                except Exception as e:
                    print(f"   recovery_episodes error: {e}")
                    recovery_eps = None

                bucket_data = buckets
                timeline_highlights = {
                    "time_span": span,
                    "top_event_types": list(summary.get("event_histogram", {}).items())[:5] if summary else [],
                    "hot_buckets": [
                        {
                            "bucket_start": b.get("bucket_start").isoformat() if b.get("bucket_start") else b.get("bucket_start_epoch"),
                            "max_severity": b.get("max_severity"),
                            "count": b.get("count"),
                        }
                        for b in buckets[:5]
                    ] if buckets else [],
                    "rollback_detected": rollback_info.get("detected") if rollback_info else None,
                    "recovery_episodes": recovery_eps.get("episodes") if recovery_eps else None,
                }

                phase = "B"
                print("\nPhase A complete. Moving to Phase B.")
                print(f"   Additional data records so far: {len(all_additional_data)}")


                context_dirty = True
                # Defer LLM until Phase B iteration
                continue


            # -------------------------
            # Phase C: Coverage -> Detectors (run hotspots first)
            # -------------------------
            # -------------------------
            # Global detectors first
            # -------------------------
            print("\nRunning core detectors (global only before hotspot dive)...")
            det_results = {}
            try:
                det_results["storage_engine_pressure"] = self.detectors.storage_engine_pressure()
                tools_used.append("Detectors.storage_engine_pressure(global)")
                print("   ➜ Detectors.storage_engine_pressure(global)")
            except Exception as e:
                    print(f"   storage_engine_pressure(global) error: {e}")
            try:
                det_results["recovery_loop"] = self.detectors.recovery_loop()
                tools_used.append("Detectors.recovery_loop(global)")
                print("   ➜ Detectors.recovery_loop(global)")
            except Exception as e:
                    print(f"   recovery_loop(global) error: {e}")
            try:
                det_results["ratekeeper_throttling"] = self.detectors.ratekeeper_throttling()
                tools_used.append("Detectors.ratekeeper_throttling(global)")
                print("   ➜ Detectors.ratekeeper_throttling(global)")
            except Exception as e:
                    print(f"   ratekeeper_throttling(global) error: {e}")
            try:
                det_results["missing_tlogs"] = self.detectors.missing_tlogs()
                tools_used.append("Detectors.missing_tlogs(global)")
                print("   ➜ Detectors.missing_tlogs(global)")
            except Exception as e:
                    print(f"   missing_tlogs(global) error: {e}")
            try:
                det_results["coordination_loss"] = self.detectors.coordination_loss()
                tools_used.append("Detectors.coordination_loss(global)")
                print("   ➜ Detectors.coordination_loss(global)")
            except Exception as e:
                    print(f"   coordination_loss(global) error: {e}")
            try:
                det_results["zscore_hotspots"] = self.detectors.zscore_hotspots()
                tools_used.append("Detectors.zscore_hotspots(global)")
                print("   ➜ Detectors.zscore_hotspots(global)")
            except Exception as e:
                    print(f"   zscore_hotspots(global) error: {e}")
            try:
                det_results["baseline_window_anomalies"] = self.detectors.baseline_window_anomalies()
                tools_used.append("Detectors.baseline_window_anomalies(global)")
                print("   ➜ Detectors.baseline_window_anomalies(global)")
            except Exception as e:
                    print(f"   baseline_window_anomalies(global) error: {e}")
            try:
                det_results["metric_anomalies"] = self.detectors.metric_anomalies()
                tools_used.append("Detectors.metric_anomalies(global)")
                print("   ➜ Detectors.metric_anomalies(global)")
            except Exception as e:
                    print(f"   metric_anomalies(global) error: {e}")

            self._record_additional_data(all_additional_data, "detectors", det_results)
            self._last_det_results = det_results

            # Build timeline after detectors + buckets
            try:
                self.timeline_summary = self.timeline_builder.build(
                    all_events,
                    det_results,
                    bucket_data,
                    timeline_highlights.get("recovery_episodes") if timeline_highlights else None,
                )
                if self.timeline_summary:
                    self._record_additional_data(all_additional_data, "timeline_builder", self.timeline_summary)
                    context_dirty = True
            except Exception as e:
                print(f"   timeline_builder error: {e}")

            rag_added = self._run_rag_retrieval(det_results, timeline_highlights, all_additional_data, api_key, tools_used)
            if rag_added:
                context_dirty = True

            # -------------------------
            # LLM Analysis (after detectors/RAG)
            # -------------------------
            analysis = {
                "hypothesis": hypothesis or "",
                "confidence": confidence,
                "reasoning": reasoning,
                "suggested_tools": [],
                "next_steps": ""
            }

            if llm_calls >= self.max_llm_calls:
                print(f"\nSkipping LLM: call budget {self.max_llm_calls} exhausted.")
            elif not context_dirty:
                print("\nSkipping LLM: no new context since last analysis.")
            else:
                events_text = self._format_events_for_llm(all_events[:100])

                if all_additional_data:
                    additional_text = "\n\nAdditional Investigation Data:\n" + "\n".join(
                        [
                            f"\n{tool_name}:\n{json.dumps(data, indent=2, default=str)}"
                            for tool_name, data in all_additional_data
                        ]
                    )
                    events_text += additional_text

                if self.timeline_summary:
                    events_text += "\n\nTimeline Builder:\n" + json.dumps(self.timeline_summary, indent=2, default=str)

                events_text = self._enforce_context_limit(events_text)

                try:
                    path = write_llm_input(events_text)
                    if path:
                        print(f"   LLM input written to: {path}")
                except Exception as e:
                    print(f"   llm_input_logger error: {e}")

                print(f"\nAnalyzing with LLM (call {llm_calls + 1}/{self.max_llm_calls})...")
                print(f"   Current confidence: {confidence:.2f}")
                if hypothesis:
                    print(f"   Current hypothesis: {hypothesis[:100]}...")

                analysis = self._analyze_with_llm(
                    events_text,
                    initial_question,
                    tools_doc,
                    hypothesis,
                    confidence,
                    tools_used,
                    api_key
                )
                # Persist LLM output to file
                try:
                    out_path = write_llm_output(json.dumps(analysis, indent=2, default=str))
                    if out_path:
                        print(f"   LLM output written to: {out_path}")
                except Exception as e:
                    print(f"   llm_output_logger error: {e}")
                llm_calls += 1
                context_dirty = False

            hypothesis = analysis["hypothesis"]
            confidence = analysis["confidence"]
            reasoning = analysis.get("reasoning", "")

            print(f"\nIteration {iteration} Results:")
            print(f"   Confidence: {confidence:.2f} (target: {self.confidence_threshold:.2f})")
            print(f"   Hypothesis: {hypothesis[:150]}...")

            # -------------------------
            # Hotspot dive (z-score guided) after LLM if confidence low
            # -------------------------
            chosen_bucket = None
            try:
                # Prefer z-score hotspots from detectors
                zhot = det_results.get("zscore_hotspots", {})
                if zhot.get("detected") and zhot.get("hotspots"):
                    chosen_bucket = zhot["hotspots"][0]
                    print("   Using z-score hotspot for next dive.")
            except Exception:
                pass

            if not chosen_bucket:
                print("\nChecking for uncovered high-severity buckets (HotspotSelector)...")
                uncovered = []
                try:
                    uncovered = self.hotspots.get_uncovered(
                        inspected_buckets,
                        min_severity=10,
                        bucket_seconds=10,
                    )
                    tools_used.append("HotspotSelector.get_uncovered")
                except Exception as e:
                    print(f"   get_uncovered error: {e}")
                if uncovered:
                    chosen_bucket = uncovered[0]

            if chosen_bucket:
                bucket_epoch = int(chosen_bucket["bucket_start_epoch"])
                bucket_start = datetime.utcfromtimestamp(bucket_epoch)
                bucket_seconds = 10
                window_start = bucket_start
                window_end = bucket_start + timedelta(seconds=bucket_seconds)
                around_time = bucket_start + timedelta(seconds=bucket_seconds / 2)

                print(f"\n   Inspecting bucket starting at {bucket_start.isoformat()}")
                try:
                    dbg_count = self.ctx.con.execute(
                        "SELECT COUNT(*) FROM events WHERE ts BETWEEN ? AND ?",
                        [window_start, window_end],
                    ).fetchone()[0]
                    print(f"   Window row count (ts between {window_start} and {window_end}): {dbg_count}")
                except Exception as e:
                    print(f"   window count debug failed: {e}")

                try:
                    bucket_events = self.ctx.context_window(
                        around_time=around_time,
                        window_seconds=bucket_seconds / 2,
                        limit=200
                    )
                    tools_used.append("ContextAnalyzer.context_window")
                    if self._append_events(all_events, bucket_events):
                        context_dirty = True
                    inspected_buckets.append(bucket_epoch)
                    print(f"   Added {len(bucket_events)} events from hotspot bucket")
                    hotspot_inspected = True
                except Exception as e:
                    print(f"   context_window error: {e}")
            else:
                print("\n   No remaining hotspots/buckets to inspect.")
                coverage_complete = True

            # If confidence already high, stop early (but ensure at least one hotspot inspected if available)
            if confidence >= self.confidence_threshold and (hotspot_inspected or coverage_complete):
                print("\nConfidence threshold reached; stopping iterations.")
                break

            # If we've hit max iterations, stop
            if iteration >= self.max_iterations:
                print(f"\nReached max iterations ({self.max_iterations})")
                break

        # ============================
        # Final Reporting
        # ============================
        self._print_structured_report(hypothesis, confidence, reasoning, self.timeline_summary, self._last_det_results, all_additional_data)
        return InvestigationResult(
            hypothesis=hypothesis or "",
            confidence=confidence,
            reasoning=reasoning,
            tools_used=tools_used,
            evidence_events=[]
        )

    # ======================================================
    # STRUCTURED RCA OUTPUT
    # ======================================================
    def _print_structured_report(
        self,
        hypothesis: str,
        confidence: float,
        reasoning: str,
        timeline: Optional[Dict[str, Any]],
        det_results: Dict[str, Any],
        additional_data: List[tuple],
    ):
        """Print final RCA in compact hypothesis/confidence/cluster form."""
        rag_block = next((d for d in additional_data if d[0] == "rag_retrieval"), None)
        rag_text = ""
        if rag_block and isinstance(rag_block[1], dict):
            rag_text = rag_block[1].get("response_text") or rag_block[1].get("response", "")

        def _collect_citations(det: Dict[str, Any]) -> list:
            cites = []
            # storage_engine_pressure anomalies
            sp = det.get("storage_engine_pressure", {}) if det else {}
            for a in sp.get("anomalies_sample", [])[:2]:
                cites.append({
                    "ts": a.get("ts"),
                    "event": "StorageMetrics",
                    "note": f"VersionLag={a.get('value')}, z={a.get('zscore')}"
                })
            # ratekeeper events
            rk = det.get("ratekeeper_throttling", {}) if det else {}
            for e in rk.get("events", [])[:2]:
                cites.append({
                    "ts": e.get("timestamp"),
                    "event": e.get("event"),
                    "note": f"fields={e.get('fields')}"
                })
            # missing tlogs
            mt = det.get("missing_tlogs", {}) if det else {}
            for e in mt.get("events", [])[:2]:
                cites.append({
                    "ts": e.get("timestamp"),
                    "event": e.get("event"),
                    "note": f"sev={e.get('severity')}"
                })
            # baseline anomalies
            ba = det.get("baseline_window_anomalies", {}) if det else {}
            for a in ba.get("anomalies", [])[:1]:
                cites.append({
                    "ts": a.get("bucket_start"),
                    "event": "MetricBaselineZ",
                    "note": f"{a.get('metric')} z={a.get('zscore')} role={a.get('role')}"
                })
            # metric anomalies sample
            ma = det.get("metric_anomalies", {}) if det else {}
            for a in ma.get("sample", [])[:1]:
                cites.append({
                    "ts": a.get("ts"),
                    "event": a.get("metric"),
                    "note": f"z={a.get('z_score')} val={a.get('value')}"
                })
            # keep top few
            return cites[:5]

        citations = _collect_citations(det_results or {})

        print("\n" + "=" * 70)
        print("RCA Summary")
        print("=" * 70)
        print(f"Hypothesis: {hypothesis}")
        print(f"Confidence: {confidence:.2f}")
        if timeline and timeline.get("timeline"):
            first_anom = timeline.get("first_anomaly")
            if first_anom:
                print(f"First anomaly: {first_anom}")
            print("Timeline (top 5):")
            for item in timeline["timeline"][:5]:
                print(f"  {item.get('t')}: {item.get('event')} — {item.get('note')}")

        if citations:
            print("\nCitations (log/metric snippets):")
            for c in citations:
                print(f"  [{c.get('ts')}] {c.get('event')}: {c.get('note')}")

        print("\nSuggested fixes:")
        print("  - Check storage server I/O and backlog (VersionLag, disk queue) if storage pressure is detected.")
        print("  - Investigate TLog health (SharedTLogFailed/TLogError) and disk I/O if TLog failures appear.")
        print("  - Examine recovery loops frequency and quiet database failures; stabilize before config changes.")

        print("\nSuggested commands (run in DuckDB or shell):")
        print("  duckdb <db_path> \"SELECT ts, event, severity, fields_json FROM events WHERE severity>=20 ORDER BY ts LIMIT 20;\"")
        print("  duckdb <db_path> \"SELECT ts, role, fields_json->>'VersionLag' AS versionlag FROM events WHERE event='StorageMetrics' ORDER BY ts DESC LIMIT 20;\"")
        print("  duckdb <db_path> \"SELECT ts, event, severity, fields_json FROM events WHERE event LIKE 'QuietDatabase%' ORDER BY ts LIMIT 20;\"")
        print("  duckdb <db_path> \"SELECT ts, event, severity, fields_json FROM events WHERE event LIKE '%TLog%' AND severity>=10 ORDER BY ts LIMIT 20;\"")

        # Add a blank line before tools to improve readability if upstream prints them
        print()
