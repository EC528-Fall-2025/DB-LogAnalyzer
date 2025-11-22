"""
Investigation Agent - LLM-based log analysis with tool access.

Purpose: Analyze DuckDB events using LLM and investigation tools to identify root causes.
"""

import json
import os
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Tuple

# Suppress FutureWarning from google.api_core (dependency of google-genai)
warnings.filterwarnings('ignore', category=FutureWarning, module='google.api_core')

# Workaround for importlib.metadata.packages_distributions compatibility
# This attribute exists in Python 3.10+, but some dependencies may access it incorrectly
try:
    import importlib.metadata as _metadata
    # Ensure packages_distributions exists (Python 3.10+)
    if not hasattr(_metadata, 'packages_distributions'):
        # Fallback: provide a minimal implementation
        def _packages_distributions():
            """Fallback for packages_distributions if not available."""
            return {}
        _metadata.packages_distributions = _packages_distributions
except (ImportError, AttributeError):
    # If importlib.metadata doesn't exist at all, we can't fix it
    pass

import google.generativeai as genai

from data_transfer_object.event_dto import EventModel
from tools.investigation_tools import get_events


def load_events_window(
    db_path: str,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    limit: Optional[int] = None
) -> List[EventModel]:
    """
    Load events from database within a time window.
    
    Args:
        db_path: Path to DuckDB database
        start_time: Optional start time filter
        end_time: Optional end time filter
        limit: Maximum number of events to return
        
    Returns:
        List of EventModel instances
    """
    return get_events(
        db_path=db_path,
        start_time=start_time,
        end_time=end_time,
        limit=limit or 1000
    )


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
    """Agent that analyzes log events using LLM and investigation tools."""
    
    def __init__(
        self,
        db_path: str,
        max_iterations: int = 10,
        confidence_threshold: float = 0.8,
        max_llm_calls: int = 4
    ):
        """
        Initialize investigation agent.
        
        Args:
            db_path: Path to DuckDB database
            max_iterations: Maximum number of investigation iterations
            confidence_threshold: Confidence level required to stop investigation
            max_llm_calls: Max Gemini invocations allowed per investigation
        """
        self.db_path = db_path
        self.max_iterations = max_iterations
        self.confidence_threshold = confidence_threshold
        self.max_llm_calls = max(1, max_llm_calls)
    
    def _get_tools_documentation(self) -> str:
        """Generate documentation for all available investigation tools."""
        return """
AVAILABLE INVESTIGATION TOOLS (trimmed for faster loops):

GLOBAL SCANS:
- global_severity_scan(db_path, min_severity=30, limit=500)
  → Mandatory severity sweep to locate hotspots
- global_severity_scan_warnings(db_path, limit=500)
  → Backup sweep when severity 30+ is empty
- global_event_histogram(db_path, limit=50)
  → Quick breakdown of dominant event types
- global_severity_counts(db_path)
  → Severity distribution for sanity checks
- bucket_max_severity(db_path, bucket_seconds=300)
  → Identifies time buckets that deserve attention
- get_time_span(db_path)
  → Earliest and latest timestamps for context

FOCUSED QUERIES:
- get_events(db_path, start_time=None, end_time=None, limit=1000)
  → Retrieve detailed evidence windows
- changes_after(db_path, after_time, limit=500)
  → Extend investigation beyond a timestamp
- get_context_window(db_path, around_time, window_seconds=30, limit=200)
  → Inspect a tight neighborhood around a key event
- get_recovery_timeline(db_path, start_time=None, end_time=None)
  → Understand recovery phases and loops

METRICS / DETECTORS:
- get_lag_series(db_path, start_time=None, end_time=None, limit=1000)
  → Storage lag trend (VersionLag)
- detect_storage_engine_pressure(db_path, lag_threshold=50000.0)
  → Highlights VersionLag spikes
- detect_ratekeeper_throttling(db_path)
  → Finds Ratekeeper slowdowns
- detect_recovery_loop(db_path, time_window_seconds=60)
  → Flags repeated recoveries

COVERAGE / COMPLETENESS:
- get_uncovered_buckets(db_path, inspected_buckets, bucket_seconds=600, min_severity=20)
  → Ensures severe windows are inspected
- get_global_summary(db_path)
  → Final roll-up before exiting

All other heavy tools are intentionally skipped to save LLM bandwidth.
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
        destination: List[Tuple[str, Any]],
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
    
    def _extract_metrics_from_events(self, events: List[EventModel]) -> Dict[str, Any]:
        """Extract critical metrics from events to prioritize over event severity."""
        metrics = {
            "version_lag_spikes": [],  # VersionLag > 100k
            "negative_latencies": [],  # Negative latency values
            "slow_ss_loops": [],  # SlowSSLoop events
            "throttling_reasons": [],  # Throttling in RkUpdate
            "high_lag_timestamps": []
        }
        
        for event in events:
            if not event.fields_json:
                continue
            
            fields = event.fields_json
            
            # Extract VersionLag from StorageMetrics, RkUpdate, etc.
            version_lag = None
            for key in ["VersionLag", "versionLag", "VersionLagValue", "Lag", "lag"]:
                if key in fields:
                    try:
                        version_lag = float(fields[key])
                        break
                    except (ValueError, TypeError):
                        continue
            
            if version_lag is not None:
                if version_lag > 100000:  # >100k is significant
                    metrics["version_lag_spikes"].append({
                        "timestamp": event.ts.isoformat() if event.ts else "N/A",
                        "event_type": event.event,
                        "version_lag": version_lag,
                        "role": event.role,
                        "severity": event.severity
                    })
                    if version_lag > 1000000:  # >1M is critical
                        metrics["high_lag_timestamps"].append(event.ts)
            
            # Detect negative latencies (timing bugs/overflows)
            for key in fields:
                key_lower = str(key).lower()
                if "latency" in key_lower or "min" in key_lower or "max" in key_lower:
                    try:
                        val = float(fields[key])
                        if val < 0:  # Negative value indicates bug/overflow
                            metrics["negative_latencies"].append({
                                "timestamp": event.ts.isoformat() if event.ts else "N/A",
                                "event_type": event.event,
                                "metric": key,
                                "value": val,
                                "role": event.role
                            })
                    except (ValueError, TypeError):
                        continue
            
            # Detect SlowSSLoop
            if "SlowSSLoop" in str(event.event) or "SlowSS" in str(event.event):
                metrics["slow_ss_loops"].append({
                    "timestamp": event.ts.isoformat() if event.ts else "N/A",
                    "event_type": event.event,
                    "severity": event.severity,
                    "fields": fields
                })
            
            # Detect throttling in RkUpdate events
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
        
        # Extract metrics FIRST (these are more important than event severity)
        metrics = self._extract_metrics_from_events(events)
        
        # Sort by severity 40+ first, then chronologically
        sorted_events = sorted(events, key=lambda e: (
            -(e.severity or 0) if (e.severity or 0) >= 40 else -1000,
            e.ts if e.ts else datetime.max
        ))
        
        severity_40_plus = [e for e in sorted_events if (e.severity or 0) >= 40]
        severity_20_warnings = [e for e in sorted_events if (e.severity or 0) == 20]
        severity_30_info = [e for e in sorted_events if (e.severity or 0) == 30]
        
        display_events = sorted_events[:20]
        total_count = len(events)
        
        lines = []
        lines.append(f"Found {total_count} events:")
        lines.append(f"  - Severity 40+ (Errors): {len(severity_40_plus)}")
        lines.append(f"  - Severity 30 (Info): {len(severity_30_info)}")
        lines.append(f"  - Severity 20 (Warnings): {len(severity_20_warnings)}")
        lines.append("")
        
        # PRIORITIZE METRICS OVER SEVERITY
        lines.append("=" * 70)
        lines.append("🔴 CRITICAL: PRIORITIZE METRICS OVER EVENT SEVERITY")
        lines.append("=" * 70)
        lines.append("")
        lines.append("⚠️  METRICS ARE MORE IMPORTANT THAN EVENT SEVERITY!")
        lines.append("   - VersionLag spikes (>100k, especially >1M) indicate storage pressure")
        lines.append("   - Negative latencies indicate timing bugs/overflows")
        lines.append("   - Throttling reasons show performance degradation")
        lines.append("   - SlowSSLoop indicates storage server performance issues")
        lines.append("")
        lines.append("❌ DO NOT identify Severity 30 events (like FKReenableLB) as root cause")
        lines.append("   These are SYMPTOMS, not causes. Find the METRIC anomalies that caused them.")
        lines.append("")
        
        # Show metric anomalies FIRST
        if metrics["version_lag_spikes"]:
            lines.append("🚨 VERSIONLAG SPIKES (CRITICAL - Storage Engine Pressure):")
            lines.append(f"   Found {len(metrics['version_lag_spikes'])} events with VersionLag > 100k")
            critical_lags = [m for m in metrics["version_lag_spikes"] if m["version_lag"] > 1000000]
            if critical_lags:
                lines.append(f"   ⚠️  {len(critical_lags)} events with VersionLag > 1M (CRITICAL)")
                for lag in critical_lags[:5]:  # Show top 5
                    lines.append(f"      - {lag['timestamp']}: VersionLag={lag['version_lag']:.0f} (event: {lag['event_type']})")
            else:
                for lag in metrics["version_lag_spikes"][:5]:  # Show top 5
                    lines.append(f"      - {lag['timestamp']}: VersionLag={lag['version_lag']:.0f} (event: {lag['event_type']})")
            lines.append("")
        
        if metrics["negative_latencies"]:
            lines.append("🚨 NEGATIVE LATENCIES (Timing Bug/Overflow):")
            lines.append(f"   Found {len(metrics['negative_latencies'])} negative latency values")
            for neg in metrics["negative_latencies"][:5]:  # Show top 5
                lines.append(f"      - {neg['timestamp']}: {neg['metric']}={neg['value']} (event: {neg['event_type']})")
            lines.append("")
        
        if metrics["slow_ss_loops"]:
            lines.append("🚨 SLOW SS LOOPS (Storage Server Performance):")
            lines.append(f"   Found {len(metrics['slow_ss_loops'])} SlowSSLoop events")
            for slow in metrics["slow_ss_loops"][:3]:  # Show top 3
                lines.append(f"      - {slow['timestamp']}: {slow['event_type']}")
            lines.append("")
        
        if metrics["throttling_reasons"]:
            lines.append("🚨 THROTTLING DETECTED (Performance Degradation):")
            lines.append(f"   Found {len(metrics['throttling_reasons'])} throttling events")
            for throttle in metrics["throttling_reasons"][:5]:  # Show top 5
                lines.append(f"      - {throttle['timestamp']}: {throttle['reason']}")
            lines.append("")
        
        # Now show event details
        lines.append("=" * 70)
        lines.append("EVENT DETAILS (for context, but METRICS above are more important)")
        lines.append("=" * 70)
        lines.append("")
        lines.append("⚠️  REMEMBER: Severity 20 warnings (like FileNotFoundError) are NON-FATAL startup warnings, NOT root causes!")
        lines.append("⚠️  REMEMBER: Severity 30 events (like FKReenableLB) are SYMPTOMS, not root causes!")
        lines.append("   Focus on METRICS (VersionLag, latencies, throttling) to find the actual root cause.\n")
        
        if events and any(e.ts for e in events):
            timestamps = [e.ts for e in events if e.ts]
            if timestamps:
                earliest = min(timestamps)
                latest = max(timestamps)
                time_span = (latest - earliest).total_seconds()
                lines.append(f"Time range: {earliest.isoformat()} to {latest.isoformat()} ({time_span:.1f} seconds)\n")
        
        lines.append(f"Top {len(display_events)} events:\n")
        
        for i, event in enumerate(display_events, 1):
            timestamp_str = event.ts.isoformat() if event.ts else "N/A"
            role_str = event.role or "N/A"
            
            severity_indicator = ""
            if (event.severity or 0) >= 40:
                severity_indicator = " ⚠️ CRITICAL ERROR"
            elif (event.severity or 0) == 20:
                severity_indicator = " ⚠️ WARNING (non-fatal)"
            
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
    
    def _analyze_with_llm(
        self, 
        events_text: str, 
        question: str,
        tools_doc: str,
        hypothesis: Optional[str] = None,
        confidence: float = 0.0,
        tools_used: List[str] = None,
        api_key: Optional[str] = None
    ) -> Dict[str, Any]:
        """Analyze events with LLM, including tool suggestions."""
        api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not set. Provide api_key or set GEMINI_API_KEY env var.")
        
        try:
            genai.configure(api_key=api_key)
        except Exception:
            pass
        
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        tools_used_str = f"\nTools already used: {', '.join(tools_used) if tools_used else 'None'}" if tools_used else ""
        
        prompt = f"""You are an expert at analyzing FoundationDB log events to identify the SPECIFIC ISSUE being tested in simulations.

QUESTION: {question}

{f'CURRENT HYPOTHESIS: {hypothesis} (Confidence: {confidence:.2f})' if hypothesis else ''}
{tools_used_str}

{tools_doc}

CURRENT EVENTS FOUND:
{events_text}

CRITICAL RULES - PRIORITIZE METRICS OVER EVENT SEVERITY:

🔴 RULE 1: METRICS ARE MORE IMPORTANT THAN EVENT SEVERITY
   - VersionLag spikes (>100k = warning, >1M = CRITICAL) indicate storage engine pressure
   - Negative latencies indicate timing bugs/overflows
   - Throttling reasons show performance degradation
   - SlowSSLoop indicates storage server performance issues
   - These METRIC ANOMALIES are the ROOT CAUSES, not event names

   ⚠️  METRIC THRESHOLDS (Critical):
   - VersionLag > 100,000 (100k) = storage pressure warning
   - VersionLag > 1,000,000 (1M) = CRITICAL storage engine pressure
   - Negative latencies = timing bug/overflow
   - Throttling = performance degradation

🔴 RULE 2: Event names are SYMPTOMS, not root causes
   - FKReenableLB (Severity 30) = SYMPTOM of overload from fetch keys blocking
   - It's NOT the root cause - find what CAUSED it (VersionLag spikes, throttling)
   - Severity 30 events are INFORMATIONAL - they show something happened, not why

🔴 RULE 3: DO NOT prioritize by event severity alone
   - Severity 40+ errors are important, but METRICS tell the real story
   - Severity 30 (like FKReenableLB) are SYMPTOMS - find the metric anomaly that caused them
   - Severity 20 warnings (like FileNotFoundError) are NON-FATAL startup warnings

🔴 RULE 4: Adjust confidence based on metrics
   - HIGH confidence ONLY if metrics support the hypothesis
   - DROP confidence if metrics contradict warnings
   - If VersionLag >1M found, that's the root cause regardless of event severity

WHAT NOT TO DO:
- ❌ Do NOT identify FKReenableLB (Severity 30) as root cause - it's a SYMPTOM
- ❌ Do NOT prioritize Severity 30 events over metrics
- ❌ Do NOT ignore VersionLag spikes or negative latencies
- ❌ Do NOT hallucinate causes not in the data (e.g., "configuration_never_created")

Respond in JSON format:
{{
    "hypothesis": "your hypothesis about what issue/scenario is being tested (must reference METRICS, not just event names)",
    "confidence": 0.65,
    "reasoning": "brief explanation focusing on METRIC ANOMALIES (VersionLag, latencies, throttling)",
    "suggested_tools": ["detect_storage_engine_pressure", "get_recovery_timeline", ...],
    "next_steps": "what investigation tools to call next or additional analysis needed"
}}

LOW CONFIDENCE (<0.6) WHEN:
- only Severity 20/30 events found without metric analysis
- metrics contradict the hypothesis
- no VersionLag/latency/throttling analysis performed
- hypothesis focuses on event names rather than metrics

HIGH CONFIDENCE (>0.8) ONLY IF:
- identified specific METRIC ANOMALY (VersionLag >100k, negative latencies, throttling)
- can explain HOW the metric anomaly relates to the issue
- temporal progression makes sense (metrics degrade over time)
- hypothesis focuses on METRICS, not just event names
"""
        
        try:
            import time
            from google.api_core import exceptions as api_exceptions
            
            max_retries = 3
            retry_delay = 10  # Start with 10 seconds
            
            for attempt in range(max_retries):
                try:
                    response = model.generate_content(prompt)
                    response_text = response.text.strip()
                    
                    # Parse JSON from response
                    if "```json" in response_text:
                        json_start = response_text.find("```json") + 7
                        json_end = response_text.find("```", json_start)
                        response_text = response_text[json_start:json_end].strip()
                    elif "```" in response_text:
                        json_start = response_text.find("```") + 3
                        json_end = response_text.find("```", json_start)
                        response_text = response_text[json_start:json_end].strip()
                    
                    result = json.loads(response_text)
                    break  # Success, exit retry loop
                    
                except Exception as api_error:
                    error_str = str(api_error).lower()
                    
                    # Check for quota/rate limit errors
                    is_quota_error = (
                        "quota" in error_str or 
                        "rate limit" in error_str or 
                        "429" in error_str or 
                        "resource has been exhausted" in error_str or
                        "exceeded" in error_str and "quota" in error_str
                    )
                    
                    # Check for specific API exceptions
                    if hasattr(api_exceptions, 'ResourceExhausted'):
                        if isinstance(api_error, api_exceptions.ResourceExhausted):
                            is_quota_error = True
                    
                    if is_quota_error:
                        if attempt < max_retries - 1:
                            wait_time = retry_delay * (2 ** attempt)  # Exponential backoff: 10s, 20s, 40s
                            print(f"\n⚠️  API Quota Exceeded. Waiting {wait_time} seconds before retry {attempt + 2}/{max_retries}...")
                            print(f"   Error: {str(api_error)[:100]}")
                            print(f"   Tip: Check your quota at https://aistudio.google.com/app/apikey")
                            time.sleep(wait_time)
                            continue
                        else:
                            # Final retry failed, return error result
                            print(f"\n❌ API Quota Exceeded after {max_retries} retries.")
                            print(f"   Please check your billing/plan and quota at: https://aistudio.google.com/app/apikey")
                            print(f"   Wait some time before trying again, or upgrade your API plan.")
                            return {
                                "hypothesis": "API Quota Exceeded: Unable to complete LLM analysis due to quota limits.",
                                "confidence": 0.0,
                                "reasoning": f"The Gemini API quota has been exceeded after {max_retries} retries. Error: {str(api_error)[:200]}. Please check your billing/plan at https://aistudio.google.com/app/apikey. Wait some time before trying again, or upgrade your API plan.",
                                "suggested_tools": [],
                                "next_steps": "Check API quota/billing, wait before retrying, or upgrade your plan."
                            }
                    else:
                        # Non-quota error, don't retry
                        raise
            
            hypothesis = result.get("hypothesis", "")
            confidence = float(result.get("confidence", 0.0))
            reasoning = result.get("reasoning", "")
            
            # Adjust confidence based on metrics analysis
            # Check if hypothesis mentions metrics vs event names
            has_metric_focus = any(keyword in hypothesis.lower() or keyword in reasoning.lower() 
                                 for keyword in ["versionlag", "version_lag", "lag", "latency", 
                                                "throttl", "slowss", "metric", "anomaly", 
                                                "storage pressure", "performance", "degradation"])
            
            # Check if hypothesis focuses on event names (symptoms) rather than metrics (root causes)
            event_name_focus = any(keyword in hypothesis.lower() 
                                  for keyword in ["fkreenablelb", "file not found", "severity 30", 
                                                 "severity 20", "recovery", "informational"])
            
            # Check if events_text contains metric anomalies
            has_metric_anomalies = any(keyword in events_text.lower() 
                                     for keyword in ["versionlag spike", "negative latenc", 
                                                    "slowssloop", "throttling", ">100k", ">1m"])
            
            # Adjust confidence based on metrics vs event names
            if event_name_focus and not has_metric_focus and has_metric_anomalies:
                # Critical: Metrics exist but hypothesis ignores them - heavily penalize
                confidence = min(confidence, 0.4)  # Cap at 0.4
                reasoning = f"[⚠️ Confidence reduced: Metrics detected but hypothesis focuses on event names. Metrics are more important than event severity.] {reasoning}"
            elif event_name_focus and not has_metric_focus:
                # Hypothesis focuses on event names without metrics - reduce confidence
                confidence = min(confidence, 0.5)  # Cap at 0.5
                reasoning = f"[Confidence adjusted: Hypothesis focuses on event names rather than metrics] {reasoning}"
            elif has_metric_focus and has_metric_anomalies:
                # Good: Hypothesis mentions metrics and they exist - boost confidence slightly
                confidence = min(confidence + 0.1, 1.0)
                reasoning = f"[✓ Confidence boosted: Hypothesis correctly focuses on metrics] {reasoning}"
            
            return {
                "hypothesis": hypothesis,
                "confidence": confidence,
                "reasoning": reasoning,
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
    
    def investigate(
        self,
        initial_question: str,
        api_key: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> InvestigationResult:
        """
        Investigate issue by querying database, using tools, and analyzing with LLM.
        Implements two-phase strategy: Global sweep → Focused drill-down → Coverage check
        """
        api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not set. Provide api_key or set GEMINI_API_KEY env var.")
        
        genai.configure(api_key=api_key)
        
        # Import all investigation tools
        from tools.investigation_tools import (
            get_events,
            changes_after,
            get_recovery_timeline,
            get_context_window,
            get_lag_series,
            detect_storage_engine_pressure,
            detect_ratekeeper_throttling,
            detect_recovery_loop,
            global_severity_scan,
            global_severity_scan_warnings,
            global_event_histogram,
            global_severity_counts,
            bucket_max_severity,
            get_time_span,
            get_global_summary,
            get_uncovered_buckets,
        )
        
        # Map tool names to functions
        tool_map = {
            "get_events": get_events,
            "changes_after": changes_after,
            "get_recovery_timeline": get_recovery_timeline,
            "get_context_window": get_context_window,
            "get_lag_series": get_lag_series,
            "detect_storage_engine_pressure": detect_storage_engine_pressure,
            "detect_ratekeeper_throttling": detect_ratekeeper_throttling,
            "detect_recovery_loop": detect_recovery_loop,
            "global_severity_scan": global_severity_scan,
            "global_severity_scan_warnings": global_severity_scan_warnings,
            "global_event_histogram": global_event_histogram,
            "global_severity_counts": global_severity_counts,
            "bucket_max_severity": bucket_max_severity,
            "get_time_span": get_time_span,
            "get_global_summary": get_global_summary,
            "get_uncovered_buckets": get_uncovered_buckets,
        }
        
        hypothesis = None
        confidence = 0.0
        reasoning = ""
        tools_used = []
        all_events = []
        all_additional_data = []
        inspected_buckets = []  # Track which time buckets have been inspected
        global_summary_done = False
        phase = "A"  # Phase A = Global, B = Focused, C = Coverage Check
        tools_doc = self._get_tools_documentation()
        iteration = 0
        context_dirty = True
        llm_calls = 0
        
        print(f"\n🔍 Investigating: {initial_question}")
        print(f"   Target confidence: {self.confidence_threshold:.2f}")
        print(f"   Max iterations: {self.max_iterations}\n")
        
        # Iterative investigation loop with two-phase strategy
        while iteration < self.max_iterations:
            iteration += 1
            print(f"\n{'='*70}")
            print(f"📊 Iteration {iteration}/{self.max_iterations} - Phase {phase}")
            print(f"{'='*70}")
            
            # Phase A: Mandatory global sweep first (iteration 1 only)
            if iteration == 1 and phase == "A":
                print("\n🌐 Phase A: Mandatory Global Sweep")
                print("   Ensuring broad context before focused drill-down...\n")
                
                # 1. Global severity scan (severity >= 30)
                print("1️⃣  Global Severity Scan (>=30)...")
                events = global_severity_scan(self.db_path, min_severity=30, limit=500)
                tools_used.append("global_severity_scan")
                if self._append_events(all_events, events):
                    context_dirty = True
                print(f"   Found {len(events)} Severity 30+ events")
                
                # 2. If no severity 30+, check severity 20 warnings
                if not events:
                    print("\n2️⃣  Checking Severity 20 Warnings (well-known types)...")
                    warning_events = global_severity_scan_warnings(self.db_path, limit=500)
                    tools_used.append("global_severity_scan_warnings")
                    if self._append_events(all_events, warning_events):
                        context_dirty = True
                    print(f"   Found {len(warning_events)} Severity 20 warnings")
                
                # 3. Global histograms
                print("\n3️⃣  Global Event Histogram...")
                histogram = global_event_histogram(self.db_path, limit=50)
                tools_used.append("global_event_histogram")
                if self._record_additional_data(all_additional_data, "global_event_histogram", histogram):
                    context_dirty = True
                print(f"   Top event types: {list(histogram.items())[:5]}")
                
                # 4. Global severity counts
                print("\n4️⃣  Global Severity Counts...")
                severity_counts = global_severity_counts(self.db_path)
                tools_used.append("global_severity_counts")
                if self._record_additional_data(all_additional_data, "global_severity_counts", severity_counts):
                    context_dirty = True
                print(f"   Severity distribution: {severity_counts}")
                
                # 5. Bucket max severity (identify hotspots)
                print("\n5️⃣  Bucket Max Severity (identifying hotspots)...")
                buckets = bucket_max_severity(self.db_path, bucket_seconds=300)
                tools_used.append("bucket_max_severity")
                if self._record_additional_data(all_additional_data, "bucket_max_severity", buckets):
                    context_dirty = True
                print(f"   Found {len(buckets)} buckets with severity >= 20")
                
                # Track inspected buckets
                for bucket in buckets[:10]:  # Track top 10 buckets
                    if "bucket_start_epoch" in bucket:
                        inspected_buckets.append(bucket["bucket_start_epoch"])
                
                # 6. Recovery timeline (full span)
                print("\n6️⃣  Recovery Timeline (full span)...")
                recovery_timeline = get_recovery_timeline(self.db_path)
                tools_used.append("get_recovery_timeline")
                if self._record_additional_data(all_additional_data, "recovery_timeline", recovery_timeline):
                    context_dirty = True
                print(f"   Found {len(recovery_timeline)} recovery events")
                
                # 7. Time span
                print("\n7️⃣  Time Span...")
                time_span = get_time_span(self.db_path)
                tools_used.append("get_time_span")
                if self._record_additional_data(all_additional_data, "time_span", time_span):
                    context_dirty = True
                print(f"   Span: {time_span.get('earliest')} to {time_span.get('latest')}")
                print(f"   Duration: {time_span.get('span_seconds', 0) / 3600:.2f} hours")
                
                # Move to Phase B after global sweep
                phase = "B"
                print(f"\n✅ Phase A Complete. Moving to Phase B (Focused Drill-Down)")
            
            analysis = {
                "hypothesis": hypothesis or "",
                "confidence": confidence,
                "reasoning": reasoning,
                "suggested_tools": [],
                "next_steps": ""
            }
            
            if llm_calls >= self.max_llm_calls:
                print(f"\n🤖 Skipping LLM: call budget {self.max_llm_calls} exhausted.")
            elif not context_dirty:
                print("\n🤖 Skipping LLM: no new context since last analysis.")
            else:
                events_text = self._format_events_for_llm(all_events[:100])  # Limit to 100 for analysis
                
                if all_additional_data:
                    additional_text = "\n\nAdditional Investigation Data:\n" + "\n".join(
                        [f"\n{tool_name}:\n{json.dumps(data, indent=2, default=str)}"
                         for tool_name, data in all_additional_data[-5:]]
                    )
                    events_text += additional_text
                
                events_text = self._enforce_context_limit(events_text)
                
                print(f"\n🤖 Analyzing with LLM (call {llm_calls + 1}/{self.max_llm_calls})...")
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
                llm_calls += 1
                context_dirty = False
            
            hypothesis = analysis["hypothesis"]
            confidence = analysis["confidence"]
            reasoning = analysis.get("reasoning", "")
            suggested_tools = analysis.get("suggested_tools", [])
            next_steps = analysis.get("next_steps", "")
            
            print(f"\n📊 Iteration {iteration} Results:")
            print(f"   Confidence: {confidence:.2f} (target: {self.confidence_threshold:.2f})")
            print(f"   Hypothesis: {hypothesis[:150]}...")
            
            # Check if we've reached target confidence
            if confidence >= self.confidence_threshold:
                # Before finalizing, do Phase C: Coverage Check
                if phase != "C" and not global_summary_done:
                    print(f"\n🌐 Moving to Phase C: Coverage Check (before finalizing)")
                    phase = "C"
                else:
                    print(f"\n✅ Confidence threshold reached!")
                    break
            
            # Phase C: Coverage Check - verify all buckets were seen
            if phase == "C" or (iteration >= self.max_iterations - 2 and not global_summary_done):
                print(f"\n🔍 Phase C: Coverage Check - Verifying all buckets inspected...")
                
                # Get uncovered buckets
                uncovered = get_uncovered_buckets(self.db_path, inspected_buckets, bucket_seconds=600, min_severity=20)
                tools_used.append("get_uncovered_buckets")
                
                if uncovered:
                    print(f"   Found {len(uncovered)} uncovered buckets with severity >= 20")
                    # Inspect top uncovered bucket
                    top_uncovered = uncovered[0]
                    bucket_start = datetime.fromtimestamp(top_uncovered["bucket_start_epoch"])
                    bucket_end = bucket_start + timedelta(seconds=600)
                    
                    print(f"   Inspecting uncovered bucket: {bucket_start.isoformat()} (severity {top_uncovered['max_severity']})")
                    
                    # Get events from this bucket
                    bucket_events = get_events(
                        self.db_path,
                        start_time=bucket_start,
                        end_time=bucket_end,
                        limit=200
                    )
                    if self._append_events(all_events, bucket_events):
                        context_dirty = True
                    inspected_buckets.append(top_uncovered["bucket_start_epoch"])
                    print(f"   Added {len(bucket_events)} events from uncovered bucket")
                else:
                    print(f"   ✅ All buckets with severity >= 20 have been inspected")
                
                # Generate global summary (required before finalizing)
                if not global_summary_done:
                    print(f"\n📋 Generating Global Summary...")
                    summary = get_global_summary(self.db_path)
                    tools_used.append("get_global_summary")
                    if self._record_additional_data(all_additional_data, "global_summary", summary):
                        context_dirty = True
                    global_summary_done = True
                    print(f"   Max severity: {summary.get('max_severity')}")
                    print(f"   Recovery count: {summary.get('recovery_count')}")
                    print(f"   Top event types: {list(summary.get('top_5_event_types', {}).keys())}")
                
                phase = "B"  # Return to focused mode
            
            # Phase B: Focused drill-down on hotspots
            if phase == "B" and iteration < self.max_iterations:
                # Check if we should do coverage check first
                if iteration >= 3 and (iteration % 3 == 0):  # Every 3 iterations after initial
                    print(f"\n🔍 Escape Hatch: Returning to global view to check coverage...")
                    phase = "C"
                    continue  # Skip to coverage check
                
                new_tools_executed = False
                
                # Try suggested tools first
                if suggested_tools:
                    print(f"\n💡 LLM suggested tools: {', '.join(suggested_tools)}")
                    if next_steps:
                        print(f"📝 Next steps: {next_steps}")
                    
                    # Execute suggested tools (up to 3 at a time)
                    for tool_name in suggested_tools[:3]:  # Limit to 3 tools per iteration
                        if tool_name in tool_map and tool_name not in tools_used:
                            print(f"\n🔧 Executing tool: {tool_name}...")
                            try:
                                tool_func = tool_map[tool_name]
                                
                                # Call tool with db_path
                                if tool_name == "get_context_window":
                                    # Special case: needs around_time parameter
                                    if all_events and all_events[0].ts:
                                        result = tool_func(self.db_path, all_events[0].ts)
                                        if isinstance(result, list):
                                            if self._append_events(all_events, result):
                                                context_dirty = True
                                    else:
                                        result = tool_func(self.db_path, datetime.now())
                                else:
                                    result = tool_func(self.db_path)
                                
                                tools_used.append(tool_name)
                                new_tools_executed = True
                                
                                # Store results
                                if isinstance(result, list):
                                    # If it's a list of EventModel, add to events
                                    if result and isinstance(result[0], EventModel):
                                        if self._append_events(all_events, result):
                                            context_dirty = True
                                        print(f"   Added {len(result)} events from {tool_name}")
                                    else:
                                        # Store as additional data
                                        if self._record_additional_data(all_additional_data, tool_name, result):
                                            context_dirty = True
                                        print(f"   Stored results from {tool_name}")
                                else:
                                    # Store as additional data
                                    if self._record_additional_data(all_additional_data, tool_name, result):
                                        context_dirty = True
                                    print(f"   Stored results from {tool_name}")
                                    
                            except Exception as e:
                                print(f"   ⚠️  Error executing {tool_name}: {e}")
                
                # If no suggested tools or they're all used, try common detection tools
                if not new_tools_executed and confidence < 0.5:
                    print(f"\n🔍 Low confidence ({confidence:.2f}), trying common detection tools...")
                    common_tools = [
                        "detect_storage_engine_pressure",
                        "detect_recovery_loop",
                        "get_recovery_timeline",
                        "detect_ratekeeper_throttling",
                        "get_lag_series"
                    ]
                    
                    for tool_name in common_tools:
                        if tool_name in tool_map and tool_name not in tools_used:
                            print(f"\n🔧 Executing tool: {tool_name}...")
                            try:
                                tool_func = tool_map[tool_name]
                                result = tool_func(self.db_path)
                                
                                tools_used.append(tool_name)
                                new_tools_executed = True
                                
                                # Store as additional data
                                if self._record_additional_data(all_additional_data, tool_name, result):
                                    context_dirty = True
                                print(f"   Stored results from {tool_name}")
                                break  # Try one at a time
                                
                            except Exception as e:
                                print(f"   ⚠️  Error executing {tool_name}: {e}")
                
                # If still no new tools, try getting more events from different time ranges
                if not new_tools_executed and len(all_events) < 2000:
                    print(f"\n🔍 Gathering more events from different sources...")
                    try:
                        # Try getting events from later time periods
                        if all_events and all_events[-1].ts:
                            later_events = changes_after(self.db_path, all_events[-1].ts, limit=500)
                            if later_events:
                                if self._append_events(all_events, later_events):
                                    context_dirty = True
                                tools_used.append("changes_after")
                                new_tools_executed = True
                                print(f"   Added {len(later_events)} events from later time period")
                    except Exception as e:
                        print(f"   ⚠️  Error gathering more events: {e}")
                
                # If no new tools executed, try to find uncovered buckets
                if not new_tools_executed:
                    print(f"\n🔍 No suggested tools available, checking for uncovered buckets...")
                    uncovered = get_uncovered_buckets(self.db_path, inspected_buckets, bucket_seconds=600, min_severity=20)
                    
                    if uncovered:
                        top_uncovered = uncovered[0]
                        bucket_start = datetime.fromtimestamp(top_uncovered["bucket_start_epoch"])
                        bucket_end = bucket_start + timedelta(seconds=600)
                        
                        print(f"   Found uncovered bucket: {bucket_start.isoformat()} (severity {top_uncovered['max_severity']})")
                        bucket_events = get_events(
                            self.db_path,
                            start_time=bucket_start,
                            end_time=bucket_end,
                            limit=200
                        )
                        if bucket_events:
                            if self._append_events(all_events, bucket_events):
                                context_dirty = True
                            inspected_buckets.append(top_uncovered["bucket_start_epoch"])
                            tools_used.append("get_events")
                            new_tools_executed = True
                            print(f"   Added {len(bucket_events)} events from uncovered bucket")
                    
                    if not new_tools_executed:
                        print(f"\n⚠️  No new tools/data available. Confidence: {confidence:.2f}")
                        # Move to coverage check before stopping
                        phase = "C"
                        continue
                    else:
                        if iteration >= self.max_iterations:
                            print(f"\n⚠️  Reached max iterations ({self.max_iterations})")
                            break
        
        print(f"\n{'='*70}")
        print(f"✅ Investigation Complete")
        print(f"{'='*70}")
        print(f"   Final Hypothesis: {hypothesis}")
        print(f"   Final Confidence: {confidence:.2f}")
        print(f"   Reasoning: {reasoning}")
        print(f"   Tools Used: {', '.join(tools_used)}")
        print(f"   Iterations: {iteration}")
        
        evidence_events = all_events[:50]
        
        # Print evidence events
        print(f"\n{'='*70}")
        print(f"📋 Evidence Events ({len(evidence_events)} events)")
        print(f"{'='*70}")
        
        if evidence_events:
            # Sort by severity first
            sorted_evidence = sorted(evidence_events, key=lambda e: (
                -(e.severity or 0) if (e.severity or 0) >= 40 else -1000,
                e.ts if e.ts else datetime.max
            ))
            
            for i, event in enumerate(sorted_evidence, 1):
                timestamp_str = event.ts.isoformat() if event.ts else "N/A"
                role_str = event.role or "N/A"
                
                severity_label = ""
                if (event.severity or 0) >= 40:
                    severity_label = " ⚠️ ERROR"
                elif (event.severity or 0) == 20:
                    severity_label = " ⚠️ WARNING"
                
                print(f"\n[{i}] Event: {event.event}{severity_label}")
                print(f"     Timestamp: {timestamp_str}")
                print(f"     Severity: {event.severity}")
                print(f"     Role: {role_str}")
                
                if event.fields_json:
                    # Show key fields (limit to important ones)
                    key_fields = {}
                    for key, value in list(event.fields_json.items())[:5]:
                        # Truncate long values
                        val_str = str(value)
                        if len(val_str) > 100:
                            val_str = val_str[:100] + "..."
                        key_fields[key] = val_str
                    
                    if key_fields:
                        print(f"     Key Fields:")
                        for key, value in key_fields.items():
                            print(f"       {key}: {value}")
                    print()
                else:
                    print("   No evidence events found.\n")
        
        return InvestigationResult(
            hypothesis=hypothesis,
            confidence=confidence,
            reasoning=reasoning,
            tools_used=tools_used,
            evidence_events=evidence_events
        )

