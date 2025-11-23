"""
Investigation Agent - Iterative query-based log analysis.

Purpose: Query DuckDB iteratively until root cause is found with high confidence.
"""

import json
import os
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

import google.generativeai as genai

from data_transfer_object.event_dto import EventModel
from tools.database import get_conn


# Helper functions for EventModel conversion
def _ensure_dict(maybe_json) -> dict:
    """Ensure value is a dictionary."""
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


def _uuid_to_int(u) -> int:
    """
    Convert DuckDB UUID (or str UUID) to a stable 64-bit-ish int that fits typical int fields.
    """
    try:
        if isinstance(u, uuid.UUID):
            n = u.int
        else:
            n = uuid.UUID(str(u)).int
        # Squeeze to signed 63-bit range to be safe for pydantic/int
        return n & ((1 << 63) - 1)
    except Exception:
        return 0  # fallback


def load_events_window(
    db_path: str,
    start_ts: Optional[str] = None,
    end_ts: Optional[str] = None,
    limit: Optional[int] = None,
) -> List[EventModel]:
    """
    Read events from DuckDB and return EventModel objects.
    We hydrate required fields and coerce JSON where needed.
    """
    con = get_conn(db_path)

    where = []
    params = []
    if start_ts:
        where.append("ts >= ?")
        params.append(start_ts)
    if end_ts:
        where.append("ts < ?")
        params.append(end_ts)
    where_clause = ("WHERE " + " AND ".join(where)) if where else ""
    lim = f"LIMIT {int(limit)}" if limit else ""

    q = f"""
      SELECT event_id, event, ts, severity, role, fields_json
      FROM events
      {where_clause}
      ORDER BY ts ASC
      {lim}
    """
    rows = con.execute(q, params).fetchall()

    out: List[EventModel] = []
    for rid, event, ts, sev, role, fields in rows:
        fdict = _ensure_dict(fields)

        raw_blob = {
            "id": str(rid) if rid is not None else None,
            "event": event,
            "ts": ts.isoformat() if ts else None,
            "severity": sev,
            "role": role,
            "fields_json": fdict,
        }

        em = EventModel(
            # Core fields
            event=event,
            ts=ts,
            severity=sev,
            role=role,
            fields_json=fdict,

            # EventModel-required fields
            event_id=_uuid_to_int(rid),  # int, derived from UUID
            process=role or None,  # best-effort mapping
            pid=None,
            machine_id=None,
            address=None,
            trace_file=None,
            src_line=None,
            raw_json=raw_blob,  # dict, not string
        )
        out.append(em)

    return out


@dataclass
class InvestigationContext:
    """Context passed between agent iterations."""
    db_path: str
    max_iterations: int = 10
    hypothesis: Optional[str] = None
    confidence: float = 0.0
    iteration: int = 0
    query_history: List = field(default_factory=list)
    focus_time_start: Optional[datetime] = None
    focus_time_end: Optional[datetime] = None
    initial_question: Optional[str] = None
    next_strategy: Optional[str] = None
    custom_sql: Optional[str] = None
    seen_event_ids: set = field(default_factory=set)  # Track event IDs we've already seen
    tried_strategies: set = field(default_factory=set)  # Track strategies we've tried
    consecutive_repeats: int = 0  # Track consecutive repeated query detections
    log_file_paths: List[str] = field(default_factory=list)  # Paths to log files for grep search
    grep_pattern: Optional[str] = None  # Pattern to search for in files
    exploration_data: Dict[str, Any] = field(default_factory=dict)  # Store exploration results (dir listings, samples, etc.)
    metric_anomalies: List[Dict[str, Any]] = field(default_factory=list)  # Store detected metric anomalies

@dataclass
class QueryResult:
    """Result from a single SQL query."""
    query: str
    strategy: str
    events: List[EventModel]
    event_count: int
    timestamp: datetime


class QueryGenerator:
    """Generates SQL queries based on strategies."""
    def __init__(self, default_limit: int = 1000, ):
        self.default_limit = default_limit
    
    def generate(self, strategy: str, context: InvestigationContext, limit: Optional[int] = None, custom_sql: Optional[str] = None) -> str:
        """
        Generate SQL query based on strategy or custom SQL.
        
        Args:
            strategy: Strategy name or 'custom' for custom SQL
            context: Investigation context
            limit: Optional limit override
            custom_sql: Custom SQL query (if strategy is 'custom')
        
        Returns:
            SQL query string
        """
        # If strategy is 'custom', use the provided SQL (with safety validation)
        if strategy == "custom" and custom_sql:
            # Basic safety check - only allow SELECT statements
            sql_upper = custom_sql.strip().upper()
            if not sql_upper.startswith("SELECT"):
                raise ValueError("Custom SQL must be a SELECT statement only")
            if any(dangerous in sql_upper for dangerous in ["DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "CREATE", "TRUNCATE"]):
                raise ValueError("Custom SQL cannot contain DROP, DELETE, UPDATE, INSERT, ALTER, CREATE, or TRUNCATE")
            
            # Normalize SELECT * to use specific columns expected by _execute_query
            import re
            # Match SELECT * or SELECT* (with optional whitespace) before FROM
            custom_sql = re.sub(
                r'\bSELECT\s+\*\s+FROM',
                'SELECT event_id, ts, severity, event, role, fields_json FROM',
                custom_sql,
                flags=re.IGNORECASE
            )
            custom_sql = re.sub(
                r'\bSELECT\s+\*',
                'SELECT event_id, ts, severity, event, role, fields_json',
                custom_sql,
                flags=re.IGNORECASE
            )
            
            # Add time filter if context has time window
            time_filter = self._build_time_filter(context)
            if time_filter:
                # Need to insert time filter before ORDER BY/LIMIT clauses
                # Find positions of ORDER BY and LIMIT (case-insensitive search)
                import re
                sql_lower = custom_sql.lower()
                
                # Find ORDER BY position
                order_by_match = re.search(r'\border\s+by\b', sql_lower)
                order_by_pos = order_by_match.start() if order_by_match else -1
                
                # Find LIMIT position
                limit_match = re.search(r'\blimit\s+', sql_lower)
                limit_pos = limit_match.start() if limit_match else -1
                
                # Find WHERE position
                where_match = re.search(r'\bwhere\b', sql_lower)
                where_pos = where_match.start() if where_match else -1
                
                # Determine where to insert the time filter
                # Insert before ORDER BY or LIMIT, whichever comes first
                insert_pos = len(custom_sql)
                if order_by_pos >= 0:
                    insert_pos = min(insert_pos, order_by_pos)
                if limit_pos >= 0:
                    insert_pos = min(insert_pos, limit_pos)
                
                # Build the modified query
                if where_pos >= 0 and insert_pos > where_pos:
                    # WHERE clause exists, add AND condition before ORDER BY/LIMIT
                    before_filter = custom_sql[:insert_pos].rstrip()
                    after_filter = custom_sql[insert_pos:]
                    # Remove trailing semicolon if present
                    before_filter = before_filter.rstrip(";")
                    # Check if there's already an AND/OR at the end
                    before_filter = before_filter.rstrip()
                    if not before_filter.endswith(("AND", "OR")) and not before_filter.endswith("("):
                        custom_sql = f"{before_filter} AND {time_filter} {after_filter}"
                    else:
                        custom_sql = f"{before_filter} {time_filter} {after_filter}"
                else:
                    # No WHERE clause, add WHERE before ORDER BY/LIMIT
                    before_where = custom_sql[:insert_pos].rstrip()
                    after_where = custom_sql[insert_pos:]
                    before_where = before_where.rstrip(";")
                    custom_sql = f"{before_where} WHERE {time_filter} {after_where}"
            
            # Ensure limit if not present
            sql_upper = custom_sql.upper()
            if "LIMIT" not in sql_upper:
                query_limit = limit if limit is not None else self.default_limit
                custom_sql = f"{custom_sql} LIMIT {query_limit}"
            
            return custom_sql.strip()
        
        # Use predefined strategies
        if strategy == "find_recovery":
            return self._generate_recovery_query(context, limit)
        elif strategy == "find_high_severity":
            return self._generate_severity_query(context, limit)
        elif strategy == "find_time_window":
            return self._generate_time_window_query(context, limit)
        elif strategy == "find_connection_errors":
            return self._generate_connection_errors_query(context, limit)
        else:
            available = ["find_recovery", "find_high_severity", "find_time_window", "find_connection_errors", "custom"]
            raise ValueError(f"Unknown strategy: {strategy}. Available: {available}. Use 'custom' with custom_sql parameter for custom queries.")
    
    def _build_where_clause(self, conditions: List[str]) -> str:
        
        if not conditions:
            return ""
    
        return "WHERE " + " AND ".join(conditions)
    
    def _build_time_filter(self, context: InvestigationContext) -> str:
        if context.focus_time_start and context.focus_time_end:
            start_str = context.focus_time_start.strftime('%Y-%m-%d %H:%M:%S')
            end_str = context.focus_time_end.strftime('%Y-%m-%d %H:%M:%S')
            return f"ts BETWEEN '{start_str}' AND '{end_str}'"
        elif context.focus_time_start:
            start_str = context.focus_time_start.strftime('%Y-%m-%d %H:%M:%S')
            return f"ts >= '{start_str}'"
        elif context.focus_time_end:
            end_str = context.focus_time_end.strftime('%Y-%m-%d %H:%M:%S')
            return f"ts <= '{end_str}'"
        else:
            return ""
    
    def _generate_recovery_query(self, context: InvestigationContext, limit: Optional[int] = None) -> str:
        conditions = []

        conditions.append("event = 'MasterRecoveryState'")
        time_filter = self._build_time_filter(context)

        if time_filter:
            conditions.append(time_filter)
        
        where_clause = self._build_where_clause(conditions)

        query_limit = limit if limit is not None else self.default_limit

        query = f"""
        SELECT event_id, ts, severity, event, role, fields_json
        FROM events
        {where_clause}
        ORDER BY ts DESC
        LIMIT {query_limit}
        """
        return " ".join(query.split())
    
    def _generate_severity_query(self, context: InvestigationContext, limit: Optional[int] = None, severity_threshold: int = 40) -> str:
        """
        Generate severity query - prioritize Severity 40+ errors, not warnings.
        Default threshold changed from 30 to 40 to focus on actual errors.
        """
        conditions = []
        conditions.append(f"severity >= {severity_threshold}")
        
        time_filter = self._build_time_filter(context)
        if time_filter:
            conditions.append(time_filter)
    
        where_clause = self._build_where_clause(conditions)
    
        query_limit = limit if limit is not None else self.default_limit
    
        query = f"""
            SELECT event_id, ts, severity, event, role, fields_json
            FROM events
            {where_clause}
            ORDER BY severity DESC, ts DESC
            LIMIT {query_limit}
        """
        
        return " ".join(query.split())
    
    def _generate_time_window_query(self, context: InvestigationContext, limit: Optional[int] = None) -> str:
        
        if not context.focus_time_start and not context.focus_time_end:
            raise ValueError("Time window query requires focus_time_start or focus_time_end in context")
    
        conditions = []
    
        time_filter = self._build_time_filter(context)
        if time_filter:
            conditions.append(time_filter)
    
        where_clause = self._build_where_clause(conditions)
    
        query_limit = limit if limit is not None else self.default_limit
        
        query = f"""
            SELECT event_id, ts, severity, event, role, fields_json
            FROM events
            {where_clause}
            ORDER BY ts DESC
            LIMIT {query_limit}
        """
    
        return " ".join(query.split())
    
    def _generate_connection_errors_query(self, context: InvestigationContext, limit: Optional[int] = None) -> str:
        """
        Generate query to find high severity errors that might indicate operational issues.
        Prioritizes Severity 40+ errors, not warnings.
        """
        # Don't hardcode event names - let LLM discover what's important
        # Instead, look for high severity events that might indicate problems
        conditions = []
        conditions.append(f"severity >= 40")  # Changed from 20 to 40 - focus on actual errors, not warnings
        
        time_filter = self._build_time_filter(context)
        if time_filter:
            conditions.append(time_filter)
        
        where_clause = self._build_where_clause(conditions)
        query_limit = limit if limit is not None else self.default_limit
        
        query = f"""
            SELECT event_id, ts, severity, event, role, fields_json
            FROM events
            {where_clause}
            ORDER BY severity DESC, ts DESC
            LIMIT {query_limit}
        """
        return " ".join(query.split())


class InvestigationAgent:
    """Main agent that iteratively investigates log issues."""
    
    def __init__(self, db_path: str, max_iterations: int = 10, confidence_threshold: float = 0.8):
        """
        Initialize investigation agent.
        
        Args:
            db_path: Path to DuckDB database
            max_iterations: Maximum number of investigation iterations
            confidence_threshold: Confidence level required to stop investigation
        """
        self.db_path = db_path
        self.max_iterations = max_iterations
        self.confidence_threshold = confidence_threshold
        self.query_generator = QueryGenerator()
    
    def _explore_directory(self, directory_path: str) -> Dict[str, Any]:
        """
        Explore directory to find trace/log files.
        
        Args:
            directory_path: Path to directory to explore
            
        Returns:
            Dictionary with file information
        """
        from pathlib import Path
        
        dir_path = Path(directory_path)
        if not dir_path.exists() or not dir_path.is_dir():
            return {"error": f"Directory not found: {directory_path}", "files": []}
        
        # Find log/trace files
        log_extensions = {'.json', '.xml', '.txt', '.log', '.trace'}
        files = []
        
        for file_path in sorted(dir_path.iterdir()):
            if file_path.is_file() and file_path.suffix.lower() in log_extensions:
                try:
                    size = file_path.stat().st_size
                    files.append({
                        "path": str(file_path),
                        "name": file_path.name,
                        "size": size,
                        "size_mb": round(size / (1024 * 1024), 2),
                        "extension": file_path.suffix.lower()
                    })
                except Exception:
                    continue
        
        return {
            "directory": str(dir_path),
            "file_count": len(files),
            "files": files[:50]  # Limit to first 50 files
        }
    
    def _read_file_sample(self, file_path: str, sample_size: int = 2000, from_end: bool = False) -> Dict[str, Any]:
        """
        Read sample lines from a file (either from start or end).
        
        Args:
            file_path: Path to file
            sample_size: Number of lines to read
            from_end: If True, read from end of file
            
        Returns:
            Dictionary with sample lines and metadata
        """
        from pathlib import Path
        
        path = Path(file_path)
        if not path.exists():
            return {"error": f"File not found: {file_path}", "lines": []}
        
        try:
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                total_lines = sum(1 for _ in f)
                f.seek(0)
                
                if from_end:
                    # Read from end - skip to near the end
                    lines_to_skip = max(0, total_lines - sample_size)
                    for _ in range(lines_to_skip):
                        f.readline()
                    lines = [line.strip() for line in f.readlines()]
                else:
                    # Read from start
                    lines = [f.readline().strip() for _ in range(min(sample_size, total_lines))]
                
                # Analyze structure
                structure = {
                    "is_json": False,
                    "is_xml": False,
                    "has_severity": False,
                    "has_type": False,
                    "has_metrics": False,
                }
                
                for line in lines[:100]:  # Check first 100 lines
                    if not line:
                        continue
                    if line.strip().startswith('{') and '"' in line:
                        structure["is_json"] = True
                    if '<Event' in line or '<event' in line:
                        structure["is_xml"] = True
                    if '"Severity"' in line or '"severity"' in line or 'Severity=' in line:
                        structure["has_severity"] = True
                    if '"Type"' in line or '"type"' in line or 'Type=' in line:
                        structure["has_type"] = True
                    if 'Metrics' in line or 'metrics' in line:
                        structure["has_metrics"] = True
                
                return {
                    "file": str(path),
                    "total_lines": total_lines,
                    "sample_lines": len(lines),
                    "from_end": from_end,
                    "structure": structure,
                    "lines": lines[:100] if len(lines) <= 100 else lines[:50] + ["...", f"[{len(lines) - 100} more lines]", "..."] + lines[-50:]
                }
        except Exception as e:
            return {"error": str(e), "lines": []}
    
    def _scan_metrics_anomalies(self, events: List[EventModel]) -> List[Dict[str, Any]]:
        """
        Scan events for metric anomalies (high values, negative values, performance issues).
        Prioritizes metrics that indicate actual performance problems, not startup warnings.
        
        Args:
            events: List of events to scan
            
        Returns:
            List of anomaly descriptions, sorted by severity (most critical first)
        """
        anomalies = []
        
        # Look for metric events (StorageMetrics, DiskMetrics, etc.)
        metric_events = [e for e in events if e.fields_json and isinstance(e.fields_json, dict)]
        
        for event in metric_events:
            fields = event.fields_json
            issues = []
            severity_score = 0  # Track severity of anomalies
            
            # Check for extreme values
            for key, value in fields.items():
                try:
                    # Try to parse numeric value
                    if isinstance(value, (int, float)):
                        num_val = float(value)
                    elif isinstance(value, str):
                        # Handle strings that might be numeric
                        clean_val = value.replace(',', '').strip()
                        num_val = float(clean_val) if clean_val.replace('.', '').replace('-', '').isdigit() else None
                    else:
                        num_val = None
                    
                    if num_val is None:
                        continue
                    
                    # CRITICAL: Check for negative latencies (indicates clock skew or serious timing issues)
                    if 'latency' in key.lower() or 'lag' in key.lower() or 'grv' in key.lower():
                        if num_val < 0:
                            issues.append(f"CRITICAL: Negative {key}: {num_val}")
                            severity_score += 100  # Highest priority
                        # Check for extreme VersionLag (indicates storage server lagging)
                        if 'lag' in key.lower() or 'versionlag' in key.lower():
                            if num_val > 90000:  # Very high lag
                                issues.append(f"CRITICAL: Extreme VersionLag: {num_val}")
                                severity_score += 80
                            elif num_val > 50000:  # High lag
                                issues.append(f"HIGH: High VersionLag: {num_val}")
                                severity_score += 50
                        
                        # Check for high latencies
                        if 'latency' in key.lower():
                            if num_val > 5000:  # > 5 seconds
                                issues.append(f"HIGH: Very high latency {key}: {num_val}ms")
                                severity_score += 40
                            elif num_val > 1000:  # > 1 second
                                issues.append(f"MEDIUM: High latency {key}: {num_val}ms")
                                severity_score += 20
                    
                    # Check for SlowSSLoop (performance degradation indicator)
                    if 'slow' in key.lower() and ('loop' in key.lower() or 'ss' in key.lower()):
                        if num_val > 0:  # Any slow loop count is significant
                            issues.append(f"HIGH: SlowSSLoop detected: {key}={num_val}")
                            severity_score += 60
                    
                    # Check for large queues (indicates backpressure)
                    if 'queue' in key.lower():
                        if num_val > 1000:
                            issues.append(f"HIGH: Large queue {key}: {num_val}")
                            severity_score += 30
                        elif num_val > 500:
                            issues.append(f"MEDIUM: Queue building {key}: {num_val}")
                            severity_score += 15
                    
                    # Check for other performance indicators
                    if 'stall' in key.lower() or 'throttle' in key.lower():
                        if num_val > 0:
                            issues.append(f"MEDIUM: Performance issue {key}: {num_val}")
                            severity_score += 25
                            
                except (ValueError, AttributeError, TypeError):
                    continue
            
            if issues:
                anomalies.append({
                    "event": event.event,
                    "timestamp": event.ts.isoformat() if event.ts else None,
                    "severity": event.severity,
                    "issues": issues,
                    "severity_score": severity_score,  # For sorting
                    "fields": {k: v for k, v in list(fields.items())[:10]}  # First 10 fields
                })
        
        # Sort by severity score (most critical first)
        anomalies.sort(key=lambda x: x.get("severity_score", 0), reverse=True)
        
        return anomalies
    
    def _grep_files(self, pattern: str, file_paths: List[str], context: InvestigationContext, max_results: int = 100) -> List[EventModel]:
        """
        Search log files using regex pattern (like grep).
        
        Args:
            pattern: Regex pattern to search for in log files
            file_paths: List of log file paths to search
            context: Investigation context for time filtering
            max_results: Maximum number of matching lines to return
            
        Returns:
            List of EventModel instances from matching lines
        """
        if not file_paths:
            return []
        
        events: List[EventModel] = []
        from tools.parser import LogParser
        parser = LogParser()
        
        try:
            # Compile regex pattern
            regex = re.compile(pattern, re.IGNORECASE | re.MULTILINE)
        except re.error as e:
            print(f"⚠️  Invalid regex pattern '{pattern}': {e}")
            return []
        
        matched_lines = []
        
        # Search through all log files
        for file_path in file_paths:
            path = Path(file_path)
            if not path.exists():
                continue
            
            try:
                with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                    for line_num, line in enumerate(f, start=1):
                        if regex.search(line):
                            matched_lines.append((str(path), line_num, line.strip()))
                            if len(matched_lines) >= max_results:
                                break
                        
                        if len(matched_lines) >= max_results:
                            break
            except Exception as e:
                print(f"⚠️  Error reading file {file_path}: {e}")
                continue
        
        # Parse matched lines into EventModel instances
        event_id_offset = 1000000  # Offset to avoid conflicts with DB event IDs
        for idx, (file_path, line_num, line_text) in enumerate(matched_lines):
            try:
                # Try to parse as JSON first
                try:
                    obj = json.loads(line_text)
                except json.JSONDecodeError:
                    # Not JSON, create a minimal event from the raw line
                    obj = {
                        "Type": "RawLogMatch",
                        "Severity": None,
                        "DateTime": None,
                        "_raw_line": line_text,
                        "_file": file_path,
                        "_line": line_num,
                    }
                
                # Parse datetime if present
                ts = None
                if "DateTime" in obj:
                    try:
                        ts = datetime.fromisoformat(obj["DateTime"].replace("Z", "+00:00"))
                    except Exception:
                        try:
                            ts = datetime.strptime(obj["DateTime"], "%Y-%m-%dT%H:%M:%SZ")
                        except Exception:
                            pass
                
                # Apply time filter if specified
                if context.focus_time_start and ts and ts < context.focus_time_start:
                    continue
                if context.focus_time_end and ts and ts > context.focus_time_end:
                    continue
                
                # Create EventModel
                event = EventModel(
                    event_id=event_id_offset + idx,
                    ts=ts,
                    severity=int(obj.get("Severity")) if obj.get("Severity") is not None else None,
                    event=obj.get("Type") or "LogMatch",
                    process=obj.get("Processes"),
                    role=obj.get("Roles"),
                    pid=int(obj["PID"]) if "PID" in obj and obj["PID"] is not None else None,
                    machine_id=obj.get("Machine") or obj.get("MachineId"),
                    address=obj.get("Address"),
                    trace_file=file_path,
                    src_line=line_num,
                    raw_json=obj,
                    fields_json={k: v for k, v in obj.items() if k not in ["Type", "Severity", "DateTime", "Processes", "Roles", "PID", "Machine", "MachineId", "Address"]},
                )
                events.append(event)
                
            except Exception as e:
                # Skip lines that can't be parsed
                continue
        
        return events
    
    def _execute_query(self, sql: str, db_path: str) -> List[EventModel]:

        con = get_conn(db_path)
        rows = con.execute(sql).fetchall()
        
        events: List[EventModel] = []
        # SQL query returns: event_id, ts, severity, event, role, fields_json
        for rid, ts, sev, event, role, fields in rows:
            fdict = _ensure_dict(fields)
            
            # Handle timestamp - DuckDB might return datetime or int (Unix timestamp)
            ts_datetime = None
            ts_iso = None
            if ts:
                if isinstance(ts, datetime):
                    ts_datetime = ts
                    ts_iso = ts.isoformat()
                elif isinstance(ts, (int, float)):
                    # Convert Unix timestamp to datetime
                    ts_datetime = datetime.fromtimestamp(ts)
                    ts_iso = ts_datetime.isoformat()
                else:
                    # Try to parse as datetime string if it's a string
                    try:
                        ts_datetime = ts if isinstance(ts, datetime) else None
                        ts_iso = ts_datetime.isoformat() if ts_datetime else str(ts)
                    except Exception:
                        ts_iso = str(ts)
            
            raw_blob = {
                "id": str(rid) if rid is not None else None,
                "event": event,
                "ts": ts_iso,
                "severity": sev,
                "role": role,
                "fields_json": fdict,
            }
            
            em = EventModel(
                # Core fields
                event=event,
                ts=ts_datetime,  # Use datetime object for EventModel
                severity=sev,
                role=role,
                fields_json=fdict,
                
                # EventModel-required fields
                event_id=_uuid_to_int(rid),
                process=role or None,
                pid=None,
                machine_id=None,
                address=None,
                trace_file=None,
                src_line=None,
                raw_json=raw_blob,
            )
            events.append(em)
        
        return events
    
    def _format_events_for_llm(self, events: List[EventModel], context: InvestigationContext) -> str:
        """
        Format EventModel list into readable text for LLM analysis.
        Prioritizes severity and temporal context to avoid misidentifying warnings as root causes.
        
        Args:
            events: List of EventModel objects from query
            context: Current investigation context
            
        Returns:
            Formatted string containing event details with severity and temporal context
        """
        if not events:
            return "No events found."
        
        # Sort events: prioritize Severity 40+ errors, then by timestamp
        sorted_events = sorted(events, key=lambda e: (
            -(e.severity or 0) if (e.severity or 0) >= 40 else -1000,  # Severity 40+ first
            e.ts if e.ts else datetime.max  # Then by timestamp
        ))
        
        # Group events by severity for context
        severity_40_plus = [e for e in sorted_events if (e.severity or 0) >= 40]
        severity_20_warnings = [e for e in sorted_events if (e.severity or 0) == 20]
        severity_10_info = [e for e in sorted_events if (e.severity or 0) == 10]
        
        # Limit to top 15 events, but ensure we show severity 40+ first
        display_events = sorted_events[:15]
        total_count = len(events)
        
        lines = []
        lines.append(f"Found {total_count} events:")
        lines.append(f"  - Severity 40+ (Errors): {len(severity_40_plus)}")
        lines.append(f"  - Severity 20 (Warnings): {len(severity_20_warnings)}")
        lines.append(f"  - Severity 10 (Info): {len(severity_10_info)}")
        lines.append("")
        lines.append("⚠️  REMEMBER: Severity 20 warnings (like FileNotFoundError) are NON-FATAL startup warnings, NOT root causes!")
        lines.append("   Focus on Severity 40+ errors OR metric anomalies that correlate with test context.\n")
        
        # Calculate time range for temporal context
        if events and any(e.ts for e in events):
            timestamps = [e.ts for e in events if e.ts]
            if timestamps:
                earliest = min(timestamps)
                latest = max(timestamps)
                time_span = (latest - earliest).total_seconds()
                lines.append(f"Time range: {earliest.isoformat()} to {latest.isoformat()} ({time_span:.1f} seconds)")
                lines.append("⚠️  Early events (<20s) are often startup/initialization, not root causes if problems build later.\n")
        
        lines.append(f"Showing top {len(display_events)} events (prioritized by Severity 40+, then chronologically):\n")
        
        for i, event in enumerate(display_events, 1):
            timestamp_str = event.ts.isoformat() if event.ts else "N/A"
            role_str = event.role or "N/A"
            
            # Calculate relative time if we have events with timestamps
            relative_time = ""
            if event.ts and events:
                timestamps = [e.ts for e in events if e.ts]
                if timestamps:
                    earliest = min(timestamps)
                    relative_seconds = (event.ts - earliest).total_seconds()
                    relative_time = f" (T+{relative_seconds:.1f}s)"
            
            # Highlight severity
            severity_indicator = ""
            if (event.severity or 0) >= 40:
                severity_indicator = " ⚠️ CRITICAL ERROR"
            elif (event.severity or 0) == 20:
                severity_indicator = " ⚠️ WARNING (likely non-fatal)"
            
            # Format fields_json (show top 5 keys if too many)
            fields_str = "N/A"
            if event.fields_json:
                if len(event.fields_json) <= 5:
                    fields_str = json.dumps(event.fields_json, indent=2)
                else:
                    # Show first 5 keys
                    top_fields = dict(list(event.fields_json.items())[:5])
                    fields_str = json.dumps(top_fields, indent=2) + "\n    ... (truncated)"
            
            event_block = f"""
Event {i}{severity_indicator}:
  Timestamp: {timestamp_str}{relative_time}
  Event Type: {event.event}
  Severity: {event.severity} ({'ERROR' if (event.severity or 0) >= 40 else 'WARNING' if (event.severity or 0) == 20 else 'INFO'})
  Role: {role_str}
  Fields:
{fields_str}
"""
            lines.append(event_block)
        
        if total_count > len(display_events):
            lines.append(f"\n... and {total_count - len(display_events)} more events (truncated for brevity)")
        
        return "\n".join(lines)
    
    def _analyze_with_llm(
        self, 
        events_text: str, 
        context: InvestigationContext,
        api_key: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Send formatted event text and context to LLM and get analysis.
        
        Args:
            events_text: Formatted event text from _format_events_for_llm()
            context: Current investigation context
            api_key: Gemini API key (or uses GEMINI_API_KEY env var)
            
        Returns:
            Dict with keys: 'hypothesis', 'confidence', 'next_strategy', 'reasoning'
        """
        # Initialize Gemini
        api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not set. Provide api_key or set GEMINI_API_KEY env var.")
        
        try:
            genai.configure(api_key=api_key)
        except Exception:
            pass  # Already configured
        
        # Use gemini-2.5-flash model
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        # Build prompt
        prompt_parts = []
        prompt_parts.append("You are an expert at analyzing FoundationDB log events to identify the SPECIFIC ISSUE being tested in simulations.")
        prompt_parts.append("\n=== INVESTIGATION CONTEXT ===")
        if context.initial_question:
            prompt_parts.append(f"Initial Question: {context.initial_question}")
        if context.hypothesis:
            prompt_parts.append(f"Current Hypothesis: {context.hypothesis}")
            prompt_parts.append(f"Current Confidence: {context.confidence:.2f}")
        prompt_parts.append(f"Iteration: {context.iteration + 1}/{context.max_iterations}")
        
        prompt_parts.append("\n=== EVENTS FOUND ===")
        prompt_parts.append(events_text)
        
        # Add exploration data if available
        if context.exploration_data:
            prompt_parts.append("\n=== EXPLORATION DATA ===")
            if 'directory_listing' in context.exploration_data:
                dir_info = context.exploration_data['directory_listing']
                prompt_parts.append(f"Directory: {dir_info.get('directory', 'N/A')}")
                prompt_parts.append(f"Files found: {dir_info.get('file_count', 0)}")
                if dir_info.get('files'):
                    prompt_parts.append("Sample files:")
                    for f in dir_info['files'][:10]:
                        prompt_parts.append(f"  - {f.get('name', 'N/A')} ({f.get('size_mb', 0)} MB)")
            
            # Add file sample info
            for key, value in context.exploration_data.items():
                if key.startswith('sample_') and isinstance(value, dict):
                    if 'structure' in value:
                        struct = value['structure']
                        prompt_parts.append(f"\nFile sample ({key}):")
                        prompt_parts.append(f"  Structure: JSON={struct.get('is_json')}, XML={struct.get('is_xml')}")
                        prompt_parts.append(f"  Has Severity={struct.get('has_severity')}, Type={struct.get('has_type')}, Metrics={struct.get('has_metrics')}")
            
            # Add metric anomalies
            if context.metric_anomalies:
                prompt_parts.append(f"\n=== METRIC ANOMALIES DETECTED ===")
                prompt_parts.append(f"Found {len(context.metric_anomalies)} anomalies:")
                for i, anomaly in enumerate(context.metric_anomalies[:10], 1):
                    prompt_parts.append(f"  {i}. {anomaly.get('event', 'N/A')} - {', '.join(anomaly.get('issues', [])[:3])}")
            
            # Add temporal analysis
            if 'temporal_analysis' in context.exploration_data:
                temporal = context.exploration_data['temporal_analysis']
                if temporal.get('temporal_findings'):
                    prompt_parts.append(f"\n=== TEMPORAL ANALYSIS ===")
                    for finding in temporal['temporal_findings']:
                        if finding.get('found'):
                            prompt_parts.append(f"  Found {finding.get('event')} in {Path(finding.get('file', '')).name}")
        
        prompt_parts.append("\n=== ANALYSIS REQUEST ===")
        prompt_parts.append("""
Your goal is to analyze FoundationDB simulation/test events to identify the SPECIFIC ISSUE being tested.

CONTEXT: These are FoundationDB simulation/test events. Each simulation tests a specific scenario or bug.

CRITICAL: Event names are often SYMPTOMS or RESPONSES, not the root cause!
- Events like "FKReenableLB" mean something CAUSED load balancing to be re-enabled
- Events like "MasterRecoveryState" mean something TRIGGERED a recovery
Your job is to find WHAT FAILED/ERRORED that caused these events, not just describe the events themselves.

WHAT TO LOOK FOR (in order of priority):
1. SEVERITY 40+ ERRORS (CRITICAL): Prioritize actual errors (Severity >= 40), NOT warnings (Severity 20)
   - FDB Severity levels: 10=info, 20=warning, 40=error, 50=severe
   - FileNotFoundError at Severity 20 is a NON-FATAL startup warning, NOT a root cause
   - Focus on Severity 40+ errors that indicate actual failures
   
2. METRIC ANOMALIES (HIGH PRIORITY): Scan StorageMetrics and other metric events for:
   - High VersionLag (>90k indicates severe lag)
   - Negative latencies (indicates clock skew or timing issues)
   - SlowSSLoopx100 (indicates performance degradation)
   - Large queues, stalls, or performance regressions
   - These are often the REAL performance issues, not surface errors
   
3. TEMPORAL PROGRESSION: Consider WHEN events occur
   - Early events (first 10-20 seconds) are often startup/initialization, not root causes
   - Problems that BUILD OVER TIME (lags, slow loops appearing later) are more likely root causes
   - If FileNotFoundError occurs at Time 4-14 but problems build later, it's NOT the root cause
   
4. TEST CONTEXT CORRELATION: Link findings to what the test is actually testing
   - Look for patterns that match the test scenario, not just any error
   - Performance degradation + test context = more likely root cause than isolated early warnings
   
5. Actual Failures/Errors (Severity 40+): ConnectionTimedOut, ConnectionFailed, Error, Exception, Fail, Timeout
6. Error fields_json: Look for error codes, error messages, failure reasons (but prioritize by severity)
7. Preceding events: What happened BEFORE current events? (but consider temporal context)


WHAT NOT TO DO (CRITICAL):
- ❌ Do NOT identify FileNotFoundError (Severity 20) as root cause - it's a non-fatal startup warning
- ❌ Do NOT over-rely on error keywords - "Error" in name doesn't mean it's fatal
- ❌ Do NOT ignore severity filtering - prioritize 40+ errors, not 20 warnings
- ❌ Do NOT skip metric analysis - VersionLag, negative latencies, SlowSSLoop are the real issues
- ❌ Do NOT ignore temporal context - early warnings ≠ root cause if problems build later
- ❌ Do NOT miss correlation - link anomalies to test context, not just surface errors
- ❌ Do NOT just repeat event names ("the test is FKReenableLB" = ❌ wrong)
- ❌ Do NOT conclude based on symptoms alone


ANALYZE EVENTS TO UNDERSTAND:
- What ACTUAL FAILURE occurred? (timeouts, disconnections, corruption, TLog crash)
- What triggered FKReenableLB, recovery, or stalls?
- What do fields_json reveal?
- What occurred BEFORE the visible symptoms?


----------------------------------------------------------------------
FOUNDATIONDB EVENT CLASSIFICATION TREE (MUST FOLLOW IN ORDER)
----------------------------------------------------------------------

LEVEL 0 — Fatal vs Non-Fatal
0.1 If ANY Severity >= 40 events exist → Fatal Path
     Else → Non-Fatal Path


----------------------------------------------------------------------
FATAL PATH (Cluster-breaking failures)
----------------------------------------------------------------------

1A — Recovery Loop Failure (Cluster 0)
Indicators:
- repeated “reading_coordinated_state”
- “Aborting recovery”
- recovery never reaches fully_recovered

1B — TLog Failure (Cluster 5)
Indicators:
- “TLogStopped”, “TLogError”
- missing TLog metrics
- StorageServer VersionLag extremely large

1C — Proxy/Resolver Crash (Cluster 7)
Indicators:
- “pipeline broken”
- resolver/proxy disappeared
- master forced restart

1D — Network Partition (Cluster 8)
Indicators:
- many ConnectionFailed/TimedOut events
- quorum unreachable

1E — Clock Skew (Cluster 10)
Indicators:
- “time moved backward”
- negative GRV or commit latencies
- timestamp anomalies

1F — Data Corruption (Cluster 11)
Indicators:
- checksum mismatch
- KVStoreError
- corruption detected

If none match → classify as Cluster 15: cascading failure.


----------------------------------------------------------------------
NON-FATAL PATH (Simulation passed but reveals issues)
----------------------------------------------------------------------

2A — Shard Movement / Data Distribution (FK*)
Events:
- FetchKeysBlock
- FetchKeysUnblocked
- FKBeforeFinalCommit
- FKAfterFinalCommit
- FKReenableLB

NOTE:
These are symptoms. They indicate shard movement is happening or thrashing.  
Root cause is always elsewhere → see LEVEL 3.

2B — Performance Degradation
Indicators:
- SlowSSLoopx100
- SlowTask
- VersionLag > 50k
- stalls in DD or storage

2C — Transaction Conflicts
Events:
- transaction_conflict
- past_version
- tag_throttled

2D — ReadClean / WriteDuringReadClean (RARE)
Only if these appear:
- WriteDuringReadClean
- ReadDuringClean
- SSCleanData
- ShardChangeError

If none appear → DO NOT classify as read-clean.

2E — Resource Exhaustion
Indicators:
- too many open files
- cannot allocate memory
- port exhaustion
- ENOMEM / EMFILE


----------------------------------------------------------------------
LEVEL 3 — Determining Root Cause Behind FKReenableLB (SYMPTOM)
----------------------------------------------------------------------

If FKReenableLB or heavy FK events exist, determine WHY:

3A — Storage Server is lagging behind TLogs
Indicators:
- VersionLag > 20k
- FK loops restarting
- negative GRV latency
ROOT: storage pressure, SS behind TLogs → Cluster 6 or 10.

3B — TLog starvation or missing TLogs
Indicators:
- FK loops but TLog metrics missing
- SS stuck at same version regardless of FK
ROOT: TLog problem → Cluster 5.

3C — Version corruption / time warp
Indicators:
- negative GRV or commit latency
- sudden version jumps
ROOT: clock skew or corruption → Cluster 10 or 11.

3D — Shard movement thrashing
Indicators:
- hundreds of FK cycles
- FKReenableLB spam
- FetchKeysBlock returns empty blocks
ROOT: DD thrashing → Cluster 6 (engine pressure).


----------------------------------------------------------------------
INVESTIGATION STRATEGIES
----------------------------------------------------------------------

Strategies the agent may choose:

1. find_high_severity  
   → search for severity >= 40 errors (ACTUAL ERRORS, not warnings)
   → DO NOT confuse with Severity 20 warnings (like FileNotFoundError) which are non-fatal

2. find_recovery  
   → search for recovery-state events that indicate failure

3. find_time_window  
   → inspect early window (often 0–10 seconds) for triggers

4. find_connection_errors  
   → search for Severity 40+ connection errors (ConnectionTimedOut, ConnectionFailed, ConnectionClosed)
   → Prioritizes actual errors, not warnings

5. custom  (RECOMMENDED)
   → SQL queries like:
     WHERE event LIKE '%Error%' OR event LIKE '%Timeout%'
     WHERE fields_json::string LIKE '%error%'
     WHERE fields_json::string LIKE '%write%' OR '%read%' OR '%timeout%'
     WHERE ts BETWEEN <start> AND <end>

6. grep
   → Search raw log files directly using regex patterns (like grep)
   → Use when you need to find specific text patterns that might not be indexed in DB
   → Provide grep_pattern in response (e.g., "error|Error|ERROR", "timeout.*failed", "Exception.*Connection")
   → Useful for finding error messages, stack traces, or raw log patterns

7. explore_directory
   → List directory contents to identify trace/log files
   → Use at the start of investigation to understand available files
   → Provide directory_path in response (optional, uses log_file_paths from context if not provided)

8. read_file_sample
   → Read sample lines from log files (first N lines or last N lines)
   → Use to understand log structure before deeper analysis
   → Provide file_path and from_end (true for end of file) in response

9. scan_metrics_anomalies
   → Scan events for metric anomalies (extreme values, negative latencies, performance issues)
   → Use after gathering events to identify performance problems
   → Looks for red flags: high lag values, negative latencies, large queues, slow loops

10. temporal_analysis
   → Analyze progression over time by checking file ends and completion events
   → Use to understand how the system progressed and if issues resolved
   → Checks for completion events and time-based patterns

11. done  
   → ONLY allowed once a TRUE root cause has been identified  
     (with confidence > 0.8)


----------------------------------------------------------------------
OUTPUT FORMAT (STRICT)
----------------------------------------------------------------------


Respond in JSON format:
{
    "hypothesis": "your hypothesis about what issue/scenario is being tested in this simulation",
    "confidence": 0.65,
    "next_strategy": "find_high_severity",
    "custom_sql": "SELECT event_id, ts, severity, event, role, fields_json FROM events WHERE event LIKE '%Error%' ORDER BY ts DESC LIMIT 100",
    "grep_pattern": "error|Error|ERROR|timeout|Timeout|Exception",
    "directory_path": "/path/to/logs",
    "file_path": "/path/to/log.json",
    "from_end": false,
    "focus_time_start": "2025-10-03 01:15:00",
    "focus_time_end": "2025-10-03 01:20:55",
    "reasoning": "brief explanation - what do these events indicate about what's being tested"
}


If next_strategy != custom → omit custom_sql.  
If next_strategy != find_time_window → omit focus_time_start and focus_time_end.
If next_strategy != grep → omit grep_pattern.
If next_strategy != explore_directory → omit directory_path.
If next_strategy != read_file_sample → omit file_path and from_end.

LOW CONFIDENCE (<0.6) WHEN:
- you only see symptoms (FKReenableLB, recovery events) without root cause
- you only see Severity 20 warnings (like FileNotFoundError) - these are non-fatal
- no actual Severity 40+ error/failure encountered
- no metric anomalies found (VersionLag, negative latencies, SlowSSLoop)
- early warnings without later correlation to problems

HIGH CONFIDENCE (>0.8) ONLY IF:
- you identified a specific Severity 40+ failure (e.g., TLog crash, timeout, corruption)
- OR you found metric anomalies (high VersionLag, negative latencies, SlowSSLoopx100) that correlate with test context
- AND you can explain HOW it relates to the test scenario (e.g., "writeduringreadclean" causing lags)
- AND temporal progression makes sense (anomalies building over time, not isolated early warnings)


----------------------------------------------------------------------
FOUNDATIONDB RECOVERY CLUSTER KNOWLEDGE BASE
----------------------------------------------------------------------

CLUSTER 0: recovery_restart_cascade  
- Problem: recovery loops due to coordinator state/generation mismatch  
- Causes: concurrent recovery attempts, stale metadata, clock skew  
- Indicators: repeated “reading_coordinated_state”, “Aborting recovery”

CLUSTER 4: transaction_tag_throttling  
- Problem: GRV blocked by exhausted tag budgets  
- Indicators: TAG_THROTTLED, GRV queue > 10k

CLUSTER 5: tlog_failure_recovery  
- Problem: TLog failure causes cluster to recover  
- Indicators: TLog failed/degraded message, missing TLog, partial commit

CLUSTER 6: storage_engine_pressure  
- Problem: storage engine under extreme load after recovery  
- Indicators: large txn mode, DiskQueue lag, I/O saturation

CLUSTER 7: commit_proxy_pipeline_crash  
- Problem: commit proxy or resolver crash  
- Indicators: “pipeline broken”, proxy/resolver gone

CLUSTER 8: network_partition_recovery  
- Problem: partition causing loss of quorum  
- Indicators: many connection failures

CLUSTER 9: configuration_change_recovery  
- Problem: unstable config change  
- Indicators: exclude/include ops, replication change, version mismatch

CLUSTER 10: clock_skew_recovery  
- Problem: time moves backward or huge drift  
- Indicators: negative latencies, inconsistent timestamps

CLUSTER 11: data_corruption_recovery  
- Problem: data corruption forces replica rebuild  
- Indicators: checksum mismatch, corruption, TLog replay errors

CLUSTER 12: resource_exhaustion_recovery  
- Problem: OS-level exhaustion  
- Indicators: EMFILE, ENOMEM, port exhaustion

CLUSTER 13: workload_spike_recovery  
- Problem: sudden spike in load  
- Indicators: retry storm, queue overflow, timeout cascades

CLUSTER 14: upgrade_rollback_recovery  
- Problem: broken upgrade/rollback  
- Indicators: mixed versions, protocol mismatch

CLUSTER 15: cascading_failure_recovery  
- Problem: chain reaction of failures  
- Indicators: multiple roles failing in sequence

CLUSTER 16: lease_expiration_recovery  
- Problem: master lease expiration  
- Indicators: failed lease renewal, lease_expired events


----------------------------------------------------------------------
FINAL INSTRUCTIONS
----------------------------------------------------------------------

You MUST follow the classification tree EXACTLY.
Never claim a root cause based on symptoms.
Only conclude “done” when an actual failure/error is found and justified.
""")
        
        prompt = "\n".join(prompt_parts)
        
        try:
            response = model.generate_content(prompt)
            response_text = response.text.strip()
            
            # Parse JSON from response (might have markdown code blocks)
            if "```json" in response_text:
                json_start = response_text.find("```json") + 7
                json_end = response_text.find("```", json_start)
                response_text = response_text[json_start:json_end].strip()
            elif "```" in response_text:
                json_start = response_text.find("```") + 3
                json_end = response_text.find("```", json_start)
                response_text = response_text[json_start:json_end].strip()
            
            result = json.loads(response_text)
            
            # Validate result
            analysis_result = {
                "hypothesis": result.get("hypothesis", ""),
                "confidence": float(result.get("confidence", 0.0)),
                "next_strategy": result.get("next_strategy", "done"),
                "reasoning": result.get("reasoning", ""),
            }
            
            # Extract custom SQL if provided for custom strategy
            if analysis_result["next_strategy"] == "custom":
                if "custom_sql" in result:
                    analysis_result["custom_sql"] = result["custom_sql"].strip()
                else:
                    # Fallback if custom strategy but no SQL provided
                    print("⚠️  Warning: Strategy is 'custom' but no custom_sql provided. Falling back to 'find_high_severity'.")
                    analysis_result["next_strategy"] = "find_high_severity"
                    analysis_result.pop("custom_sql", None)  # Remove custom_sql if not provided
            
            # Extract time window if provided for find_time_window strategy
            if analysis_result["next_strategy"] == "find_time_window":
                if "focus_time_start" in result:
                    try:
                        # Parse time string to datetime (handle both "2025-10-03 01:15:00" and ISO format)
                        time_str = result["focus_time_start"].replace(" ", "T")
                        if "T" not in time_str:
                            time_str = time_str.replace(" ", "T")
                        analysis_result["focus_time_start"] = datetime.fromisoformat(time_str)
                    except Exception:
                        pass  # Invalid format, will use context's existing time
                if "focus_time_end" in result:
                    try:
                        time_str = result["focus_time_end"].replace(" ", "T")
                        if "T" not in time_str:
                            time_str = time_str.replace(" ", "T")
                        analysis_result["focus_time_end"] = datetime.fromisoformat(time_str)
                    except Exception:
                        pass
            
            # Extract grep pattern if provided for grep strategy
            if analysis_result["next_strategy"] == "grep":
                if "grep_pattern" in result:
                    analysis_result["grep_pattern"] = result["grep_pattern"].strip()
                else:
                    # Fallback if grep strategy but no pattern provided
                    print("⚠️  Warning: Strategy is 'grep' but no grep_pattern provided. Falling back to 'find_high_severity'.")
                    analysis_result["next_strategy"] = "find_high_severity"
                    analysis_result.pop("grep_pattern", None)
            
            # Extract directory_path for explore_directory strategy
            if analysis_result["next_strategy"] == "explore_directory":
                if "directory_path" in result:
                    analysis_result["directory_path"] = result["directory_path"].strip()
            
            # Extract file_path and from_end for read_file_sample strategy
            if analysis_result["next_strategy"] == "read_file_sample":
                if "file_path" in result:
                    analysis_result["file_path"] = result["file_path"].strip()
                if "from_end" in result:
                    analysis_result["from_end"] = bool(result["from_end"])
            
            return analysis_result
        except json.JSONDecodeError as e:
            # Fallback if JSON parsing fails
            return {
                "hypothesis": response_text[:200] if 'response_text' in locals() else "Unable to parse LLM response",
                "confidence": 0.0,
                "next_strategy": "done",
                "reasoning": f"JSON parse error: {e}"
            }
        except Exception as e:
            return {
                "hypothesis": f"Error during LLM analysis: {e}",
                "confidence": 0.0,
                "next_strategy": "done",
                "reasoning": str(e)
            }
    
    def _should_continue(self, context: InvestigationContext) -> bool:
        """
        Check if investigation should continue.
        
        Args:
            context: Current investigation context
            
        Returns:
            True if should continue, False if should stop
        """
        # Stop if confidence threshold reached
        if context.confidence >= self.confidence_threshold:
            return False
        
        # Stop if max iterations reached
        if context.iteration >= context.max_iterations:
            return False
        
        # Continue otherwise
        return True
    
    def investigate(
        self,
        initial_question: str,
        api_key: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        log_file_paths: Optional[List[str]] = None,
    ) -> 'InvestigationResult':
        """
        Main investigation loop - iteratively queries and analyzes until root cause found.
        
        Args:
            initial_question: User's question about the issue
            api_key: Gemini API key (or uses GEMINI_API_KEY env var)
            start_time: Optional start time for investigation window
            end_time: Optional end time for investigation window
            log_file_paths: Optional list of log file paths for grep search capability
            
        Returns:
            InvestigationResult with final diagnosis
        """
        # Initialize context
        context = InvestigationContext(
            db_path=self.db_path,
            max_iterations=self.max_iterations,
            initial_question=initial_question,
            focus_time_start=start_time,
            focus_time_end=end_time,
            log_file_paths=log_file_paths or [],
        )
        
        # Initialize Gemini
        api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not set. Provide api_key or set GEMINI_API_KEY env var.")
        genai.configure(api_key=api_key)
        
        analysis = None
        
        # Main loop
        while self._should_continue(context):
            context.iteration += 1
            
            # Choose strategy based on iteration
            # First iteration: try high severity events (let LLM discover what's important)
            # Later iterations: use LLM to suggest strategy
            if context.iteration == 1:
                strategy = "find_high_severity"  # Start broad, let LLM discover what's relevant
            else:
                # Use previous LLM suggestion or default
                strategy = getattr(context, 'next_strategy', 'find_high_severity')
            
            # Check if LLM said "done" - if so, stop without generating another query
            if strategy == "done":
                print(f"✓ LLM indicated investigation complete. Stopping at iteration {context.iteration}.")
                break
            
            # Handle exploration strategies separately (not DB queries)
            events = []
            sql = None
            exploration_result = None
            is_exploration_strategy = strategy in ["grep", "explore_directory", "read_file_sample", "scan_metrics_anomalies", "temporal_analysis"]
            
            # Get LLM analysis from previous iteration if available
            previous_analysis = getattr(context, '_previous_analysis', None)
            
            if strategy == "explore_directory":
                directory_path = None
                if previous_analysis and "directory_path" in previous_analysis:
                    directory_path = previous_analysis["directory_path"]
                elif context.log_file_paths:
                    # Use parent directory of first log file
                    directory_path = str(Path(context.log_file_paths[0]).parent)
                elif context.exploration_data and 'last_directory_path' in context.exploration_data:
                    directory_path = context.exploration_data['last_directory_path']
                
                if directory_path:
                    print(f"\n📊 Iteration {context.iteration} - Strategy: {strategy}")
                    print(f"🔍 Exploring directory: {directory_path}")
                    exploration_result = self._explore_directory(directory_path)
                    context.exploration_data['directory_listing'] = exploration_result
                    context.exploration_data['last_directory_path'] = directory_path
                    # Extract file paths from directory listing
                    if exploration_result and 'files' in exploration_result:
                        new_files = [f['path'] for f in exploration_result.get('files', [])]
                        context.log_file_paths.extend([f for f in new_files if f not in context.log_file_paths])
                else:
                    print(f"⚠️  No directory path provided for exploration. Skipping.")
            
            elif strategy == "read_file_sample":
                file_path = previous_analysis.get("file_path") if previous_analysis else None
                if not file_path and context.log_file_paths:
                    file_path = context.log_file_paths[0]  # Use first log file
                
                from_end = previous_analysis.get("from_end", False) if previous_analysis else False
                
                if file_path:
                    print(f"\n📊 Iteration {context.iteration} - Strategy: {strategy}")
                    print(f"🔍 Reading sample from: {file_path} ({'end' if from_end else 'start'})")
                    exploration_result = self._read_file_sample(file_path, sample_size=2000, from_end=from_end)
                    context.exploration_data[f'sample_{Path(file_path).name}'] = exploration_result
                else:
                    print(f"⚠️  No file path provided for sampling. Skipping.")
            
            elif strategy == "scan_metrics_anomalies":
                # Scan previous query results for anomalies
                print(f"\n📊 Iteration {context.iteration} - Strategy: {strategy}")
                print(f"🔍 Scanning metrics for anomalies...")
                # Collect events from recent queries
                all_recent_events = []
                for qr in context.query_history[-5:]:  # Last 5 queries
                    all_recent_events.extend(qr.events)
                if events:  # Also scan current events if available
                    all_recent_events.extend(events)
                exploration_result = self._scan_metrics_anomalies(all_recent_events)
                context.metric_anomalies.extend(exploration_result)
            
            elif strategy == "temporal_analysis":
                # Check file ends for completion events
                print(f"\n📊 Iteration {context.iteration} - Strategy: {strategy}")
                print(f"🔍 Analyzing temporal progression...")
                temporal_results = []
                for file_path in context.log_file_paths[:5]:  # Check first 5 files
                    sample = self._read_file_sample(file_path, sample_size=500, from_end=True)
                    if 'lines' in sample and sample['lines']:
                        # Look for completion/end events
                        completion_patterns = ["QuietDatabaseEnd", "DatabaseEnd", "Shutdown", "Cleanup"]
                        for line in sample['lines'][-100:]:  # Last 100 lines
                            for pattern in completion_patterns:
                                if pattern in line:
                                    temporal_results.append({
                                        "file": file_path,
                                        "event": pattern,
                                        "found": True
                                    })
                                    break
                exploration_result = {"temporal_findings": temporal_results}
                context.exploration_data['temporal_analysis'] = exploration_result
            
            # Handle grep strategy separately (file search, not DB query)
            is_grep_strategy = strategy == "grep"
            
            if is_grep_strategy:
                # Get grep pattern from context or LLM analysis
                grep_pattern = getattr(context, 'grep_pattern', None)
                if not grep_pattern:
                    print(f"⚠️  No grep pattern provided for grep strategy. Falling back to 'find_high_severity'.")
                    strategy = "find_high_severity"
                    is_grep_strategy = False
                
                if grep_pattern and context.log_file_paths:
                    print(f"\n📊 Iteration {context.iteration} - Strategy: {strategy}")
                    print(f"🔍 Grep Pattern: {grep_pattern}")
                    events = self._grep_files(grep_pattern, context.log_file_paths, context)
                elif grep_pattern:
                    print(f"⚠️  No log files provided for grep search. Skipping grep strategy.")
                    strategy = "find_high_severity"
                    is_grep_strategy = False
            
            # Generate SQL query for non-exploration strategies (only DB query strategies)
            if not is_exploration_strategy:
                # Check if we have custom SQL from context (set by previous LLM analysis)
                custom_sql = None
                if strategy == "custom" and context.custom_sql:
                    custom_sql = context.custom_sql
                
                sql = self.query_generator.generate(strategy, context, custom_sql=custom_sql)
                
                # Print the query for debugging
                print(f"\n📊 Iteration {context.iteration} - Strategy: {strategy}")
                print(f"🔍 Query: {sql[:200]}..." if len(sql) > 200 else f"🔍 Query: {sql}")
            
            # Track this strategy and query
            context.tried_strategies.add(strategy)
            
            # If we've seen many of these events before, modify query to get different results
            # Check if this SQL query (or similar) has been executed before
            # Only do this for DB query strategies, not exploration strategies
            query_hashes = [hash(qr.query.strip()) for qr in context.query_history[-5:]]  # Check last 5 queries
            current_query_hash = hash(sql.strip()) if sql else None
            
            # Keep trying to get a different query until we succeed or exhaust options
            # Only for DB query strategies (skip for exploration strategies)
            max_retries = 3
            retry_count = 0
            while (not is_exploration_strategy and current_query_hash and 
                   current_query_hash in query_hashes and context.iteration > 2 and retry_count < max_retries):
                retry_count += 1
                context.consecutive_repeats += 1
                
                print(f"⚠️  Detected repeated query (attempt {retry_count}/{max_retries}). Trying to find different query...")
                
                # If we've detected repeats multiple times in a row, force a complete strategy change
                if context.consecutive_repeats >= 2 or retry_count >= 2:
                    print(f"⚠️  Multiple consecutive repeated queries detected ({context.consecutive_repeats}). Forcing complete strategy change...")
                    # Try a completely different strategy
                    all_strategies = ["find_high_severity", "find_connection_errors", "find_time_window", "find_recovery", "grep"]
                    # Filter out grep if no log files provided
                    if not context.log_file_paths:
                        all_strategies = [s for s in all_strategies if s != "grep"]
                    # Filter out strategies that require prerequisites
                    available_strategies = []
                    for s in all_strategies:
                        if s not in context.tried_strategies:
                            # Skip find_time_window if we don't have time bounds (unless we can set defaults)
                            if s == "find_time_window":
                                # Only include if we have time bounds OR can query the database for time range
                                if context.focus_time_start or context.focus_time_end:
                                    available_strategies.append(s)
                                else:
                                    # Try to get time range from database if we have seen events
                                    if context.query_history:
                                        # Use time from last query
                                        available_strategies.append(s)
                                    # Otherwise skip it
                            else:
                                available_strategies.append(s)
                    
                    if available_strategies:
                        strategy = available_strategies[0]
                        # If find_time_window but no time bounds, set defaults based on seen events
                        if strategy == "find_time_window" and not context.focus_time_start and not context.focus_time_end:
                            # Try to get time bounds from query history
                            if context.query_history:
                                # Get earliest and latest times from query history
                                all_times = []
                                for qr in context.query_history:
                                    if qr.events:
                                        for e in qr.events:
                                            if e.ts:
                                                all_times.append(e.ts)
                                if all_times:
                                    from datetime import timedelta
                                    earliest = min(all_times)
                                    latest = max(all_times)
                                    # Expand window by 5 minutes on each side
                                    context.focus_time_start = earliest - timedelta(minutes=5)
                                    context.focus_time_end = latest + timedelta(minutes=5)
                                else:
                                    # No times found, skip this strategy
                                    available_strategies.remove(strategy)
                                    if available_strategies:
                                        strategy = available_strategies[0]
                                    else:
                                        strategy = "find_high_severity"  # Fallback
                            else:
                                # No query history, skip time window
                                available_strategies.remove(strategy)
                                if available_strategies:
                                    strategy = available_strategies[0]
                                else:
                                    strategy = "find_high_severity"  # Fallback
                        
                        context.next_strategy = strategy
                        context.consecutive_repeats = 0  # Reset counter
                        # Generate new query with new strategy - clear custom_sql to ensure fresh query
                        context.custom_sql = None
                        # Update is_exploration_strategy flag for new strategy
                        is_exploration_strategy = strategy in ["grep", "explore_directory", "read_file_sample", "scan_metrics_anomalies", "temporal_analysis"]
                        if not is_exploration_strategy:
                            sql = self.query_generator.generate(strategy, context)
                            context.tried_strategies.add(strategy)
                            # Recalculate hash for new query
                            current_query_hash = hash(sql.strip()) if sql else None
                            print(f"🔄 Switched to strategy: {strategy}")
                            print(f"🔍 New Query: {sql[:200]}..." if len(sql) > 200 else f"🔍 New Query: {sql}")
                        else:
                            # Exploration strategy - break out of retry loop, will be handled in main flow
                            break
                    else:
                        # No available strategies - use fallback with exclusion
                        strategy = "custom"
                        context.next_strategy = strategy
                        context.consecutive_repeats = 0
                        # Create a query that excludes seen events
                        if context.seen_event_ids:
                            event_ids_str = "', '".join(str(eid) for eid in list(context.seen_event_ids)[:100])
                            context.custom_sql = (
                                "SELECT event_id, ts, severity, event, role, fields_json "
                                "FROM events "
                                f"WHERE event_id NOT IN ('{event_ids_str}') "
                                "AND (event LIKE '%Error%' OR event LIKE '%Fail%' OR event LIKE '%Timeout%' "
                                "OR event LIKE '%Connection%') "
                                "ORDER BY ts ASC LIMIT 100"
                            )
                        else:
                            context.custom_sql = (
                                "SELECT event_id, ts, severity, event, role, fields_json "
                                "FROM events "
                                "WHERE (event LIKE '%Error%' OR event LIKE '%Fail%' OR event LIKE '%Timeout%' "
                                "OR event LIKE '%Connection%') "
                                "ORDER BY ts ASC LIMIT 100"
                            )
                        sql = self.query_generator.generate(strategy, context, custom_sql=context.custom_sql)
                        context.tried_strategies.add(strategy)
                        current_query_hash = hash(sql.strip())
                        print(f"🔄 Using fallback custom query with exclusions")
                        print(f"🔍 New Query: {sql[:200]}..." if len(sql) > 200 else f"🔍 New Query: {sql}")
                else:
                    # First repeat - try to modify query
                    # Modify query to exclude seen events or expand search
                    if context.seen_event_ids and strategy == "custom":
                        # Add exclusion clause for seen events
                        event_ids_str = "', '".join(str(eid) for eid in list(context.seen_event_ids)[:100])  # Limit to 100
                        if "WHERE" in sql.upper():
                            sql = sql.replace("WHERE", f"WHERE event_id NOT IN ('{event_ids_str}') AND", 1)
                        else:
                            # Extract the column list from SELECT to preserve it
                            import re
                            select_match = re.match(r'\s*SELECT\s+(.+?)\s+FROM', sql, re.IGNORECASE | re.DOTALL)
                            if select_match:
                                columns = select_match.group(1).strip()
                                # Wrap query preserving columns
                                sql = sql.replace("SELECT", f"SELECT {columns} FROM (SELECT", 1).replace("LIMIT", f") WHERE event_id NOT IN ('{event_ids_str}') LIMIT", 1)
                            else:
                                # Fallback: use expected columns
                                sql = sql.replace("SELECT", f"SELECT event_id, ts, severity, event, role, fields_json FROM (SELECT", 1).replace("LIMIT", f") WHERE event_id NOT IN ('{event_ids_str}') LIMIT", 1)
                        current_query_hash = hash(sql.strip())
                        print(f"🔍 Modified Query (excluded seen events): {sql[:200]}..." if len(sql) > 200 else f"🔍 Modified Query: {sql}")
                    elif not context.focus_time_start or not context.focus_time_end:
                        # Expand time window - look at different time ranges
                        from datetime import timedelta
                        if context.focus_time_start:
                            # Look earlier
                            context.focus_time_start = context.focus_time_start - timedelta(minutes=5)
                            sql = self.query_generator.generate("find_time_window", context)
                            current_query_hash = hash(sql.strip())
                            print(f"🔍 Modified Query (expanded time window): {sql[:200]}..." if len(sql) > 200 else f"🔍 Modified Query: {sql}")
                        else:
                            # Try a completely different approach - look for connection errors
                            strategy = "find_connection_errors"
                            context.custom_sql = None
                            sql = self.query_generator.generate(strategy, context)
                            context.next_strategy = strategy
                            context.consecutive_repeats = 0  # Reset on strategy change
                            current_query_hash = hash(sql.strip())
                            print(f"🔄 Switched to strategy: {strategy}")
                            print(f"🔍 New Query: {sql[:200]}..." if len(sql) > 200 else f"🔍 New Query: {sql}")
            
            # If we exited the loop normally (not because of max retries), reset counter
            if retry_count == 0 or current_query_hash not in query_hashes:
                context.consecutive_repeats = 0
            
            # Execute query only if not exploration strategy (exploration strategies already executed above)
            if not is_exploration_strategy and sql:
                events = self._execute_query(sql, self.db_path)
            
            # Track seen event IDs
            new_event_ids = {e.event_id for e in events}
            previously_seen_count = len(new_event_ids & context.seen_event_ids)
            context.seen_event_ids.update(new_event_ids)
            
            # If we're seeing mostly the same events, force a different approach
            if events and previously_seen_count > len(events) * 0.8 and context.iteration > 2:
                print(f"⚠️  Getting mostly the same events ({previously_seen_count}/{len(events)} already seen). Trying different search...")
                # Try different strategies
                all_strategies = ["find_high_severity", "find_connection_errors", "find_time_window", "find_recovery", "grep"]
                # Filter out grep if no log files provided
                if not context.log_file_paths:
                    all_strategies = [s for s in all_strategies if s != "grep"]
                untried_strategies = [s for s in all_strategies if s not in context.tried_strategies]
                # Filter out find_time_window if we don't have time bounds
                available_strategies = [s for s in untried_strategies if s != "find_time_window" or context.focus_time_start or context.focus_time_end]
                if available_strategies:
                    strategy = available_strategies[0]
                    context.next_strategy = strategy
                else:
                    # All strategies tried - expand time window significantly or search without time filter
                    if context.focus_time_start and context.focus_time_end:
                        from datetime import timedelta
                        # Expand time window backward by 10 minutes
                        context.focus_time_start = context.focus_time_start - timedelta(minutes=10)
                        context.focus_time_end = context.focus_time_end + timedelta(minutes=5)
                        strategy = "find_time_window"
                        sql = self.query_generator.generate(strategy, context)
                    else:
                        # Remove time filter and search all events
                        context.focus_time_start = None
                        context.focus_time_end = None
                        strategy = "custom"
                        context.custom_sql = (
                            "SELECT event_id, ts, severity, event, role, fields_json "
                            "FROM events "
                            "WHERE (event LIKE '%Error%' OR event LIKE '%Fail%' OR event LIKE '%Timeout%' "
                            "OR event LIKE '%Connection%') "
                            "AND event_id NOT IN ('" + "', '".join(str(eid) for eid in list(context.seen_event_ids)[:100]) + "') "
                            "ORDER BY ts ASC LIMIT 100"
                        )
                        sql = self.query_generator.generate(strategy, context, custom_sql=context.custom_sql)
                    events = self._execute_query(sql, self.db_path)
                    context.seen_event_ids.update({e.event_id for e in events})
            
            # Store query result (for DB/grep strategies) or exploration data
            if is_exploration_strategy and exploration_result:
                # Store exploration result in context instead of query result
                print(f"📋 Exploration result: {json.dumps(exploration_result, indent=2, default=str)[:500]}...")
                # For scan_metrics_anomalies, we can still create events from anomalies
                if strategy == "scan_metrics_anomalies" and exploration_result:
                    # Convert anomalies to summary for LLM
                    context.exploration_data['latest_anomalies'] = exploration_result
                    # Create a dummy query result to track this iteration
                    query_result = QueryResult(
                        query=f"scan_metrics: {len(exploration_result)} anomalies found",
                        strategy=strategy,
                        events=events,  # Events may be empty for pure exploration
                        event_count=len(exploration_result),
                        timestamp=datetime.now()
                    )
                    context.query_history.append(query_result)
            else:
                # Regular query result
                query_str = sql or (f"grep: {context.grep_pattern}" if is_grep_strategy and context.grep_pattern else "N/A")
                query_result = QueryResult(
                    query=query_str,
                    strategy=strategy,
                    events=events,
                    event_count=len(events),
                    timestamp=datetime.now()
                )
                context.query_history.append(query_result)
            
            # Format for LLM (include exploration data for non-exploration strategies)
            events_text = self._format_events_for_llm(events, context)
            
            # For exploration strategies, format exploration result instead
            if is_exploration_strategy and exploration_result and strategy != "grep":
                # Format exploration result as text for LLM
                exploration_text = json.dumps(exploration_result, indent=2, default=str)[:2000]  # Limit length
                events_text = f"Exploration result:\n{exploration_text}"
            
            # Analyze with LLM
            analysis = self._analyze_with_llm(events_text, context, api_key)
            
            # Store LLM analysis in context for next iteration's exploration strategies
            if analysis:
                context._previous_analysis = analysis
            
            # Update context
            context.hypothesis = analysis["hypothesis"]
            context.confidence = analysis["confidence"]
            context.next_strategy = analysis.get("next_strategy", "done")
            
            # Store custom SQL from LLM if provided (check this first before auto-overrides)
            llm_provided_custom_sql = analysis.get("custom_sql") if analysis else None
            
            # Store grep pattern from LLM if provided
            if analysis and "grep_pattern" in analysis:
                context.grep_pattern = analysis["grep_pattern"]
            
            # Auto-detect if we're only seeing routine/informational events OR just Severity 20 warnings
            # If all events are routine (Knob, Configuration, etc.) OR just Severity 20 warnings, prioritize metric analysis
            routine_event_patterns = ["knob", "configuration", "setup", "initialize", "startup"]
            error_event_patterns = ["error", "fail", "timeout", "exception", "disconnect", "closed", "corruption"]
            
            if events:
                # Check for Severity 40+ errors first
                severity_40_plus = [e for e in events if (e.severity or 0) >= 40]
                severity_20_only = [e for e in events if (e.severity or 0) == 20]
                
                # Check if we only have Severity 20 warnings (like FileNotFoundError) - these are NOT root causes
                only_warnings = len(severity_40_plus) == 0 and len(severity_20_only) > 0
                
                # Check if all events are routine (no error indicators)
                all_routine = all(
                    any(pattern in str(e.event).lower() for pattern in routine_event_patterns)
                    and not any(pattern in str(e.event).lower() for pattern in error_event_patterns)
                    for e in events[:10]  # Check first 10 events
                )
                
                # Check fields_json for error keywords
                has_errors_in_fields = False
                for e in events[:10]:
                    if e.fields_json and isinstance(e.fields_json, dict):
                        fields_str = str(e.fields_json).lower()
                        if any(keyword in fields_str for keyword in error_event_patterns + ["errorcode", "error code"]):
                            has_errors_in_fields = True
                            break
                
                # If only Severity 20 warnings (like FileNotFoundError), prioritize metric analysis instead
                if only_warnings:
                    print(f"⚠️  Only Severity 20 warnings found (likely non-fatal startup warnings like FileNotFoundError). Prioritizing metric analysis over error search...")
                    context.confidence = min(context.confidence, 0.4)
                    # Suggest scan_metrics_anomalies if available
                    if "scan_metrics_anomalies" not in context.tried_strategies and context.query_history:
                        context.next_strategy = "scan_metrics_anomalies"
                
                # If only routine events and no errors found, force keyword search
                # But check if LLM already provided a keyword search query - if so, use that
                elif all_routine and not has_errors_in_fields:
                    print(f"⚠️  Only routine/informational events found (Knob, Configuration, etc.). Forcing keyword search for actual failures...")
                    context.confidence = min(context.confidence, 0.3)
                    # Force custom keyword search
                    context.next_strategy = "custom"
                    # If LLM already provided a custom SQL that searches for errors, use it; otherwise use our default
                    if llm_provided_custom_sql and any(kw in llm_provided_custom_sql.lower() for kw in error_event_patterns):
                        # Normalize LLM-provided SQL to use correct columns
                        normalized_sql = llm_provided_custom_sql.replace("SELECT *", "SELECT event_id, ts, severity, event, role, fields_json")
                        context.custom_sql = normalized_sql
                    else:
                        context.custom_sql = (
                            "SELECT event_id, ts, severity, event, role, fields_json "
                            "FROM events "
                            "WHERE (event LIKE '%Error%' OR event LIKE '%Fail%' OR event LIKE '%Timeout%' "
                            "OR event LIKE '%Connection%' OR event LIKE '%Exception%' "
                            "OR fields_json::string LIKE '%error%' OR fields_json::string LIKE '%fail%' "
                            "OR fields_json::string LIKE '%timeout%' OR fields_json::string LIKE '%Exception%') "
                            "ORDER BY ts ASC LIMIT 100"
                        )
            
            # Auto-detect if hypothesis is too surface-level (just describing symptoms)
            # Even if LLM says high confidence, lower it if hypothesis doesn't identify actual failures
            hypothesis_lower = (context.hypothesis or "").lower()
            symptom_patterns = ["re-enable", "re-enablement", "reenable", "recovery state", "configuration", "knob", "routine"]
            failure_keywords = ["error", "fail", "timeout", "disconnect", "corruption", "race condition",
                               "bug", "issue", "exception", "error code", "failed"]
            
            # If hypothesis only describes symptoms/routine events without identifying failures, lower confidence
            is_symptom_only = any(pattern in hypothesis_lower for pattern in symptom_patterns) and \
                             not any(keyword in hypothesis_lower for keyword in failure_keywords)
            
            # If hypothesis mentions only routine events (Knob, configuration) or says "cannot be determined"
            is_routine_only = ("knob" in hypothesis_lower or "configuration" in hypothesis_lower or 
                             "routine" in hypothesis_lower or "cannot be determined" in hypothesis_lower) and \
                             not any(keyword in hypothesis_lower for keyword in failure_keywords)
            
            if (is_symptom_only or is_routine_only) and context.confidence > 0.6:
                print(f"⚠️  Auto-adjusting: Hypothesis only describes routine events, forcing keyword search for actual failures...")
                context.confidence = min(context.confidence, 0.3)
                # Force investigation of actual failures with keyword search
                if context.next_strategy != "custom":
                    context.next_strategy = "custom"
                    # If LLM already provided a custom SQL that searches for errors, use it; otherwise use our default
                    if llm_provided_custom_sql and any(kw in llm_provided_custom_sql.lower() for kw in error_event_patterns):
                        context.custom_sql = llm_provided_custom_sql
                    else:
                        context.custom_sql = (
                            "SELECT event_id, ts, severity, event, role, fields_json "
                            "FROM events "
                            "WHERE (event LIKE '%Error%' OR event LIKE '%Fail%' OR event LIKE '%Timeout%' "
                            "OR event LIKE '%Connection%' OR event LIKE '%Exception%' "
                            "OR fields_json::string LIKE '%error%' OR fields_json::string LIKE '%fail%' "
                            "OR fields_json::string LIKE '%timeout%' OR fields_json::string LIKE '%Exception%') "
                            "ORDER BY ts ASC LIMIT 100"
                        )
            elif is_symptom_only and context.confidence <= 0.6:
                # Even with low confidence, if strategy isn't searching for errors, force it
                if context.next_strategy not in ["find_connection_errors", "find_time_window", "custom"]:
                    context.next_strategy = "find_connection_errors"
            
            # Store custom SQL from LLM if provided and not already overridden (for next iteration)
            if not context.custom_sql and "custom_sql" in analysis:
                context.custom_sql = analysis["custom_sql"]
            elif context.next_strategy != "custom" and not llm_provided_custom_sql:
                # Clear custom_sql if strategy changed away from custom and LLM didn't provide one
                context.custom_sql = None
            
            # Optional: Auto-detect recovery events and suggest looking before them (if LLM hasn't already)
            # Only do this on first few iterations and if LLM hasn't provided a time window
            # Not all cases have recovery issues, so don't force this
            if (context.iteration <= 2 and 
                events and 
                not context.focus_time_start and 
                not context.focus_time_end and
                any("recovery" in str(e.event).lower() or "recoverystate" in str(e.event).lower() for e in events)):
                # Find earliest recovery timestamp
                recovery_times = [e.ts for e in events if e.ts and ("recovery" in str(e.event).lower() or "recoverystate" in str(e.event).lower())]
                if recovery_times:
                    earliest_recovery = min(recovery_times)
                    from datetime import timedelta
                    # Just suggest it, don't force it - let LLM decide if recovery is relevant
                    print(f"💡 Found recovery at {earliest_recovery}. LLM can investigate events before this if relevant.")
            
            # Update time window from LLM suggestion if provided
            if "focus_time_start" in analysis:
                context.focus_time_start = analysis["focus_time_start"]
            if "focus_time_end" in analysis:
                context.focus_time_end = analysis["focus_time_end"]
            
            # Check if LLM says we're done
            # Only stop if:
            # 1. LLM says "done" AND
            # 2. Confidence is high (>= threshold) AND
            # 3. Hypothesis actually identifies a FAILURE/BUG (not just event names or symptoms)
            should_stop = False
            if context.next_strategy == "done" and context.confidence >= self.confidence_threshold:
                hypothesis_lower = context.hypothesis.lower() if context.hypothesis else ""
                
                # Check if hypothesis is too surface-level (just describing events, not identifying failures)
                # Event names that are symptoms/responses (not root causes)
                symptom_event_patterns = ["re-enable", "re-enablement", "reenable", "recovery state", 
                                         "recoverystate", "configuration", "initialization", "setup"]
                # Keywords that suggest actual failures/bugs
                failure_keywords = ["error", "fail", "timeout", "disconnect", "corruption", "race condition",
                                   "bug", "issue", "problem", "exception", "error code", "failed"]
                
                # Check if hypothesis just repeats event names without identifying failures
                is_symptom_only = any(pattern in hypothesis_lower for pattern in symptom_event_patterns) and \
                                 not any(keyword in hypothesis_lower for keyword in failure_keywords)
                
                # Check if hypothesis actually identifies a BUG, not just test failures or expected behavior
                # Keywords that suggest it's just describing test scenario (not a bug)
                test_scenario_keywords = ["intentional", "expected", "supposed", "test failure", "injected failure", 
                                         "simulation", "test run", "by design"]
                # Keywords that suggest it's just dismissing events without analyzing
                dismissive_keywords = ["expected", "informational", "low-severity", "normal", "standard", 
                                      "need to look", "need to search", "need to find", "we need to"]
                
                # Check if hypothesis actually identifies what's being tested vs just saying "need to search more"
                is_test_scenario_only = any(keyword in hypothesis_lower for keyword in test_scenario_keywords)
                is_dismissive = any(keyword in hypothesis_lower for keyword in dismissive_keywords) and context.confidence < 0.8
                
                # If hypothesis is too surface-level (just symptoms), dismissive, or just says "need to search", continue
                if is_symptom_only or is_test_scenario_only or is_dismissive:
                    print(f"⚠️  Hypothesis too surface-level - only describing symptoms, not finding actual failures. Continuing investigation...")
                    # Force it to look for actual failures
                    context.next_strategy = "find_connection_errors"
                    context.confidence = min(context.confidence, 0.5)  # Lower confidence if only symptoms found
                    # Check if we've been stuck in the same time window - if so, expand search
                    if context.iteration > 3 and context.focus_time_start and context.focus_time_end:
                        from datetime import timedelta
                        # Expand time window or look elsewhere
                        if context.iteration % 3 == 0:  # Every 3 failed iterations, expand window
                            context.focus_time_start = context.focus_time_start - timedelta(minutes=5)
                            print(f"⚠️  Expanding search window - looking further back in time")
                            context.next_strategy = "find_time_window"
                        else:
                            # Try different strategy
                            context.next_strategy = "find_high_severity"
                        context.confidence = min(context.confidence, 0.7)
                    else:
                        if is_dismissive:
                            print("⚠️  Hypothesis dismisses events. Analyzing what events indicate about test scenario...")
                        else:
                            print("⚠️  Hypothesis doesn't identify what's being tested. Continuing investigation...")
                        context.next_strategy = "find_high_severity"
                        context.confidence = min(context.confidence, 0.7)
                    should_stop = False
                else:
                    # Trust LLM's confidence level - if high confidence, it identified what's being tested
                    should_stop = True
            
            if should_stop:
                break
        
        # Build result
        result = InvestigationResult(
            hypothesis=context.hypothesis or "No hypothesis generated",
            confidence=context.confidence,
            iterations=context.iteration,
            query_count=len(context.query_history),
            evidence_events=context.query_history[-1].events[:10] if context.query_history else [],
            reasoning=analysis.get("reasoning", "") if analysis else "",
        )
        
        return result


@dataclass
class InvestigationResult:
    """Result object containing diagnosis, confidence, evidence."""
    hypothesis: str
    confidence: float
    iterations: int
    query_count: int
    evidence_events: List[EventModel]
    reasoning: str = ""