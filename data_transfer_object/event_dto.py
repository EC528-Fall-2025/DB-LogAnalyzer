"""
Event data transfer objects (DTOs) for FDB log analysis.
"""
from datetime import datetime
from typing import Optional, Dict, Any
from pydantic import BaseModel


# Mandatory fields in logs
MANDATORY_FIELDS = {
    "Severity", "Time", "DateTime", "Type", "Process", "Role",
    "PID", "Machine", "MachineId", "Address", "LogGroup",
    "File", "Line"
}


class EventModel(BaseModel):
    """Event data model, representing a single log event"""
    event_id: int
    ts: Optional[datetime]
    severity: Optional[int]
    event: Optional[str]
    process: Optional[str]
    role: Optional[str]
    pid: Optional[int]
    machine_id: Optional[str]
    address: Optional[str]
    trace_file: Optional[str]
    src_line: Optional[int]
    raw_json: Dict[str, Any]
    fields_json: Dict[str, Any]
