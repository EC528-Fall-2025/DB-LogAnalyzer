"""
Log parsing service for FDB trace logs.
Supports both JSON and XML formats.
"""
import json
import re
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import Generator, Dict, Any
from dto.event import EventModel, MANDATORY_FIELDS


class LogParser:
    """Log parsing service"""
    
    @staticmethod
    def parse_plaintext(line: str) -> dict:
        """Parse plaintext format log line"""
        matches = re.findall(r'(\w+)=([^\s]+)', line)
        return {k: v for k, v in matches}
    
    @staticmethod
    def parse_datetime(date_str: str) -> datetime:
        """Parse datetime string"""
        try:
            return datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%SZ")
        except Exception:
            return None
    
    def parse_xml_logs(self, path: str) -> Generator[EventModel, None, None]:
        """Parse XML format log file"""
        for i, (_, elem) in enumerate(ET.iterparse(path, events=("end",)), start=1):
            if elem.tag != "Event":
                continue
            
            obj = dict(elem.attrib)
            ts = self.parse_datetime(obj.get("DateTime")) if "DateTime" in obj else None
            
            yield self._create_event_model(i, obj, ts)
            elem.clear()  # Clear memory
    
    def parse_json_logs(self, path: str) -> Generator[EventModel, None, None]:
        """Parse JSON/plaintext format log file"""
        with open(path, 'r') as file:
            for id, line in enumerate(file, start=1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    obj = self.parse_plaintext(line)
                
                ts = self.parse_datetime(obj.get("DateTime")) if "DateTime" in obj else None
                yield self._create_event_model(id, obj, ts)
    
    def parse_logs(self, path: str) -> Generator[EventModel, None, None]:
        """
        Unified interface for parsing log files
        
        Args:
            path: Log file path
            
        Yields:
            EventModel: Parsed event model
        """
        if path.endswith(".xml"):
            yield from self.parse_xml_logs(path)
        else:
            yield from self.parse_json_logs(path)
    
    def _create_event_model(self, event_id: int, obj: Dict[str, Any], ts: datetime) -> EventModel:
        """Create EventModel from raw data"""
        return EventModel(
            event_id=event_id,
            ts=ts,
            severity=int(obj["Severity"]) if "Severity" in obj else None,
            event=obj.get("Type"),
            process=obj.get("Processes"),
            role=obj.get("Roles"),
            pid=int(obj["PID"]) if "PID" in obj else None,
            machine_id=obj.get("Machine") or obj.get("MachineId"),
            address=obj.get("Address"),
            trace_file=obj.get("File"),
            src_line=int(obj["Line"]) if "Line" in obj else None,
            raw_json=obj,
            fields_json={k: v for k, v in obj.items() if k not in MANDATORY_FIELDS}
        )
