from typing import Optional, Dict, Any
from pydantic import BaseModel
import json
from datetime import datetime
import xml.etree.ElementTree as ET


#pydantic model for easy data transfer
class EventModel(BaseModel):
    event_id: int
    ts: Optional[datetime]
    severity: Optional[int]
    event: Optional[str]
    process: Optional[str]
    role: Optional[str]
    pid: Optional[str]
    machine_id: Optional[str]
    address: Optional[str]
    trace_file: Optional[str]
    src_line: Optional[int]
    raw_json: Dict[str, Any]
    fields_json: Dict[str, Any]

MANDATORY_FIELDS = {
    "Severity", "Time", "DateTime", "Type", "Process", "Role",
    "PID", "Machine", "MachineId", "Address", "LogGroup", "File", "Line"
}

def parse_logs(path: str):
    if path.endswith(".xml"):
        # stream XML events
        for i, (_, elem) in enumerate(ET.iterparse(path, events=("end",)), start=1):
            if elem.tag != "Event":
                continue
            obj = dict(elem.attrib)
            yield EventModel(
                event_id=i,
                ts=obj.get("DateTime"),
                severity=int(obj["Severity"]) if "Severity" in obj else None,
                event=obj.get("Type"),
                process=obj.get("Process"),
                role=obj.get("Role"),
                pid=int(obj["PID"]) if "PID" in obj else None,
                machine_id=obj.get("Machine") or obj.get("MachineId"),
                address=obj.get("Address"),
                trace_file=obj.get("File"),
                src_line=int(obj["Line"]) if "Line" in obj else None,
                raw_json=obj,
                fields_json={k: v for k, v in obj.items() if k not in MANDATORY_FIELDS},
            )
            elem.clear()
    else:
        # keep your current JSON line logic
        with open(path) as f:
            for i, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                obj = json.loads(line)
                ...
                yield EventModel(...)
    

#for testing purposes
if __name__ == "__main__":
    sample_path = "/Users/leo/Desktop/Academics/Fall25/Cloud Computing/trace.127.0.0.1.32304.1758756063.r7ZsGg.0.1.xml"
    it = parse_logs(sample_path)
    first_three = [next(it).dict() for _ in range(3)]
    for row in first_three:
        print(row)
    
    

    
