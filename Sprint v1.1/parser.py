import json
import re
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import Optional, Dict, Any
from pydantic import BaseModel

mandatory_fields = {
    "Severity", "Time", "DateTime", "Type", "Process", "Role",
    "PID", "Machine", "MachineId", "Address", "LogGroup",
    "File", "Line"
}

class EventModel(BaseModel):
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

def parse_plaintext(line: str) -> dict:
    matches = re.findall(r'(\w+)=([^\s]+)', line)
    return {k: v for k, v in matches}

def parse_logs(path: str):
    if path.endswith(".xml"):
        for i, (_, elem) in enumerate(ET.iterparse(path, events=("end",)), start=1):
            if elem.tag != "Event":
                continue
            obj = dict(elem.attrib)

            ts = None
            if "DateTime" in obj:
                try:
                    ts = datetime.strptime(obj["DateTime"], "%Y-%m-%dT%H:%M:%SZ")
                except Exception:
                    pass

            yield EventModel(
                event_id=i,
                ts=ts,
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
                fields_json={k: v for k, v in obj.items() if k not in mandatory_fields}
            )
            elem.clear()
    else:
        with open(path, 'r') as file:
            for id, line in enumerate(file, start=1):
                line = line.strip()
                if not line:
                    continue

                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    obj = parse_plaintext(line)

                ts = None
                if "DateTime" in obj:
                    try:
                        ts = datetime.strptime(obj["DateTime"], "%Y-%m-%dT%H:%M:%SZ")
                    except Exception:
                        pass

                yield EventModel(
                    event_id=id,
                    ts=ts,
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
                    fields_json={k: v for k, v in obj.items() if k not in mandatory_fields}
                )

#for testing purposes
if __name__ == "__main__":
    sample_path = "/Users/leo/Desktop/Academics/Fall25/Cloud Computing/trace.127.0.0.1.32304.1758756063.r7ZsGg.0.1.xml"
    it = parse_logs(sample_path)
    first_three = [next(it).dict() for _ in range(3)]
    for row in first_three:
        print(row)
    
    

    
