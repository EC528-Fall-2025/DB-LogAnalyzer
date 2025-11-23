import sys
import json
import xml.etree.ElementTree as ET
from typing import List, Dict, Any

CRITICAL_SEVERITY = 20  # threshold


def parse_events(path: str) -> List[ET.Element]:
    """Parse XML -> list of <Event>."""
    with open(path, "r", errors="ignore") as f:
        text = f.read()

    try:
        root = ET.fromstring(text)
        return list(root)
    except Exception:
        wrapped = "<Root>" + text + "</Root>"
        root = ET.fromstring(wrapped)
        return list(root)


def extract_critical(ev: ET.Element) -> Dict[str, Any]:
    """Convert an event into structured JSON."""
    a = ev.attrib
    return {
        "Time": a.get("Time"),
        "DateTime": a.get("DateTime") or a.get("datetime"),
        "Type": a.get("Type") or a.get("type"),
        "Severity": int(a.get("Severity", "0")),
        "Error": a.get("Error") or a.get("error"),
        "ErrorDescription": a.get("ErrorDescription") or a.get("errordescription"),
        "ErrorCode": a.get("ErrorCode") or a.get("errorcode"),
        "Status": a.get("Status") or a.get("status"),
        "StatusCode": a.get("StatusCode") or a.get("statuscode"),
        "Machine": a.get("Machine") or a.get("machine"),
        "Role": a.get("Role") or a.get("role"),
        "Address": a.get("Address") or a.get("address"),
        "RawAttrs": dict(a),
    }


def is_critical(ev: ET.Element) -> bool:
    """Define what counts as critical."""
    a = ev.attrib
    sev = int(a.get("Severity", "0"))

    # severity threshold
    if sev >= CRITICAL_SEVERITY:
        return True

    # explicit error signals
    for k in a.keys():
        if "error" in k.lower():
            return True

    return False


def process_file(path: str):
    events = parse_events(path)

    crit = []
    for ev in events:
        if is_critical(ev):
            crit.append(extract_critical(ev))

    out_path = path + ".critical.json"

    with open(out_path, "w") as f:
        json.dump(crit, f, indent=2)

    print(f"Extracted {len(crit)} critical events → {out_path}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python critical_extractor.py <trace.xml> [trace2.xml...]")
        sys.exit(1)

    for file in sys.argv[1:]:
        try:
            process_file(file)
        except Exception as e:
            print(f"Failed {file}: {e}")


if __name__ == "__main__":
    main()