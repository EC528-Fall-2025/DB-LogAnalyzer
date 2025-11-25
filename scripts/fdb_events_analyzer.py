import sys
import json
import xml.etree.ElementTree as ET
from typing import List, Dict, Any

CRITICAL_SEVERITY = 20

START_TRIGGER_TYPES = {
    # Role termination
    "MasterTerminated",
    "TLogTerminated",
    "CommitProxyTerminated",
    "ResolverTerminated",
    "DataDistributionTerminated",
    "RatekeeperTerminated",

    # Explicit recovery start
    "ClusterRecoveryRetrying",
    "ClusterRecoveryInitiated",
    "ClusterRecoveryFromChildren",

    # Bootstrap failures & recruitment
    "RecruitTLogFailed",
    "RecruitStorageNotAvailable",
    "RecruitStorageServerFailed",
    "RecruitRatekeeperFailed",
    "NoMoreServers",

    # Storage engine & disk errors
    "FileOpenError",
    "KeyValueStoreError",
    "DiskError",
    "SQLiteError",
    "IOError",
    "ChecksumMismatch",

    # Coordinator / lease / epoch
    "CoordinatorChanged",
    "MasterLeaseExpired",
    "MasterLeaseLost",
    "EpochBegin",
    "EpochEnd",
    "RecruitMaster",

    # Performance-related triggers
    "RatekeeperSaturated",
    "TagThrottled",
    "QueueOverflow",
    "ProxyMemoryThresholdExceeded",
    "LargeTransactionModeSwitch",
    "ResolverTooSlow",
}


def parse_xml(path: str) -> List[ET.Element]:
    """Parse XML trace file -> list of <Event>."""
    with open(path, "r", errors="ignore") as f:
        text = f.read()

    # try normal document first
    try:
        root = ET.fromstring(text)
        return list(root)
    except Exception:
        wrapped = "<Root>" + text + "</Root>"
        root = ET.fromstring(wrapped)
        return list(root)


def extract_attributes(ev: ET.Element) -> Dict[str, Any]:
    """Flatten and normalize attributes into JSON."""
    a = ev.attrib

    return {
        "Type": a.get("Type") or a.get("type"),
        "Time": a.get("Time"),
        "DateTime": a.get("DateTime") or a.get("datetime"),

        "Severity": int(a.get("Severity", "0")),

        # Core recovery context
        "Machine": a.get("Machine") or a.get("machine"),
        "Role": a.get("Role") or a.get("role"),
        "Address": a.get("Address") or a.get("address"),
        "PID": a.get("PID") or a.get("pid"),
        "ID": a.get("ID") or a.get("id"),

        # Failure + recovery flags
        "Error": a.get("Error") or a.get("error"),
        "ErrorDescription": a.get("ErrorDescription") or a.get("errordescription"),
        "ErrorCode": a.get("ErrorCode") or a.get("errorcode"),
        "Status": a.get("Status") or a.get("status"),
        "StatusCode": a.get("StatusCode") or a.get("statuscode"),

        # Source context (important for internal debugging)
        "File": a.get("File") or a.get("file"),
        "Line": a.get("Line") or a.get("line"),
        "SourceVersion": a.get("SourceVersion"),

        # Keep full original attributes
        "RawAttrs": dict(a),
    }


def is_important(ev: ET.Element) -> bool:
    """Select events that matter for anomaly/recovery classification."""
    a = ev.attrib
    t = a.get("Type") or a.get("type")
    sev = int(a.get("Severity", "0"))

    # 1) High-severity errors
    if sev >= CRITICAL_SEVERITY:
        return True

    # 2) Explicit recovery triggers
    if t in START_TRIGGER_TYPES:
        return True

    # 3) Any ERROR keyword
    for k in a.keys():
        if "error" in k.lower():
            return True

    # 4) CodeCoverage events with semantic content
    if t == "CodeCoverage":
        comment = a.get("Comment") or a.get("comment") or ""
        covered = a.get("Covered", "")
        if covered == "1" and comment.strip() != "":
            return True

    return False


def process(path: str):
    events = parse_xml(path)

    important = []
    for ev in events:
        if is_important(ev):
            important.append(extract_attributes(ev))

    out_path = path + ".important.json"
    with open(out_path, "w") as f:
        json.dump(important, f, indent=2)

    print(f"✔ Extracted {len(important)} important logs → {out_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python fdb_important_events.py <trace.xml>")
        sys.exit(1)

    process(sys.argv[1])