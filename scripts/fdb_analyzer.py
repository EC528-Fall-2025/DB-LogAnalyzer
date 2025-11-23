
import sys
import xml.etree.ElementTree as ET
from typing import List, Dict, Any, Optional


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

CRITICAL_SEVERITY = 20  # treat >= 20 as interesting


def parse_events(path: str):
    """
    Parse an FDB XML trace file into a flat list of <Event> elements.
    Tries both standard <Trace> root and raw <Event> stream (wrapped).
    """
    with open(path, "r", errors="ignore") as f:
        text = f.read()

    # First try standard <Trace> root
    try:
        root = ET.fromstring(text)
        # If this is a normal FDB trace, root.tag should be "Trace"
        if root.tag.lower() not in ("trace", "root"):
            # Unexpected root, but we'll still treat children as events
            pass
        return list(root)
    except Exception:
        # Maybe the file is a concatenation or a chunk of <Event> lines.
        wrapped = "<Root>" + text + "</Root>"
        root = ET.fromstring(wrapped)
        return list(root)


def is_master_fully_recovered(ev: ET.Element) -> bool:
    """Return True if this event is a MasterRecoveryState that indicates fully recovered."""
    ev_type = ev.attrib.get("Type") or ev.attrib.get("type")
    if ev_type != "MasterRecoveryState":
        return False

    status = ev.attrib.get("Status") or ev.attrib.get("status")
    status_code = ev.attrib.get("StatusCode") or ev.attrib.get("statuscode")

    # Robust checks: either textual or numeric
    if status == "fully_recovered":
        return True
    if status_code == "14":
        return True
    if status == "14":  # sometimes Status stores the numeric code
        return True
    return False


def is_recovery_start(ev: ET.Element) -> bool:
    """
    Heuristic: decide if this event is a good start marker for an anomaly / recovery window.
    """
    a = ev.attrib
    ev_type = a.get("Type") or a.get("type")
    sev = int(a.get("Severity", "0"))

    # 1) explicit role / recovery triggers
    if ev_type in START_TRIGGER_TYPES:
        return True

    # 2) MasterRecoveryState in a non-fully-recovered state
    if ev_type == "MasterRecoveryState" and not is_master_fully_recovered(ev):
        return True

    # 3) Any high-severity error event
    if sev >= CRITICAL_SEVERITY and ("Error" in a or "error" in a):
        return True

    return False


def event_time(ev: ET.Element) -> float:
    try:
        return float(ev.attrib.get("Time", "0"))
    except ValueError:
        return 0.0


def summarize_event(ev: ET.Element) -> Dict[str, Any]:
    a = ev.attrib
    return {
        "Type": a.get("Type") or a.get("type"),
        "Time": a.get("Time"),
        "DateTime": a.get("DateTime") or a.get("datetime"),
        "Severity": int(a.get("Severity", "0")),
        "Error": a.get("Error") or a.get("error"),
        "ErrorDescription": a.get("ErrorDescription") or a.get("errordescription"),
        "ErrorCode": a.get("ErrorCode") or a.get("errorcode"),
        "Status": a.get("Status") or a.get("status"),
        "StatusCode": a.get("StatusCode") or a.get("statuscode"),
        "Machine": a.get("Machine") or a.get("machine"),
        "ID": a.get("ID") or a.get("id"),
        "RawAttrs": dict(a),
    }


def analyze_anomalies(events: List[ET.Element]) -> List[Dict[str, Any]]:
    """
    Analyze the event stream and return a list of "anomaly windows".
    Each window is a dict with:
        - start_index, end_index
        - start_event (summary)
        - end_event (summary)
        - duration
        - critical_events: list of high-severity / error events inside
    """
    anomalies: List[Dict[str, Any]] = []
    current_start: Optional[int] = None

    for i, ev in enumerate(events):
        # Check for end condition first (Master fully recovered)
        if current_start is not None and is_master_fully_recovered(ev):
            start_idx = current_start
            end_idx = i
            window_events = events[start_idx:end_idx + 1]

            start_ev = window_events[0]
            end_ev = ev

            # Collect critical events within the window
            crits = []
            for e in window_events:
                a = e.attrib
                sev = int(a.get("Severity", "0"))
                if sev >= CRITICAL_SEVERITY or ("Error" in a or "error" in a):
                    crits.append(summarize_event(e))

            anomaly = {
                "start_index": start_idx,
                "end_index": end_idx,
                "start_event": summarize_event(start_ev),
                "end_event": summarize_event(end_ev),
                "duration": event_time(end_ev) - event_time(start_ev),
                "critical_events": crits,
            }
            anomalies.append(anomaly)
            current_start = None  # close window and wait for next anomaly
            continue

        # If not currently inside a window, look for a start trigger
        if current_start is None and is_recovery_start(ev):
            current_start = i

    return anomalies


def print_report(anomalies: List[Dict[str, Any]], file_name: str):
    if not anomalies:
        print(f"[{file_name}] No anomaly windows detected (no recovery segments ending at Status=14).")
        return

    print(f"=== Anomaly / Recovery Windows for {file_name} ===")
    for idx, an in enumerate(anomalies, 1):
        s = an["start_event"]
        e = an["end_event"]
        print(f"\n--- Window #{idx} ---")
        print(f"  Start: idx={an['start_index']}  Time={s['Time']}  Type={s['Type']}  Severity={s['Severity']}")
        if s["Error"]:
            print(f"         Error={s['Error']}  Desc={s['ErrorDescription']}")
        if s["Status"] or s["StatusCode"]:
            print(f"         Status={s['Status']}  StatusCode={s['StatusCode']}")

        print(f"  End:   idx={an['end_index']}  Time={e['Time']}  Type={e['Type']}")
        print(f"         Status={e['Status']}  StatusCode={e['StatusCode']}")
        print(f"  Duration: {an['duration']:.6f} seconds")
        print(f"  Critical events inside: {len(an['critical_events'])}")
        for ce in an["critical_events"][:10]:  # show up to 10 per window
            line = f"    - t={ce['Time']}  Sev={ce['Severity']}  Type={ce['Type']}"
            if ce["Error"]:
                line += f"  Error={ce['Error']}"
            if ce["Status"]:
                line += f"  Status={ce['Status']}"
            print(line)
        if len(an["critical_events"]) > 10:
            print(f"    ... (+{len(an['critical_events'])-10} more)")


def main():
    if len(sys.argv) < 2:
        print("Usage: python fdb_analyzer.py <trace.xml> [more_traces.xml ...]")
        sys.exit(1)

    for path in sys.argv[1:]:
        try:
            events = parse_events(path)
        except Exception as e:
            print(f"[{path}] Failed to parse XML: {e}")
            continue

        anomalies = analyze_anomalies(events)
        print_report(anomalies, path)


if __name__ == "__main__":
    main()
