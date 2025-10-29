import re
import json
import glob
# --------- REGEX PATTERNS ---------
codecov_pattern = re.compile(
    r'<Event[^>]*Type="CodeCoverage"[^>]*Comment="([^"]+)"[^>]*>',
    re.DOTALL
)

mrs_pattern = re.compile(
    r'<Event[^>]*Type="MasterRecoveryState"[^>]*StatusCode="(\d+)"[^>]*>',
    re.DOTALL
)

time_pattern = re.compile(r'Time="([\d.]+)"')

def classify_reason(comment: str) -> str:
    c = comment.lower()

    # ---- Simulation boot / test harness setup ----
    if "simulation start" in c:
        return "SIMULATION_START"

    # ---- Encryption toggles ----
    if "disable encryption" in c or "encryption" in c:
        return "ENCRYPTION_CONFIG_CHANGE"

    # ---- SSL toggles ----
    if "ssl" in c:
        return "SSL_CONFIG_CHANGE"

    # ---- Storage engine choice (ssd, rocksdb, ssd-2) ----
    if "ssd" in c or "storage engine" in c:
        return "STORAGE_ENGINE_SELECTION"

    # ---- Redundancy mode change ----
    if "redundancy mode" in c or "single redundancy" in c:
        return "REDUNDANCY_MODE_CHANGE"

    # ---- IPv6 locality / network placement ----
    if "ipv6" in c or "locality" in c:
        return "IPV6_LOCALITY_CHANGE"

    # ---- Disk buffer / write stalls ----
    if "write buffer" in c or "didn't write everything" in c or "io_error" in c:
        return "disk_write_pressure"

    # ---- Commit unknown results / dummy tx ----
    if "commit_unknown_result" in c or "dummy transaction" in c:
        return "transaction_commit_stall"

    # ---- Cluster / master failure paths ----
    if "clusterwatchdatabase" in c or "master failed" in c:
        return "master_role_failure"

    # ---- Config never created startup fault ----
    if "configuration_never_created" in c or "never created" in c:
        return "configuration_missing"

    # ---- Log / TLog pipeline faults ----
    if "tlog" in c or "logrouter" in c or "spill" in c:
        return "tlog_failure"

    # ---- Storage server worker death ----
    if "workerfailed" in c or "sharedtlogfailed" in c:
        return "storage_server_failure"

    # ---- Network partitions / broken connections ----
    if "connection_failed" in c or "peeraddr" in c or "peeraddress" in c:
        return "network_partition"

    # ---- Transaction system metadata stalls ----
    if "reading_transaction_system_state" in c:
        return "metadata_state_stall"

    # ---- Transaction server recruitment delays ----
    if "recruiting_transaction_servers" in c or "initializing_transaction_servers" in c:
        return "recruitment_delay"

    return "unknown"


# --------- CONFIG ---------
MAX_BACKWARD_LINES = 100
MAX_BACKWARD_SECONDS = 10.0

def parse_time(line):
    m = time_pattern.search(line)
    if not m:
        return None
    try:
        return float(m.group(1))
    except:
        return None

def extract_recoveries(xml_text):
    """
    New strategy:
    1. Find all MasterRecoveryState events.
    2. For each StatusCode=0 (recovery start),
       scan backward for the closest CodeCoverage.
    3. Collect forward until StatusCode=14.
    """

    # Split into lines for positional scanning
    lines = xml_text.splitlines()

    # Pre-compute per-line:
    # (time, statuscode)
    meta = []
    for ln in lines:
        t = parse_time(ln)
        mrs = mrs_pattern.search(ln)
        code = mrs.group(1) if mrs else None
        meta.append((ln, t, code))

    events = []
    n = len(lines)

    # Find all recovery start lines
    recovery_starts = [i for i, (_, _, code) in enumerate(meta) if code == "0"]

    for idx, start_i in enumerate(recovery_starts):
        # Determine previous recovery start index
        prev_start_i = recovery_starts[idx - 1] if idx > 0 else None

        # Scan backward for CodeCoverage
        reason = None
        reason_time = None
        reason_line = None

        (_, t_start, _) = meta[start_i]

        # backward search
        j_limit = prev_start_i + 1 if prev_start_i is not None else 0
        j_limit = max(j_limit, start_i - MAX_BACKWARD_LINES)

        j = start_i - 1
        while j >= j_limit:
            line_j, t_j, _ = meta[j]

            # time window threshold
            if t_start and t_j and (t_start - t_j > MAX_BACKWARD_SECONDS):
                break

            m = codecov_pattern.search(line_j)
            if m:
                reason = m.group(1).strip()
                reason_time = t_j
                reason_line = j
                break
            j -= 1

        if not reason:
            # no CodeCoverage close enough → skip
            continue

        # Collect recovery MRS lines forward until StatusCode=14
        context_lines = []
        k = start_i
        while k < n:
            line_k, _, code_k = meta[k]
            if code_k is not None:
                context_lines.append(line_k)
                if code_k == "14":
                    break
            k += 1

        events.append({
            "reason": reason,
            "canonical_reason": classify_reason(reason),
            "context": "\n".join(context_lines),
            "reason_meta": {
                "line": reason_line,
                "time": reason_time
            },
            "recovery_meta": {
                "start_line": start_i,
                "end_line": k
            }
        })

    return events


# --------- MAIN ---------
all_samples = []
for path in glob.glob("ai_analysis/samples/*.xml"):
    with open(path, "r") as f:
        text = f.read()
    recs = extract_recoveries(text)
    for r in recs:
        r["file"] = path
    all_samples.extend(recs)

out_path = "ai_analysis/recovery_dataset_prod.json"
with open(out_path, "w") as f:
    json.dump(all_samples, f, indent=2)

print(f"✅ Extracted {len(all_samples)} recoveries → saved to {out_path}")