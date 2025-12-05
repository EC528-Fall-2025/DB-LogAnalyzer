# Problem Summary

## Problem ID  
Log9-ClogWithRollbacks – No-More-Servers Recovery Loop

---

## Which clusters best match this incident. Provide Cluster IDs.  
Cluster 0 – recovery_restart_cascade (dominant)  
Cluster 6 – storage_engine_pressure (monitoring-only snapshots after failed recovery)

---

# Overview

### What is this problem?  
Recovery restarts again and again because the ClusterController cannot recruit storage servers. Each attempt logs RecruitStorageNotAvailable (Error=1008 no_more_servers) immediately followed by FileOpenError(file_not_found) on logqueue / sqlite-wal paths (e.g., /simfdb/.../logqueue-V_7-*.fdq). These missing files prevent TLogs/SS from starting, so the master never reaches fully_recovered. A few windows are just post-recovery telemetry (Cluster 6), but none of them show a stable cluster.

### Why does it matter?  
Transactions cannot make progress: commit proxies and TLogs are repeatedly torn down, clients see commit_unknown_result, and data distribution/audit cannot rebuild storage teams. In a production cluster, this requires manual intervention (fix disk/mounts, restore logqueue files, rerun recovery) because waiting for automation will never finish recovery.

---

# Test Context

### Which FoundationDB test(s) trigger this?  
- Test name(s): “Log 9 – ClogWithRollbacks” scenario in Joshua simulation.  
- What the test does: mixes heavy rollback workload with chaos that deletes or hides logqueue/WAL files.  
- Expected vs unexpected failure: expected to observe recovery events, but not an endless recovery loop; here the failure persists across all 22 windows.

---

# Key Indicators

## Storage recruitment  
- Repeated RecruitStorageNotAvailable with ErrorCode=1008 (no_more_servers) in every recovery window.  
- No healthy storage team can be rebuilt; status would show storage count at 0 or below redundancy.

## On-disk log/WAL existence  
- FileOpenError(file_not_found) on logqueue-V_7-*.fdq and log2-V_7-*.sqlite-wal exactly when CC tries to start storage/TLog processes.  
- Indicates disks/mounts are missing or files were deleted.

---

# Log Patterns

## Primary indicator(s) – these logs strongly indicate this problem:  
- `RecruitStorageNotAvailable Error=no_more_servers Machine=2.0.1.0:1 Roles=CC,CD,SS`  
- `FileOpenError Error=file_not_found File=/simfdb/.../logqueue-V_7-5168af5e1e29-1.fdq`  
- `ClusterRecoveryRetrying Error=no_more_servers`

## Secondary indicator(s):  
- `CommitProxyTerminated / MasterTerminated (worker_removed)` immediately after the above events.  
- `SlowSSLoopx100 / DiskNearCapacity` snapshots (Cluster 6) once recovery temporarily restarts.

## False positives to ignore:  
- A single FileOpenError on /simfdb/.../processId right at startup that does not repeat.

---

# Timeline & Sequence

- T+0s: recovery begins; CC logs RecruitStorageNotAvailable(no_more_servers).  
- T+4s: FileOpenError(file_not_found) on logqueue-V_7-*.  
- T+5s: ClusterRecoveryRetrying(no_more_servers); commit proxy logs commit_unknown_result.  
- T+30s onward: pattern repeats across all 22 windows; occasional Cluster 6 snapshots show storage metrics but recovery never completes.

---

# Example from Actual Test Run

Test: `trace.0.0.0.0.49.1763869069.qBqtOp.0.1.xml`  
- [04.56] RecruitStorageNotAvailable Error=no_more_servers Machine=2.0.1.0:1  
- [04.59] FileOpenError file_not_found File=/simfdb/.../logqueue-V_7-5168af5e1e29-1.fdq  
- [04.60] ClusterRecoveryRetrying Error=no_more_servers  
- [04.61] CommitDummyTransactionError commit_unknown_result  

Metrics: recovery state stuck in recruiting_transaction_servers; storage_servers=0; commit proxies show minimal successful throughput.

---

# How to Interpret the Evidence

## If you see RecruitStorageNotAvailable + FileOpenError(file_not_found):  
- Meaning: recovery needs logqueue/WAL files that are missing.  
- Likely root cause: deleted logqueue files, broken mounts, or misconfigured data directories.  
- Next steps: check /simfdb path existence, fix disk/mount, restore files, rerun recovery.

## If you see SlowSSLoopx100 + DiskNearCapacity without recruitment errors:  
- Meaning: Cluster 6 monitoring window; not the primary fault.  
- Next steps: confirm whether recovery actually completed; if not, still treat as Cluster 0.

## If only one machine shows FileOpenError:  
- Interpretation: localized disk loss; replacing/restoring that host may resolve.  
- In Log9 all machines hit the same state → system-wide issue.

---

# Root Cause Analysis

1. Logqueue/WAL files removed or not mounted  
   - Evidence: repeated FileOpenError on /simfdb/.../logqueue* and log2*.sqlite-wal.  

2. Storage team configuration too strict / too few physical servers  
   - Evidence: RecruitStorageNotAvailable(no_more_servers) even before logqueue errors appear.  

3. Commit proxies failing because storage never stabilizes  
   - Evidence: occasional ClusterRecoveryRetrying(commit_proxy_failed) right after recruitment failure.

---

# Diagnostic Checklist

- [ ] Verify /simfdb/.../logqueue-V_7-* exists and is readable.  
- [ ] Check status json: storage_servers count, team health, recruitment warnings.  
- [ ] Review all ClusterRecoveryRetrying reasons (majority should be no_more_servers).

---

# Related Problems

- Often occurs alongside Cluster 6 (post-recovery storage pressure) because monitoring windows are captured even when recovery fails.  
- Can be confused with Cluster 5 (tlog_failure_recovery) if you only look at commit proxy restarts; the decisive evidence here is logqueue file_not_found + no_more_servers.  
- Can cause Cluster 7 symptoms (commit_proxy_pipeline_crash) because commit proxies lose their log streams when recovery never stabilizes.
