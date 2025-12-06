# Problem Summary

## Problem ID  
**Commit proxy–led recovery cascade with missing file errors**

---

# Which clusters best match this incident

**Primary:** Cluster 7 (commit_proxy_pipeline_crash)  
**Secondary:** Cluster 0 (recovery_restart_cascade), Cluster 5 (tlog_failure_recovery)

---

# Overview

### What is this problem?  
The cluster goes through a commit-proxy–led crash: the commit proxies and master terminate after the transaction subsystem is restarted, leading to a full recovery cycle. This happens in a context where earlier TLog termination and repeated file-not-found errors are present.

### Why does it matter?  
During this incident, client transactions briefly lose availability or see commit errors (commit_unknown_result, timeouts) while the cluster recovers. Operators experience alarming bursts of critical errors (FileOpenError, CommitProxyTerminated, MasterTerminated) followed by recovery messages instead of smooth, continuous service.

---

# Test Context

### Which FoundationDB test(s) trigger this?  
**Test name(s):** Not explicitly named in the trace, but the pattern matches a chaos / fault-injection recovery test that kills TLogs and proxies and stresses recovery.

### What the test does:  
Repeatedly terminates components (TLog, commit proxy, master) under load, causing recovery, log replay, and reconfiguration; in this run, it also appears to start with missing or misconfigured files for some processes.

### Expected vs unexpected failure:  
The recovery itself is expected for such tests, but the specific combination  
**“missing file(s) + TLog termination + commit-proxy pipeline crash”**  
is an undesirable compound failure mode that makes diagnosis harder.

---

# Key Indicators

## Metrics to Check

### RocksDB write/read pressure on storage  
Metric: **[RocksDBMetrics.WriteStall / Compaction / L0Files]** (generic metric group; exact path varies by setup)  
- **Normal range:** Memtable and SST counts stay within configured limits; write stall is 0.  
- **Problem threshold:** Persistent write stall > 0, huge L0 files spike, or compaction backlog growing rapidly.  
- **What it means when abnormal:** Would indicate storage-engine pressure (Cluster 6), **which we do NOT see here**, helping us rule that out as root cause.

### Proxy throughput / error rate  
Metric: **[ProxyMetrics.Commit.MeanLatency / ErrorRate]**  
- **Normal range:** Low error rate; commit latency within a few ms–tens of ms under test load.  
- **Problem threshold:** Sudden spikes in commit errors or latency around the times of CommitProxyTerminated / MasterTerminated.  
- **What it means when abnormal:** Aligning spikes in proxy error/latency with CommitProxyTerminated and commit_unknown_result helps confirm that the commit-proxy pipeline failure is the event that actually forces master termination and recovery (Cluster 7).

---

# Log Patterns

## Primary indicator(s) – These logs strongly indicate this problem:

Representative patterns from the combined critical logs (49., 1266., 3083.*):

### Commit-proxy / master pipeline failure sequence:
- **CommitProxyTerminated  Error=worker_removed**  
- **ResolverTerminated     Error=worker_removed**  
- **MasterTerminated       Error=worker_removed**  
- **CommitDummyTransactionError  Error=commit_unknown_result**  
- **CCWDB                 Error=commit_proxy_failed**  
- **ClusterRecoveryRetrying  Error=commit_proxy_failed**

### TLog and transaction subsystem restart:
- **TLogTerminated         Error=worker_removed**  
- **RestartingTxnSubsystem**

### What these log lines mean:
This chain tells you the commit pipeline between proxies, resolvers, and master broke:  
- the commit proxies or their connections fail first  
- the ClusterController detects the problem  
- the master is terminated  
- the transaction subsystem is restarted  

The **commit_unknown_result** and **commit_proxy_failed** entries confirm the failure is specifically in the commit pipeline (Cluster 7).  
**TLogTerminated** and **RestartingTxnSubsystem** show that logs and the transaction system are also being restarted as part of recovery → giving **Cluster 5** as a supporting factor, **but not the primary root cause**.

---

## Secondary indicator(s) – These logs often appear alongside the primary:

Recovery cascade and environment/resource symptoms:
- **ClusterRecoveryRetrying  Error=no_more_servers**  
- **RecruitStorageNotAvailable  Error=no_more_servers**  
- **FileOpenError  Error=file_not_found** (repeated bursts early in the trace)  
- **SlowSSLoopx100**  
- **DiskNearCapacity**  
- **QuietDatabaseStartFail**

### What these lines mean:
- `ClusterRecoveryRetrying` + `no_more_servers` and `RecruitStorageNotAvailable` show that the system is struggling to recruit enough servers as the pipeline crashes → **Cluster 0: recovery_restart_cascade**.
- `FileOpenError file_not_found` indicates environmental misconfiguration or missing files, acting as an underlying trigger.
- `SlowSSLoopx100`, `DiskNearCapacity`, and `QuietDatabaseStartFail` show **downstream stress**, not the root cause.

---

## False positives to ignore

These logs **alone** do NOT define this incident:
- Isolated **DiskNearCapacity** without proxy/master termination  
- **SlowSSLoopx100** spam during heavy workloads  
- Single **FileOpenError** not associated with recovery loops  

### Why ignore these?
They appear in healthy or routine restart cases.  
Only the combination with the **commit-pipeline failure sequence** and repeated **ClusterRecoveryRetrying** makes this a Cluster 7 + 0 + 5 issue.

---

# Timeline & Sequence

## How does this problem unfold over time?

### **T+0s–T+5s**  
- Bursts of **FileOpenError file_not_found**  
- Early **ClusterRecoveryRetrying** & **RecruitStorageNotAvailable (no_more_servers)**  
→ Recovery already unstable from missing files.

### **T+5s–T+30s**  
- Ongoing recovery attempts  
- **SlowSSLoopx100**, **DiskNearCapacity**  
- **TLogTerminated**, **RestartingTxnSubsystem**  
→ System cannot stabilize; TLogs restart under load.

### **T+30s–T+60s and beyond**  
- **CommitProxyTerminated**  
- **ResolverTerminated**  
- **CCWDB commit_proxy_failed**  
- **CommitDummyTransactionError commit_unknown_result**  
- **MasterTerminated**  

→ Classic **commit-proxy pipeline crash** followed by full recovery.

### Key pattern:
If proxy/resolver termination and `commit_proxy_failed` occur shortly after  
`RestartingTxnSubsystem` and `no_more_servers`, it shows:

**unstable recovery + missing files** → **resource churn** → **commit pipeline collapse (Cluster 7)**.

---

# Example from Actual Test Run

### What happened:
- System begins with missing or misconfigured files → **FileOpenError** bursts.  
- Recovery attempts fail due to lack of servers → **no_more_servers**.  
- Disk and storage server stress signs build.  
- TLogs terminate and restart.  
- Then commit proxies and resolvers die → **commit_proxy_failed** → **master killed** → recovery restarts.

### Observed logs (conceptual):
- `[~4s] FileOpenError file_not_found`  
- `[4–5s] ClusterRecoveryRetrying no_more_servers`  
- `[20–30s] SlowSSLoopx100, DiskNearCapacity`  
- `[30–35s] RestartingTxnSubsystem; TLogTerminated`  
- `[31–32s] CommitProxyTerminated, ResolverTerminated, MasterTerminated`

### Metrics:
Proxy and RocksDB metrics show stress but **not classic storage saturation**, ruling out Cluster 6.

---

# How to Interpret the Evidence

### Pattern: CommitProxyTerminated + ResolverTerminated + MasterTerminated + commit_proxy_failed  
→ Commit pipeline broken → **Cluster 7**

### Pattern: ClusterRecoveryRetrying no_more_servers + FileOpenError bursts before proxy crash  
→ Underlying instability in recruitment & filesystem → **Cluster 0 / environmental issue**

### Pattern: TLogTerminated + RestartingTxnSubsystem  
→ TLog churn contributes to recovery instability → **Cluster 5 (supporting)**

---

# Root Cause Analysis

### 1. Misconfigured or missing files (environmental root cause)
- Repeated **FileOpenError file_not_found** confirms this.
- Prevents recovery from recruiting required roles.

### 2. TLog termination during recovery (Cluster 5 – supporting)
- **TLogTerminated worker_removed**  
- **RestartingTxnSubsystem**

### 3. Commit-proxy pipeline crash (Cluster 7 – primary)
- Occurs after recruitment and TLog instability.
- Confirmed via:  
  - **CommitProxyTerminated**  
  - **ResolverTerminated**  
  - **commit_proxy_failed**  
  - **commit_unknown_result**  
  - **MasterTerminated**

---

# Diagnostic Checklist

1. **FileOpenError file_not_found bursts**  
2. **ClusterRecoveryRetrying no_more_servers**  
3. **Confirm commit pipeline failure:**  
   - CommitProxyTerminated  
   - ResolverTerminated  
   - commit_proxy_failed  
   - commit_unknown_result  
   - MasterTerminated  

---

# Related Problems

### Often occurs alongside:
- Cluster 5 (tlog_failure_recovery)

### Can be confused with:
- Cluster 6 (storage_engine_pressure)  
**Difference:** Here, missing files + proxy failures dominate.

### Can cause:
- Cluster 0 (recovery_restart_cascade)

---

# Final Takeaway

This incident is a **compound failure**:

- **Environmental / missing files** → destabilizes recovery  
- → **TLog termination** (Cluster 5)  
- → **Commit-proxy pipeline collapse** (Cluster 7 primary)  
- → **Recovery loops** (Cluster 0)

The dominant root cause classification is **Cluster 7**, with contributing factors from **Cluster 0** and **Cluster 5**.
