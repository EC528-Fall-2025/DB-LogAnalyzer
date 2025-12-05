# Problem ID  
**Cluster 7 — commit_proxy_pipeline_crash**

---

# Overview  

### What is this problem?  
A commit proxy or resolver crashes or becomes unresponsive, breaking the commit pipeline. The master is forced to terminate and a full cluster recovery is triggered.

### Why does it matter?  
All transaction commit paths fail. Clients see GRV/commit timeouts, unresolved transactions, and degraded throughput. Recovery pauses system progress until a new master and proxies re-form.

---

# Test Context  

### Which FoundationDB test(s) trigger this?  
• Likely triggered by chaos/reliability tests that kill proxies or remove network connectivity.

### What the test does:  
Rapid failure injection (kill, resource spikes, process termination), forcing proxies/resolvers to hang, crash, or disconnect.

### Expected vs unexpected failure:  
Partially expected in stress/chaos tests, but the specific sequence of cascading termination + recovery pipeline collapse is not normal runtime behavior.

---

# Key Indicators  

## Metrics to Check  

### Commit proxy progress  
• ProxyMetrics.TxnCommitOut, ProxyMetrics.CommitBatchOut  
**Normal range:** nonzero, stable or rising  
**Problem threshold:** sudden drop to near zero  
**Meaning:** proxy commit pipeline stalled.

### Version progression  
• ProxyMetrics.LastAssignedCommitVersion vs CommittedVersion  
**Normal:** Assigned >= Committed steadily  
**Problem:** diverges, stalls or plateaus  
**Meaning:** pipeline cannot produce or finalize versions.

---

# Log Patterns  

### Primary indicators — strongly identify the problem:  
`CommitProxyTerminated`, `ResolverTerminated`, `MasterTerminated` cascading or clustered together.  
These are fatal to the commit pipeline. They show the pipeline broke, not just slowed.

**Example indicators from your logs:**  
• Start of window: ProxyMetrics from all proxies at Time=74.5 showing throughput, followed by  
• waves of `RelocateShard` + `FetchKeys` cancellations (`Error=operation_cancelled`)  
• sudden failure events leading to `MasterRecoveryState` begin  

**What these lines mean:**  
A working commit path is running, then critical failure signals propagate from SS relocation → proxy pipeline stalls → CC forces recovery.

### Secondary indicators:  
• Cancellation storms (`RelocateShard_StartMoveKeys`, `FetchKeys`) — symptoms, not the cause  
• No progress in commit path while shard movement attempts continue  

### False positives to ignore:  
• High RelocateShard noise alone  
• Data distribution churn without role termination  

Shard relocation on its own is normal and should not force master termination.

---

# Timeline & Sequence  

**T+0s**  
Proxies are active, commit/resolve pipeline running normally.

**T+2–5s**  
RelocateShard cancellations, FetchKeys cancellations begin to spike, cluster trying to rebalance while commit load active.

**T+6–10s**  
Proxy responsiveness drops, commit batch output decays, transaction commit path is no longer progressing.

**T+10–12s**  
Master terminates or ClusterController initiates recovery because commit pipeline is broken.

### Key timing patterns:  
- If `CommitProxyTerminated` precedes `MasterTermination` → root cause is proxy pipeline failure.  
- If `MasterTerminated` appears first → lease/clock skew scenarios (Cluster 16) instead.

---

# Example from Actual Test Run  

### What happened:  
Commit proxies show good throughput, then shard relocation and fetch operations begin failing en masse. No corresponding resource exhaustion or corruption. Shortly after, the commit pipeline fails → recovery triggered.

### Metrics observed:  
`TxnCommitOut` falls from stable traffic to near-zero.  
`CommittedVersion` no longer advances.

---

# How to Interpret the Evidence  

### If you see `CommitProxyTerminated` / `ResolverTerminated` first:  
**This means:** pipeline component died  
**Likely root cause:** process crash / OOM / deadlock  
**Check next:** `MasterTermination` + immediate `ClusterRecoveryState`  

### If instead you see `no_more_servers` / `FileOpenError` loops:  
**This means:** recovery restart loop  
**Likely root cause:** *cluster 0 recovery_restart_cascade*

### If only shard relocations / FetchKeys cancellations:  
**Interpretation:** secondary stress  
**Not root cause, ignore as primary signal.**

### If multiple different roles fail rapidly:  
**Interpretation:** cascading failure (*Cluster 15*)

---

# Root Cause Analysis  

### Most likely causes (ranked):  

#### **1. Commit proxy process crash or deadlock**  
Why: commits stop, pipeline stops emitting versions, Master gives up.  
How to confirm: check Proxy termination logs; look for segfault/OOM, role closed, lost connection.  
Typical tests: chaos kill, crash-injection.

#### **2. Resolver failure / unresponsive resolver**  
Why: commit resolution stalls → pipeline breaks.  
How to confirm: Resolvers stop responding to version assignment or conflict resolution.  
Typical tests: resolver kill, network partition.

#### **3. Proxy–Resolver network partition**  
Why: proxies alive but logically “blind”.  
How to confirm: no successful communication logs, but no local crash logs.  
Typical tests: link failure, firewall chaos.

---

# Diagnostic Checklist  

When you see this problem, check these in order:

• Is there a `CommitProxyTerminated` or `ResolverTerminated` first?  
Why: determines whether the pipeline loss is root cause.

• Did `MasterTerminated` happen as a reaction?  
Why: if yes, master is secondary victim, not originator.

• Are shard operations failing before termination?  
Why: confirms symptom cascade (pressure → crash), not primary.

---

# Related Problems  

### Often occurs alongside:  
**Cluster 0 — recovery_restart_cascade**  
Because repeated recovery attempts after proxy death re-enter cluster recovery.

### Can be confused with:  
**Cluster 15 — cascading_failure_recovery**  
**Difference:** Cluster 15 shows different roles failing independently; Cluster 7 pivot is the commit pipeline breaking first.

### Can cause:  
**Cluster 5 — tlog_failure_recovery**  
If the crash corrupts/resets proxy → logs do not ack → recovery.

---

# Summary  

This is exactly how you should present the incident:  
**The root cause is the pipeline collapse (Cluster 7), and everything else — relocations, cancellations, master recovery — is downstream reaction, not the origin.**
