# Problem Summary

## Problem ID  
**Cluster 7 — Commit Proxy Pipeline Crash**

---

# Which clusters best match this incident

**Primary:** Cluster 7 (commit_proxy_pipeline_crash)  
**Secondary downstream effects (not root cause):**  
- Cluster 0 (recovery_restart_cascade)  
- Cluster 9 (configuration/membership instability)  
- Cluster 6 / 13 (post-crash pressure / workload spikes)

---

# Overview

### What is this problem?  
A commit-proxy–led pipeline crash repeatedly terminates critical transactional roles (proxy, resolver, master), causing the system to restart recovery multiple times.

### Why does it matter?  
When commit proxies fail, no transactions can be committed, metadata cannot progress, and master leadership collapses. The entire cluster becomes stuck in recovery loops because no component can advance version state.

---

# Test Context

### Which FoundationDB test(s) trigger this?  
Chaos / fault-injection test or simulation logs containing process-level failures.

### What the test does:  
Injects failures at the proxy or resolver layer and forces the coordinator to re-elect leadership, retriggering recovery.

### Expected vs unexpected failure:  
The recovery is expected; the repeated cascading crash after proxy failure is not.  
FDB is designed to recover once, not bounce repeatedly.

---

# Key Indicators

## Metrics to Check (priority order)

### Commit Proxies Alive Count  
`cluster.commit_proxies.alive`  
- **Normal:** >0, stable  
- **Problem threshold:** becomes 0 or fluctuates  
- **Meaning:** loss of commit proxies → pipeline broken → master death

### Recovery State / Epoch  
`cluster.recovery.phase`  
- **Normal:** advances steadily (0→14)  
- **Problem threshold:** looping 0→4→7→0 repeatedly  
- **Meaning:** recovery attempts cannot complete because critical roles keep failing

### Resolver Process Health  
`cluster.resolvers.active`  
- **Normal:** stable  
- **Problem threshold:** drops on crash  
- **Meaning:** resolver failure breaks transaction pipeline → commit_unknown_result

---

# Log Patterns

## Primary indicators — strongest evidence of CL7

Examples from your traces:
- CommitProxyTerminated (Error=worker_removed)  
- ResolverTerminated (Error=worker_removed)  
- MasterTerminated (Error=worker_removed)  
- CommitDummyTransactionError (commit_unknown_result)  
- CCWDB + ClusterRecoveryRetrying (Error=commit_proxy_failed)

**What this means:**  
This is textbook commit pipeline collapse. The proxy fails first → resolver destabilizes → master aborts → recovery restarts.

---

## Secondary indicators (downstream fallouts)

- RecruitStorageNotAvailable (no_more_servers)  
- DiskNearCapacity  
- SlowSSLoopx100  
- RelocateShard_StartMoveKeys storm  

**Interpretation:**  
These are stress reactions after the pipeline collapses — not causes.  
FDB is trying to reconfigure, rebalance, re-assign servers, and fails repeatedly because the core commit path is unstable.

---

## False positives to ignore

- FileOpenError / file_not_found bursts  
- PeerDestroy  
- IncomingConnectionError timeouts  

**Why to ignore:**  
These show consequence of node instability, filesystem churn, or simulation artifacts.  
They never generate the master termination by themselves.  
They appear after the commit pipeline is already broken.

---

# Timeline & Sequence (root cause oriented)

**T+0s** — Commit proxy crashes → CommitProxyTerminated  
**T+1s** — ResolverTerminated follows immediately  
**T+2s** — MasterTerminated → new epoch begins  
**T+3–5s** — ClusterRecoveryRetrying + CCWDB (commit_proxy_failed)  
**T+X** — Recovery restarts, temporary role recruitment  
**T+X+Δ** — Repeated recruit failures / job starvation / disk pressure  
**T+X+Δ** — System stuck in cascading restarts → no progress  

### Key pattern:
When MasterTerminated is preceded by CommitProxyTerminated or ResolverTerminated → root cause is always CL7, not CL0 or CL12.

---

# How to Interpret the Evidence

### If you see CommitProxyTerminated FIRST:
**This means:** pipeline collapse  
**Likely root cause:** CL7  
**Next check:** resolver + master logs in ±200ms  

### If you see ClusterRecoveryRetrying without proxy kill:
**This means:** recovery cascade (CL0)  
**Not root cause**  
**Next check:** coordinator mismatch / epoch loops  

### If DiskNearCapacity or SlowSSLoopx100 dominates:
**This means:** storage pressure (CL6 or CL13)  
**Downstream**  
**Next check:** when did it appear relative to proxy death?  

### If you see multiple roles die at once:
**Interpretation:** cascading (CL15)  
Still triggered by earlier root cause.

---

# Root Cause Analysis

### 1. Commit proxy failure (most likely)
**Why this happens:**  
Thread hang, OOM, internal bug, resolver deadlock, or dependency failure.

**Evidence:**  
CommitProxyTerminated first, multiple times.

**Confirm:**  
Look at immediate predecessor of MasterTerminated.  
If proxy loss precedes it → CL7 is confirmed.

---

### 2. Resolver crash
**Why:**  
Tied to proxy load and transaction state machine.

**Evidence:**  
ResolverTerminated after proxy death, Error=worker_removed.

**Confirm:**  
Look for resolver termination in same timestamp cluster.

---

### 3. Metadata pipeline starvation
**Why:**  
No commit → versions stall → recovery forced.

**Evidence:**  
CCWDB + ClusterRecoveryRetrying with commit_proxy_failed.

**Confirm:**  
Coordinator tries recovery without transaction progress.

---

# Diagnostic Checklist

- Check if proxy or resolver dies FIRST  
- Check if master death is secondary  
- Verify commit_unknown_result  
- Inspect ClusterRecoveryRetrying reason  

Only then look at:
- RecruitStorageNotAvailable  
- DiskNearCapacity  
- SlowSSLoopx100 bursts  

**Root cause always comes before these.**

---

# Related Problems

### Often occurs alongside:
- CL0 recovery cascade — because master keeps dying  
- CL6 storage pressure — because shards relocate mid-recovery  
- CL13 workload spikes — retry storms after recovery  

### Can be confused with:
- CL12 resource exhaustion (too many open files)  

**Key difference:** resource exhaustion doesn’t kill proxy and resolver in ~200ms intervals.

### Can cause:
- CL0 cascading recovery loops  
- CL9 config churn  
- CL6 performance collapse after recovery  

---

# Recommended Fix / Mitigation

- Stabilize commit proxies:  
  - Increase CPU/memory limits  
  - Pin proxies to dedicated machines or cores  
  - Avoid co-locating proxies with heavy workloads  
- Investigate proxy → resolver network or timeout config  
- Reduce chaos kill rate when simulating  
- Enable diagnostic traces around CommitProxy initiation & state machine  

---

# Final Takeaway

**The entire incident is a proxy-led pipeline collapse (Cluster 7).  
Everything else — recruitment failures, disk pressure, shard storms — is the ecosystem reacting to the death of the transactional brain.**
