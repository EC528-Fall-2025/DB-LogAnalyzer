# Problem Summary

## Problem ID  
Commit Proxy Pipeline Crash (Cluster 7 dominant, Cluster 5/0 secondary)

---

# Overview

### What is this problem?  
Under the CloggedStorefront workload, commit proxies repeatedly exit mid-recovery and the master logs `commit_proxy_failed`. In 17 of the 24 windows, `CommitProxyTerminated` + `ClusterRecoveryRetrying(commit_proxy_failed)` fires, with **3,347 `CommitDummyTransactionError`** and **3,302 `TLogQueueCommitSlow`** entries captured by the new `.important.json` extractor. A few windows show Cluster 5 (TLog failure) or Cluster 0 (no_more_servers) handoffs, while others (Cluster 6) are just post-recovery storage metrics.

### Why does it matter?  
Every proxy crash **collapses the commit/resolver pipeline**: clients see `commit_unknown_result`, GRV stalls, and all writes stop until a new proxy set comes up. When this happens multiple times per log, the cluster never reaches a stable steady state. In production, this would be a **severe outage** requiring action (proxy throttling, TLog tuning, disk fixes, etc.).

---

# Test Context

### Which FoundationDB test(s) trigger this?  
- **Test name(s):** Log 10 – “CloggedStorefront” simulation.  
- **What the test does:** floods commit proxies with storefront-style traffic and simultaneously injects log/TLog faults.  
- **Expected vs unexpected failure:** a single proxy crash is tolerable; but **17 Cluster 7 windows + two Cluster 5 windows + one Cluster 0 window** means the pipeline is fundamentally unstable.

---

# Key Indicators

## 1. Commit proxy lifecycle (`CommitProxyTerminated`, `CCWDB commit_proxy_failed`)
- **Normal:** ≈0 per run  
- **Problem:** 72 terminations / 24 windows; multiple `commit_proxy_failed` entries  
- **Meaning:** commit proxies exit mid-recovery and force master restarts

## 2. Dummy transaction health (`CommitDummyTransactionError`)
- **Normal:** rare, single-digit  
- **Problem:** bursts up to 2,234 per window; total 3,347  
- **Meaning:** commit pipeline cannot acknowledge transactions → clients see `commit_unknown_result`

## 3. TLog queue latency (`TLogQueueCommitSlow`)
- **Normal:** near zero; LoggingDelay <0.1s  
- **Problem:** hundreds per window, LoggingDelay = 1s  
- **Meaning:** TLogs confirm proxies are bottlenecked; commits cannot drain

---

# Log Patterns

## Primary indicator(s) – confirm Cluster 7:
