# Event Sequence Patterns in FoundationDB Simulations

This document summarizes common event-level temporal patterns observed in FoundationDB simulation logs. These patterns help downstream RAG components identify meaningful system transitions, failure precursors, or recovery behavior.

---

## 1. Recovery Loop Patterns

### 1.1 Master Recovery Start
**Typical sequence (≈1–3s):**
1. `ClusterControllerProcessKilled` or injected fault  
2. `GotServerDBInfoChange`  
3. `RecruitingMaster` or `RejoiningCluster`  
4. `MasterRecoveryState` transitions  
5. `ProxyRecruitment` and `ResolverRecruitment`  
6. `MasterRecovered`

**Meaning:** A full cluster recovery has begun; all commit-path components will rotate.

---

## 2. Storage Engine Pressure Patterns

### 2.1 Version Lag Escalation
**Sequence:**
1. Rising `VersionLag` in `StorageMetrics`  
2. `QuietDatabaseStartFail` citing `MaxTLogPoppedVersionLag`  
3. `TLogError` or `SharedTLogFailed`  
4. Master recovery triggered

**Meaning:** Storage servers fall behind durable version progression, causing upstream TLog stalls.

---

## 3. TLog Failure Patterns

### 3.1 Slow → Stalled → Dead
**Sequence:**
1. Increase in `TLogMetrics.DurableVersionLag`  
2. `TLogPoppedVersionLag` spikes  
3. Proxies emit `tlog_stopped` errors  
4. `SharedTLogFailed` (e.g., `io_error 1510`)  
5. Master initiates recovery  
6. New TLogs recruited

**Meaning:** A TLog becomes unresponsive, halting commit persistence.

---

## 4. Ratekeeper Throttling Patterns

### 4.1 Load → Throttle → Decline
**Sequence:**
1. High CPU/disk load in `MachineLoadDetail`  
2. Rising latency in `RatekeeperMetrics`  
3. `RatekeeperThrottle` events  
4. Throughput temporarily decreases

**Meaning:** Ratekeeper slows the system to prevent overload.

---

## 5. Storage Queue Saturation Patterns

### 5.1 Queue Bytes Surge
**Sequence:**
1. `StorageMetrics` shows high `DiskQueueBytes`  
2. Increase in `DataMoveInFlight`  
3. Decline in `FinishedQueries`  
4. Potential `StorageServerBusy` events  
5. Cluster either stabilizes or enters recovery

**Meaning:** Heavy writes or disk contention cause storage queue pressure.

---

## 6. Shard Movement / Data Distribution Patterns

### 6.1 Shard Movement Initiated
**Sequence:**
1. `DataDistributionMoveStarted`  
2. Bursts of `MoveInUpdatesPrePersist`  
3. `MoveInDurable` events  
4. `DataDistributionMoveFinished`

**Meaning:** Data distribution is rebalancing or reacting to server health changes.

---

## 7. Commit Proxy / Resolver Failure Patterns

### 7.1 Commit Log Collapse
**Sequence:**
1. `ProxyMetrics` show commit outputs dropping to zero  
2. `ResolverMetrics` become unstable or missing  
3. Error injection or process kill event  
4. `MasterRecoveryState` rebuilds proxies/resolvers

**Meaning:** Commit path interruption forces the master to restart the commit pipeline.

---

## 8. Quiet Database Failure Patterns

### 8.1 Quiet State Impossible
**Sequence:**
1. `QuietDatabaseStart` issued  
2. Immediate `QuietDatabaseStartFail`  
3. Reasons include TLog lag or storage lag  
4. Often followed by master recovery

**Meaning:** The cluster cannot quiesce due to insufficient durable version progress.

---

## 9. Simulation Infrastructure Noise Patterns

### 9.1 VFS Activity Bursts
Typical sequence:
VFSAsyncFileConstruct
VFSAsyncFileOpened
VFSAsyncFileDestroyStart
VFSAsyncFileDestroy

csharp
Copy code

**Meaning:** Filesystem activity from simulation infrastructure, not a signal of cluster failure.

### 9.2 Connection Activity
Typical sequence:
Sim2Connection
ConnectionKeeper

yaml
Copy code

**Meaning:** Simulation networking noise; usually ignorable in diagnosis.

---

## 10. Chaos / Fault Injection Patterns

### 10.1 Intentional Failure Injection
**Sequence:**
1. `ChaosMetrics` indicating injected faults (kill, slow, isolate)  
2. Worker crashes or restarts  
3. Recovery cascade (master + proxies + logs)  
4. Cluster stabilizes or reforms configuration

**Meaning:** Expected behavior in chaos and reliability workloads.