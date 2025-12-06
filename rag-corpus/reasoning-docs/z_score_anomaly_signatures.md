## 11. Z-Score Anomaly Signatures

### 11.1 Storage Version Lag Anomaly
**Typical Z-score progression:**
1. `StorageMetrics.VersionLag` z-score: 0.5 → 2.0 → 4.5+ (over 30-60s)
2. Concurrent `StorageMetrics.DiskQueueBytes` z-score: >3.0
3. `TLogMetrics.PoppedVersionLag` z-score rises: >2.5

**Thresholds:**
- **Warning:** VersionLag z-score >2.0
- **Critical:** VersionLag z-score >4.0
- **Severe:** VersionLag z-score >6.0 + sustained >30s

**Correlated metrics (should also spike):**
- `StorageMetrics.BytesFetched` (z >2.0)
- `StorageMetrics.MutationBytes` (z >2.0)

**Uncorrelated metrics (should remain normal):**
- `ProxyMetrics.TxnCommitOut` (z <1.5) — if this spikes too, different root cause

---

### 11.2 TLog Stall Signature
**Z-score pattern:**
1. `TLogMetrics.DurableVersionLag` z-score: 1.0 → 3.0 → 8.0+ (rapid climb)
2. `TLogMetrics.QueueDiskWrite` z-score: >5.0 (IO bottleneck)
3. `ProxyMetrics.TxnCommitOut` z-score: drops below -2.0 (commits stop)

**Distinctive signature:**
- TLog lag z-score climbs FASTER than storage lag (>0.5 z/sec)
- Proxies show commit rate collapse (z <-3.0)

---

### 11.3 False Positive: Workload Shift
**Z-score pattern:**
1. `ProxyMetrics.TxnCommitOut` z-score: 2.0-4.0 (sudden increase)
2. `StorageMetrics.FinishedQueries` z-score: 2.0-4.0 (matching increase)
3. **Key difference:** ALL metrics rise proportionally, no lag spikes

**Discriminator:**
- If VersionLag z-score <1.5 → normal workload increase, NOT a failure
- If VersionLag z-score >3.0 → genuine storage pressure

---

### 11.4 Ratekeeper Throttle Signature
**Z-score pattern:**
1. `RatekeeperMetrics.ReleasedTPS` z-score: drops to -2.0 or lower
2. `RatekeeperMetrics.TPSLimit` z-score: -1.5 to -3.0
3. `ProxyMetrics.TxnCommitOut` z-score: follows throttle down (-1.5 to -2.5)

**Healthy throttle vs. unhealthy:**
- **Healthy:** Throttle resolves within 10-30s, z-scores return to [-1, 1]
- **Unhealthy:** Throttle persists >60s, or cycles repeatedly (3+ times in 5 min)

---

### 11.5 Master Recovery Precursor
**Multi-metric z-score signature (30-60s window before recovery):**
1. `TLogMetrics.DurableVersionLag` z >4.0
2. `StorageMetrics.VersionLag` z >3.0  
3. `ProxyMetrics.TxnCommitOut` z <-2.0
4. `ClusterControllerMetrics` may show process kill events

**Confidence indicators:**
- **High confidence (recovery imminent):** 3+ metrics with z >3.0
- **Medium confidence:** 2 metrics with z >2.5
- **Low confidence:** Only 1 metric spiking (may self-recover)

---

## 12. Z-Score Baseline Considerations

### 12.1 Metrics with High Natural Variance
**Expect noisy z-scores (z=2-3 is normal):**
- `StorageMetrics.BytesFetched` (bursty reads)
- `MachineLoadDetail.CPUSeconds` (process scheduling noise)
- `DataDistributionMetrics.MoveInFlight` (data movement is sporadic)

**Recommendation:** Use z >3.5 thresholds for these metrics

### 12.2 Metrics with Low Natural Variance
**Expect stable z-scores (z=2.0 is significant):**
- `TLogMetrics.DurableVersionLag` (should be near-constant)
- `ProxyMetrics.TxnCommitOut` (steady under stable load)
- `StorageMetrics.VersionLag` (low unless problems occur)

**Recommendation:** Use z >2.0 thresholds for these metrics

---

## 13. Z-Score Temporal Patterns

### 13.1 Spike vs. Plateau vs. Ramp
**Spike (lasts <10s):**
- Z-score peaks rapidly (>5.0) then drops
- **Example:** Brief disk stall, temporary network hiccup
- **Action:** Log warning, continue monitoring

**Plateau (lasts 30s-5min):**
- Z-score reaches 3-5 and stays elevated
- **Example:** Storage queue saturation, sustained high load
- **Action:** Investigate correlated metrics, check for recovery

**Ramp (continuous climb over 60s+):**
- Z-score increases steadily (1 → 3 → 5 → 8+)
- **Example:** Version lag spiraling out of control
- **Action:** High confidence precursor to failure/recovery

### 13.2 Oscillating Z-Scores
**Pattern:** Z-score oscillates between -2 and +3 with period ~10-30s
- **Cause:** Feedback loop (e.g., throttle → backpressure → throttle releases → repeat)
- **Indicator:** System attempting self-regulation but struggling

---

## 14. Multi-Metric Z-Score Correlation Rules

### 14.1 Storage Pressure Fingerprint
**Required:** 
- `StorageMetrics.VersionLag` z >3.0 AND
- `StorageMetrics.DiskQueueBytes` z >2.5 AND
- `TLogMetrics.PoppedVersionLag` z >2.0

**Confidence:** 90%+ this is storage bottleneck

### 14.2 TLog Isolation Failure Fingerprint
**Required:**
- `TLogMetrics.DurableVersionLag` z >5.0 AND
- `ProxyMetrics.TxnCommitOut` z <-2.5 AND
- `StorageMetrics.VersionLag` z <2.0 (storage NOT the bottleneck)

**Confidence:** 85%+ this is TLog-specific issue

### 14.3 Cluster-Wide Overload Fingerprint
**Required:**
- `ProxyMetrics.TxnCommitOut` z >3.0 AND
- `StorageMetrics.FinishedQueries` z >3.0 AND
- `RatekeeperMetrics.ReleasedTPS` z >2.5 AND
- `MachineLoadDetail.CPUSeconds` z >2.5

**Confidence:** 95%+ this is legitimate high load (not failure)

---

## 15. Z-Score Diagnostic Decision Tree
```
IF TLogMetrics.DurableVersionLag z >4.0:
  ├─ IF ProxyMetrics.TxnCommitOut z <-2.0:
  │   └─ DIAGNOSIS: TLog stall → Master recovery imminent
  └─ ELSE:
      └─ CHECK: StorageMetrics.VersionLag z-score
          ├─ IF z >3.0: Storage pressure cascading to TLog
          └─ ELSE: TLog-specific issue (disk, process)

IF StorageMetrics.VersionLag z >3.0:
  ├─ CHECK: DiskQueueBytes z-score
  │   ├─ IF z >3.0: Disk I/O bottleneck
  │   └─ ELSE: CPU or network issue
  └─ CHECK: TLogMetrics.PoppedVersionLag
      └─ IF z >2.0: Storage falling behind durable version

IF ProxyMetrics.TxnCommitOut z >3.0:
  └─ CHECK: StorageMetrics.VersionLag z-score
      ├─ IF z <1.5: Normal workload increase (FALSE POSITIVE)
      └─ IF z >2.5: System struggling under load
```