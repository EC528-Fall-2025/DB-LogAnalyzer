# Problem Summary

## Problem ID  
**Cluster 9 — configuration_change_recovery (storage migration instability)**

---

# Overview

### What is this problem?  
During ConfigureStorageMigrationTest the storage migration / quiet database steps keep failing. `QuietDatabaseStartFail`, `BuildTeams` failures, and audit stalls prevent the migration from reaching a steady state.

### Why does it matter?  
Storage migration never completes. Operators see quiet database stuck, storage teams unhealthy, and transaction workloads must stay paused until configure succeeds.

---

# Test Context

### Which FoundationDB test(s) trigger this?
- Test name(s): **ConfigureStorageMigrationTest**
- What the test does: forces quiet database + storage migration while relocating shards.
- Expected vs unexpected failure: partial failure is expected under stress; repeated quiet-database failure means migration is unstable.

---

# Key Indicators

## Metrics to Check

### 1. Quiet database control  
`QuietDatabaseStartFail`, `QuietDatabaseConsistencyCheckStartFail`  
- Normal: 0  
- Problem: ≥1 entries (observed 3–6 per file)  
- Meaning: quiet database cannot enter/complete its phase.

### 2. Team build / migration  
`BuildTeamsLastBuildTeamsFailed`, `RecruitStorageNotAvailable`  
- Normal: no failures  
- Problem: repeated failures / `no_more_servers`  
- Meaning: configure cannot build new storage teams while migrating.

### 3. Audit / migration stall  
`AuditUtilCheckAuditProgressNotFinished`, `DDAuditStorageCoreError`  
- Normal: audit completes quietly  
- Problem: audit retries and never finishes  
- Meaning: migration and audit loops never converge.

---

# Log Patterns

### Primary indicators:
QuietDatabaseStartFail
QuietDatabaseConsistencyCheckStartFail
BuildTeamsLastBuildTeamsFailed

markdown
Copy code
These show the configure workflow is failing.

### Secondary indicators:
- `RecruitStorageNotAvailable`, `no_more_servers`
- `AuditUtilCheckAuditProgressNotFinished`
- `GrvProxyRateLeaseExpired`, `ProxyThrottleMetrics`

### False positives:
- Pure storage pressure without quiet-database failures  
- `TLogTerminated` without configure steps (then it’s Cluster 5)

---

# Timeline & Sequence

- **T+0s:** Storage migration starts; quiet database requested.  
- **T+3s:** QuietDatabaseStartFail / ConsistencyCheckStartFail logged.  
- **T+5s:** BuildTeams failures, RecruitStorageNotAvailable; audit retries start.  
- **T+10s:** Another quiet database attempt; migration never stabilizes.

### Key timing pattern  
Quiet database failure → team build failure → audit stall loop.  
Without quiet-database errors, treat as Cluster 6 or 0.

---

# Example from Actual Test Run

**Test:** Log 14 – `trace.0.0.0.0.9348.1763871323.qBqtOp.0.2.xml`  
**What happened:** QuietDatabaseStartFail and BuildTeams failures repeated; audit progress never finished; proxies reported rate lease issues.

### Logs observed:
[04:35:22] WARN QuietDatabaseStartFail Reasons=MaxTLogPoppedVersionLag
[04:35:24] WARN BuildTeamsLastBuildTeamsFailed
[04:35:27] WARN AuditUtilCheckAuditProgressNotFinished

yaml
Copy code

### Metrics observed:
- QuietDatabaseStartFail: 4  
- BuildTeamsLastBuildTeamsFailed: 2  
- AuditUtilCheckAuditProgressNotFinished: 3  

---

# How to Interpret the Evidence

### Pattern A: quiet database failures + team build failures
- Means configure workflow is unstable (Cluster 9)  
- Check next: audit progress, recruit storage  

### Pattern B: SlowSSLoop only
- Means storage pressure (Cluster 6)

**If one server impacted →** local issue  
**If all configure steps fail →** system-wide migration failure (this log)

---

# Root Cause Analysis

### 1. Quiet database orchestration failures  
- Why: configure repeatedly restarts quiet database  
- Confirm: QuietDatabaseStartFail, ConsistencyCheck errors  
- Tests: ConfigureStorageMigrationTest

### 2. Storage team rebuild failure  
- Why: BuildTeamsLastBuildTeamsFailed / RecruitStorageNotAvailable  
- Confirm: repeated team build failures  
- Tests: configure + relocation

### 3. Audit/migration stuck  
- Why: audit never finishes due to migration churn  
- Confirm: AuditUtilCheckAuditProgressNotFinished loops  
- Tests: storage migration stress

---

# Diagnostic Checklist

- [ ] QuietDatabaseStartFail / ConsistencyCheckStartFail  
- [ ] BuildTeamsLastBuildTeamsFailed  
- [ ] RecruitStorageNotAvailable / no_more_servers  
- [ ] AuditUtilCheckAuditProgressNotFinished  

---

# Related Problems

- Often alongside Cluster 6 (storage pressure) because migration runs under heavy load.  
- Can be confused with Cluster 0; difference: Cluster 9 has explicit quiet database/migration failures, not cold-start reboots.  
- Can cause Cluster 5 if migration keeps restarting TLogs.