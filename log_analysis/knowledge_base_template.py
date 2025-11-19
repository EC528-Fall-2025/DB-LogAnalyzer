KNOWLEDGE_BASE = """
# FoundationDB Recovery Cluster Knowledge Base

## CLUSTER 0: recovery_restart_cascade
Problem: Recovery loops, repeatedly restarting before completion due to coordinator state conflicts.
Causes: Concurrent recovery attempts, coordinator generation mismatches, commit_unknown_result ambiguity, incomplete previous recovery leaving stale metadata, clock skew causing epoch conflicts.
Key Indicators: Repeated "reading_coordinated_state" retries, coordinator mismatch errors, generation conflicts, "Aborting current recovery" loops.

## CLUSTER 4: transaction_tag_throttling
Problem: Tag throttling active during recovery blocks GRV requests, stalling recovery progress.
Causes: Tag budget exhausted (especially 'system' tag), rate limits too restrictive for recovery burst, leftover throttle state from previous load, GRV queue buildup (>10K entries).
Key Indicators: TAG_THROTTLED responses, tag budget exhausted, high GRV queue lengths, GRV timeouts (>5s), recovery slowed by throttling.

## CLUSTER 5: tlog_failure_recovery
Problem: TLog failure/degradation forces recovery due to inability to persist commits safely.
Causes: TLog crash, disk full, severe I/O degradation (>500ms latency), memory overflow, network partition isolating TLog, partial commits (only some TLogs acked).
Key Indicators: TLog failed/degraded messages, high disk latency, partial commit warnings, TLog excluded, "stopped by CC recovery", acknowledge timeouts.

## CLUSTER 6: storage_engine_pressure
Problem: Post-recovery memory/disk pressure reduces throughput significantly.
Causes: Recovery-triggered read surge, memory exceeding limits, disk write buffer overflow, DiskQueue durability lag, switching to large transaction mode, catch-up phase overload, I/O saturation (>95%).
Key Indicators: "KeyValueStoreMemory switching to large txn", memory >100%, DiskQueue not durable, high fsync latency (>1000ms), throughput reduction (50-70%), occurring AFTER recovery completion.

## CLUSTER 7: commit_proxy_pipeline_crash
Problem: Commit proxy or resolver crash breaks pipeline; master terminates, forcing recovery.
Causes: Process crash (segfault, OOM), deadlock/hang, connection loss between proxy-resolver, progress timeout, state machine error, bug in commit logic.
Key Indicators: Proxy/resolver crash or unresponsive (>30s), "failed to progress", pipeline broken, master terminated due to proxy failure, ClusterController detects master failure.

## CLUSTER 8: network_partition_recovery
Problem: Network partition/connectivity loss isolates components, prevents quorum, causes split-brain.
Causes: Switch/router failure, sustained packet loss (>10%), firewall misconfiguration, DNS failures, network congestion, asymmetric failures, coordinator quorum lost.
Key Indicators: Coordinator unreachable, connection timeouts/failures, quorum lost, network unreachable, split-brain detected, multiple components reporting communication failures simultaneously.

## CLUSTER 9: configuration_change_recovery
Problem: Configuration change (add/remove nodes, replication changes) triggers instability requiring recovery.
Causes: Exclude/include operations during recovery, replication factor changes, partial config updates, version mismatches during rolling upgrade, locality conflicts, coordinator set changes.
Key Indicators: Exclude/include operations logged, configuration change in progress, redundancy mode change, version mismatch, data movement during recovery, coordinator set updates.

## CLUSTER 10: clock_skew_recovery
Problem: Clock differences between nodes cause version ordering violations, lease confusion, timeout inconsistencies.
Causes: NTP failure/misconfiguration, leap second handling differences, VM time sync issues, manual clock adjustments, backward time jumps, clock drift exceeding tolerance (>1s).
Key Indicators: Clock skew/drift detected, NTP sync failures, "time jumped backward", version ordering violations, lease expiration anomalies, timestamps out of order, timing inconsistencies.

## CLUSTER 11: data_corruption_recovery
Problem: Data corruption detected in storage/logs/metadata forces recovery to restore from replicas.
Causes: Bit rot (silent data corruption), filesystem corruption, memory errors (bad RAM), storage engine bugs, incomplete writes from power failure, checksum mismatches, TLog/coordinator state corruption.
Key Indicators: Checksum mismatch, CRC errors, corruption detected, validation failures, filesystem errors, TLog replay errors, rebuilding/discarding corrupted data messages.

## CLUSTER 12: resource_exhaustion_recovery
Problem: OS-level resource exhaustion (file descriptors, ports, threads) prevents normal operation.
Causes: File descriptor limit hit (ulimit), ephemeral port exhaustion (all 64K ports used), thread creation failures, connection pool exhausted, cgroup limits, kernel OOM, socket buffer exhaustion.
Key Indicators: "Too many open files", "Cannot allocate memory" (system level), port exhaustion, thread creation failed, connection refused (resource limit), ENOMEM/EMFILE/EAGAIN errors.

## CLUSTER 13: workload_spike_recovery
Problem: Sudden workload surge overwhelms cluster, causing queue overflows and timeout cascades.
Causes: Retry storm (thundering herd), bulk data operations, badly behaving client, cache warming read amplification, seasonal traffic spike, background job overload, cascading timeouts.
Key Indicators: Traffic surge/spike, high load warnings, queue overflow, commit/read rate spikes, retry storm detected, timeout cascades, rejecting requests due to overload, sudden metric degradation.

## CLUSTER 14: upgrade_rollback_recovery
Problem: Software upgrade fails, introduces bugs, or creates version incompatibilities requiring rollback.
Causes: Protocol version incompatibility, data format changes, new version bugs, incomplete/mixed version upgrade, rollback incompatibility, feature flag conflicts, performance regression, strict validation rejecting old data.
Key Indicators: Version mismatch, protocol incompatibility, upgrade/rollback in progress, mixed versions detected, data format migration errors, validation failures on legacy data.

## CLUSTER 15: cascading_failure_recovery
Problem: One component failure triggers chain reaction of failures across cluster, causing widespread instability.
Causes: Storage server failure redistributing load overwhelming others, coordinator failure causing reconnection storm, shared dependency failure (DNS, NTP), bug affecting all processes, resource exhaustion spreading, network affecting multiple machines.
Key Indicators: Multiple different components failing rapidly in succession, load redistribution failures, timeout cascades across components, simultaneous failures on different machines, avalanche pattern (1→2→4 failures).

## CLUSTER 16: lease_expiration_recovery
Problem: Master lease expires or renewal fails, requiring new master election and recovery.
Causes: Network issues preventing renewal, clock skew making lease appear expired, master paused too long (GC, scheduling), coordinator quorum unavailable, master too busy to renew, deadlock preventing renewal thread.
Key Indicators: Lease expiration warnings, failed to renew lease, master lease loss, coordinator unavailable during renewal, master election due to lease loss, pause/GC events exceeding lease timeout.

## QUICK REFERENCE
Temporal Patterns:
- Before recovery: Clusters 5,7,8,13,15
- During recovery: Clusters 0,4,9,10
- After recovery: Cluster 6
- During operations: Clusters 9,14,16

Component Focus:
- Coordinator: 0,8,10,16
- TLog: 5,11
- Proxy: 4,7
- Storage: 6,11
- System-wide: 8,12,13,15
"""

prompt = f""" 
    You are an expert FoundationDB system debugger. Analyze the following log sequence that led to a MasterRecoveryState event and identify the root cause. 
    {KNOWLEDGE_BASE} 
    ===== CURRENT INCIDENT LOGS ===== 
    {format_logs(log_sequence)} 

    Based on the log patterns and similar incidents, determine: 
    1. Which cluster best matches this incident 
    2. The specific root cause within that cluster 
    3. Recommended fix or mitigation

    Format your response as: 
    CLUSTER: [number]
    ROOT_CAUSE: [specific cause]
    REASONING: [explain the evidence from logs] 
    FIX: [recommended action] """
