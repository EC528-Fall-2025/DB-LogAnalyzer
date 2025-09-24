# Database Log Analysis using LLMs

## Collaborators
---
| Name | Email |
|---|---|
| Miray Ayerdem | mirayrdm@bu.edu |
| Vanshika Chaddha| vchaddha@bu.edu |
| Fengxin Dai | fengx@bu.edu |
| Leo Phung| leophung@bu.edu |
| Sara Al Sowaimel | salsowa@bu.edu |

## 1.   Vision and Goals Of The Project:

### Goal
FoundationDB (FDB) is a distributed key-value database used in production by numerous companies, including Apple and Snowflake. As FDB runs, it generates logs, which are used for diagnostics. FDB logs display various aspects of runtime behavior, including ingress/egress counters, data movement progress, and errors. However, interpreting FDB’s logs is challenging due to their complexity, making manual analysis slow and error-prone.

This project aims to simplify and accelerate the process of analyzing FDB TraceEvent logs by developing a pipeline that parses raw logs into structured data, computes statistical rollups for anomaly detection, and integrates Large Language Models (LLMs) to provide guided diagnoses. By enabling operators to quickly detect anomalies, understand their causes, and take corrective action, this project will improve observability, reduce downtime, and enhance developer and operator productivity.

### Features
- Log Parsing: Convert raw FDB logs (TraceEvent) into a normalized schema with fields such as timestamp, severity, role, and event-specific metrics.
- Fast Querying: Store structured logs in DuckDB for efficient analytical queries and rollups.
- Anomaly Detection: Apply baseline statistical methods (e.g., EWMA, z-scores) over key metrics like QueueBytes, GrvLatency, and Transactions/sec to surface anomalies.
- LLM-Assisted Diagnosis: Use Retrieval-Augmented Generation (RAG) to combine logs with FDB documentation, code comments, and curated runbooks for guided problem diagnosis.
- Actionable Insights: Provide diagnoses with confidence scores, suggested next steps, and recommended validation commands.
- Extensibility: Establish a foundation for future integrations, such as dashboards, fine-tuned AI models, and more advanced anomaly detection.

**To Summarize:** By the end of the semester, the project will deliver a working CLI tool that converts FDB logs into structured data and demonstrates incident detection and explanation.

## 2. Users / Personas of the Project
This project is designed for Site Reliability Engineers (SREs), Database Operators, and new engineers working with FoundationDB. Their main challenge is diagnosing and resolving issues such as latency spikes, queue growth, or stalled transactions under time pressure. Today, this requires scanning large, complex log files and cross-referencing runbooks, which is slow and error-prone.

The system will make this process more efficient by converting logs into structured rollups, highlighting anomalies, and offering guided AI-assisted explanations. Typical questions these users need answered include:

- “Why is GRV latency spiking?”
- “Are tlogs spilling to disk?”
- “Is data movement stuck or thrashing?”
- “Which storage servers are hot or behind?”
- “Are exclusions or misconfigs hurting recruitment?”
- “Is workload imbalance causing saturation?”

These scenarios illustrate the core value of the system: helping users move from raw log scanning to fast, confident diagnosis and resolution.

**Site Reliability Engineer (SRE)**

- **Role:** Ensures uptime and performance of FoundationDB clusters, often under incident response conditions.
- **Key Characteristics:** Works under time pressure, must quickly identify root causes, relies on logs and runbooks for troubleshooting.
- **Goals:** Detect anomalies early, receive confident diagnoses, and reduce downtime through faster resolution.

**Database Operator**
- **Role:** Manages the daily operations of FoundationDB clusters, ensuring stable performance and smooth recovery from issues.
- **Key Characteristics:** Routinely checks logs, monitors system health, and applies fixes. Manual log analysis can be slow and overwhelming.
- **Goals:** Simplify log interpretation, automatically surface issues, and receive actionable next steps without deep log dives.

**Student / New Engineer**
- **Role:** A learner or junior engineer onboarding to FoundationDB, still developing expertise with TraceEvent logs.
- **Key Characteristics:** Finds raw logs complex and difficult to interpret without guidance. Depends on simplified explanations.
- **Goals:** Accelerate learning, understand system behavior, and connect log patterns to likely causes with AI-assisted summaries.

## 3.   Scope and Features Of The Project
### Log Ingestion & Normalization
- Parser supporting JSON trace format with regex fallback for plaintext logs.  
- Normalization into a typed schema with fields such as timestamp, severity, process, role, event, and event-specific metrics (e.g., `GrvLatency`, `QueueBytes`, `DurabilityLag`).  
- Storage in DuckDB/Parquet for efficient queries.  
- Windowed rollups (1s/10s/60s) for key counters to highlight trends and anomalies.  

### Knowledge Ingestion
- Selected FoundationDB documentation, TraceEvent code comments, and curated runbooks.  
- Chunking strategies: logs by time windows or anomaly slices; docs by semantic section.  
- Metadata tagging (role, subsystem, metric) for targeted retrieval.  
- *(Stretch)* Query planner mapping user questions to relevant logs, roles, metrics, and documents.  

### Anomaly Detection & Summarization
- Basic detectors (EWMA, z-score) for identifying unusual behavior.  
- Summaries per role and per time window, including cross-role correlations (e.g., *“TLog lag increases while data movement increases”*).  
- Fact table of extracted claims (e.g., *“DurabilityLag > 5s on 3 TLogs between 12:00–12:05”*).  
- *(Stretch)* Prompt caching and reusable checklists to improve LLM efficiency.  

### CLI Tools & Operator Workflow
- CLI commands for querying metrics through DuckDB.  
- LLM-proposed safe commands (`grep`, `jq`, `fdbcli`) with bounded outputs.  
- Step-by-step suggestions with diagnosis, confidence score, and next steps.  

### Training & Evaluation
- Rule-based heuristics for common fault modes (e.g., TLog spill, recruitment churn, hot shard).  
- Scenario bank with induced faults, stress tests, and normal operation logs.  
- Evaluation metrics: detection precision/recall, diagnosis quality, tool usefulness, efficiency.  
- *(Stretch)* Report cards with confusion matrices and ablations (e.g., RAG on/off, anomaly selection on/off).  
- *(Stretch)* Programmatic labeling and synthetic Q&A generation.  

### Stretch Features (Out-of-Scope)
- **Visualization & Dashboard:** Web-based or GUI dashboards (scope limited to CLI/TUI prototype).  
- **Advanced Analytics:** Dashboards for cost and throughput monitoring.  
- **Advanced ML & Scaling:** Large-scale pretraining or fine-tuning of LLMs beyond lightweight LoRA.  
- **Distributed Deployment:** Multi-node support or enterprise observability integration (e.g., Prometheus, Grafana).  
- **Security & Compliance:** Full security monitoring and compliance auditing.

## 4. Solution Concept


