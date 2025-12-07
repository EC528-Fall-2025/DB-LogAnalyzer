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

## Sprint Demo Videos and Slides
- **Sprint 1 Demo Video:** [Link](https://drive.google.com/file/d/1bHu6nhIrEkouQ02xXVG0B5ajXGo_30Uo/view?usp=drive_link)  
- **Sprint 1 Demo Slides:** [Link](https://docs.google.com/presentation/d/13x_g204QpMCRAE1XFmgZDgtySpRFXZm0/edit?usp=sharing&ouid=113513264960850511829&rtpof=true&sd=true)
- **Sprint 2 Demo Video:** [Link](https://drive.google.com/file/d/1AAPKvnSOWdhe-z5LE-CpRRRTQ_wBUVqF/view?usp=drive_link)  
- **Sprint 2 Demo Slides:** [Link](https://docs.google.com/presentation/d/1Lw4QyjTKoIy6Q7o1-tgFQYmEFzu2VFNx/edit?usp=sharing&ouid=113513264960850511829&rtpof=true&sd=true)
- **Sprint 3 Demo Video:** [Link](https://drive.google.com/file/d/1vCRLP642GrFl5wB2i-5foWVBpMSJpdlw/view?usp=sharing)  
- **Sprint 3 Demo Slides:** [Link](https://docs.google.com/presentation/d/1cjRuqCkxPs6AHikMo5JrXXBsWedwZa64/edit?usp=sharing&ouid=113513264960850511829&rtpof=true&sd=true)
- **Sprint 4 Demo Video:** [Link](https://drive.google.com/file/d/1wKtvsR3i8nEnGut6-YjPbHMEYCRf6iN4/view?usp=sharing)  
- **Sprint 4 Demo Slides:** [Link](https://docs.google.com/presentation/d/1I2S8yGN1IRloVvZTsVIafV4YTwY8hIcz/edit?usp=sharing&ouid=113513264960850511829&rtpof=true&sd=true)
- **Sprint 5 Demo Video:** [Link](https://drive.google.com/file/d/19Kr-ciG37Z2RYUtSk78TWc8ZQ5ofjWtg/view?usp=sharing)  
- **Sprint 5 Demo Slides:** [Link](https://docs.google.com/presentation/d/1BBK2vaw_zYxLhVQ5w8xMsOllPMB-lKNy2bJqaRM2Vh4/edit?usp=sharing)
- **Sprint 6 Demo Video:** [Link]()  
- **Sprint 6 Demo Slides:** [Link](https://docs.google.com/presentation/d/1kQMmr0Hp2SKAnSIPVoThiDE941WNlLWNY_2IJh3QJoc/edit?usp=sharing)


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

The system will make this process more efficient by converting logs into structured rollups, highlighting anomalies, and offering guided AI-assisted explanations. Potential questions these users need answered include:

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

## Current Architecture
<img width="1422" height="673" alt="Screenshot 2025-09-23 at 9 37 39 PM" src="https://github.com/user-attachments/assets/b980b93e-79e0-48dd-b1d2-9980b52e8a36" />


## Global Architectural Structure Of the Project

1. **Log Generation & Conversion**  
   - Run FoundationDB to generate XML log files.  
   - Convert XML logs into JSON format.  

2. **Storage in Analytical Database**  
   - Upload JSON logs into DuckDB for efficient SQL-based analysis.  

3. **Querying & Investigation**  
   - Run SQL queries to explore event patterns (e.g., *“An error occurred at time X; what happened 5 seconds before?”*).  
   - Investigate anomalies through rollups and targeted queries.  

4. **AI Confidence Scoring**  
   - AI evaluates query results and provides a confidence score for the diagnosis.  
   - If the confidence score is below a threshold, the AI proposes additional queries for further investigation.  
   - This loop continues until the confidence score exceeds the threshold.  

5. **Diagnosis & Runbook Guidance**  
   - Once confidence is high, the system outputs a diagnosis (e.g., *“TLog crashed,” “data movement stalled,” “storage server saturated”*).  
   - Next-step actions are suggested based on curated runbooks.  

6. **Operator Workflow (CLI)**  
   - The entire process is accessible through a CLI (not copy-paste chat).  
   - Operators run queries, review AI suggestions, and validate results directly.  

### Additional Considerations  
- **Testing:** Use test servers, unit tests, and simulation tests to validate accuracy.  
- **Clues & Symptoms:** Operators may provide hints (e.g., *“memory is increasing”*) to guide AI query generation.  
- **Chunking:** Logs may be too large to process at once, so the system will prioritize recent or anomaly-rich chunks for analysis.  

## Design Implications and Discussion:
1.  **Schema Normalization**  
   - **Implication:** Standardizing log fields (timestamp, severity, role, event, metrics) allows consistent queries and downstream analysis.  
   - **Reason:** Without normalization, operators must repeatedly learn FDB’s complex log structure, slowing diagnosis.  
2.  **Windowed Rollups & Anomaly Detection**  
   - **Implication:** Summarizing metrics in 1s/10s/60s windows highlights trends without overwhelming the user with raw data.  
   - **Reason:** Operators need to see *when* anomalies begin, not just static values. Simple baselines (EWMA, z-score) provide fast, explainable results.
3.  **CLI Interface**  
   - **Implication:** Focusing on a command-line tool makes the prototype lightweight, reproducible, and usable in production environments where GUIs may not be practical.  
   - **Reason:** Most SREs and DB operators already rely on CLI tools. This reduces learning curve and integration friction.
4.  **Chunking Strategy for Logs**  
   - **Implication:** Logs are split into smaller windows (time-based or anomaly slices) for efficient retrieval and processing.  
   - **Reason:** Prevents token overflows with LLMs and ensures operators focus on the most relevant data slices.
5. **ML Confidence Scoring and Feedback Loop**  
  - **Implication:** Introducing a confidence threshold formalizes the uncertainty of ML-assisted diagnosis. Below-threshold results trigger recursive querying, which increases latency but improves accuracy.  
  - **Reason:** In high-pressure environments, false positives slow down rather than help. A confidence threshold allows the system to refine results until they are trustworthy enough for operators to act on.

## 5. Acceptance criteria
At the end of this semester, the project will be considered successful if it meets the following minimum expectations and stretch/demo goals:  

### Minimum Expectations  
- **CLI Tool:** Parses raw FDB logs into a DuckDB database.  
- **Structured Storage:** Database contains structured `events` and `metrics` tables with rollups at 1s / 10s / 60s windows.  
- **Query Support:** Operators can run SQL queries on key metrics (e.g., latency, durability lag, queue size).  
- **Documentation:** README with setup instructions, CLI usage examples, and an architecture diagram.  

### End-of-Semester Demo Goals  
- **Operator Questions:** System can answer at least **10 curated operator questions**.  
  - Each answer should include:  
    - Citations (log snippets or metrics).  
    - Runnable tool commands (SQL queries, `grep`, or `jq`).  
- **Failure Mode Diagnosis:** Correctly diagnose at least **3 distinct failure modes** (e.g., TLog spill, recruitment churn, data movement stuck).  
  - Each diagnosis should achieve:  
    - ≥ 0.7 confidence score.  
    - ≥ 0.9 coverage@10 (system surfaces the correct log/doc chunk in the top 10 results).
    
### Stretch Goals  
- **Prompt Caching & Summarization:** Reuse structured prompts and generate role-based summaries to reduce LLM overhead.
- **Scenario Bank & Evaluation:** Build a dataset of induced fault scenarios with ground-truth labels; evaluate detection precision/recall, diagnosis quality, and efficiency. 
- **Visualization:** Charts or dashboards for anomaly visualization (beyond CLI scope for MVP).
- **Programmatic Labeling & Synthetic Q&A:** Use rule-based heuristics to auto-label log windows with scenarios (e.g., TLog Spill, Recruitment Churn). Generate synthetic Q&A pairs (including “bad answer → correction” examples) to improve LLM accuracy. 

## 6. Release Planning
The project will be delivered incrementally in phases, each building on the previous and providing usable functionality along the way. This phased approach will ensure early value delivery, reduces risk, allows iterative improvement based on feedback.

### Iteration 1 (Weeks 1–3): Log Ingestion & Normalization
- Logs normalized into a typed schema (timestamp, severity, process, role, event, metrics).
- Metrics stored in DuckDB/Parquet with 1s/10s/60s rollups.
- **User Story:** As an operator, I want raw logs converted into structured tables so I can query performance trends.
- **Deliverable:** CLI tool to parse FoundationDB logs (JSON trace format + regex fallback for plaintext).
### Iteration 2 (Weeks 3–5): Knowledge Ingestion
- Semantic chunking and metadata tagging for efficient retrieval.
- Early query planner mapping user questions to relevant roles/metrics/docs.
- **User Story:** As an operator, I want logs connected with documentation and runbooks so I can understand not only symptoms but also causes.
- **Deliverable:** Knowledge base built from FoundationDB docs, TraceEvent code comments, and runbooks.
### Iteration 3 (Weeks 5–7): Anomaly Detection & Summarization
- Summaries generated per role and time window, with cross-role correlations.
- Fact table storing extracted claims (e.g., “DurabilityLag >5s on 3 TLogs”).
- **User Story:** As an operator, I want anomalies automatically flagged and explained so I can quickly prioritize issues.
- **Deliverable:** Baseline anomaly detectors (EWMA, z-score) applied to log metrics.
### Iteration 4 (Weeks 7–9): Exposing Tools
- Operator workflow: diagnosis, confidence score, and recommended next steps.
- **User Story:** As an operator, I want an interface that suggests commands and provides structured results so I can troubleshoot faster.
- **Deliverable:** CLI tools for querying metrics (DuckDB SQL) and proposing safe commands (grep/jq/fdbcli).
### Iteration 5 (Weeks 9–12): Training & Evaluation
- Evaluation harness measuring precision/recall, diagnosis quality, and tool usefulness.
- Final demo: answer 10 curated operator questions and correctly diagnose 3+ failure modes with ≥0.7 confidence.
- **User Story:** As a team, we want evaluation metrics and incident replays so we can validate that the system works in real-world-like conditions.
- **Deliverable:** Rule-based heuristics for common fault modes, scenario bank with induced/stress logs.


** **
