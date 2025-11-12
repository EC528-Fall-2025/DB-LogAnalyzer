from tabulate import tabulate

# =======================
# Prompts (from Mira)
# =======================
prompts = {
    "Q1": "Provide a concise response, limited to three sentences, regarding the GRVProxyMetrics event and its distinction from CommitLatencyMetrics.",
    "Q2": "Do you see any anomalies in these FDB logs? We're particularly interested in these events: 'StorageMetrics', 'DiskMetrics', 'GRVProxyMetrics', 'UpdateLatencyMetrics', 'ReadLatencyMetrics', 'CommitLatencyMetrics', 'GetValueMetrics'.",
    "Q3": """Each detected event lists reasons for anomaly.
- threshold_violation_<metric> means the metric exceeded a set system threshold.
- z_score_anomaly_<metric> means the metric’s recent value deviated significantly from its average.
- extreme_value_<metric> means an unusually high value beyond the 99.9th percentile.
Explain what might be causing repeated threshold_violation_QueryQueue and z_score_anomaly_VersionLag anomalies in the StorageMetrics events in three sentences.""",
    "Q4": "Using the FoundationDB docs and trace logs, explain if DiskMetrics is related to very high VersionLag and FetchExecutingMS in ReadLatencyMetrics in three sentences."
}

models = ["GooseAI", "Tabby"]
scores = []

for model in models:
    print(f"\n====== Scoring {model} ======\n")
    for key, question in prompts.items():
        print(f"🔹 Model: {model} | Prompt {key}")
        print("Prompt:")
        print(question)
        print()
        print("👉 Paste or summarize the model's answer (press Enter on a blank line to finish):")
        lines = []
        while True:
            line = input()
            if not line.strip():
                break
            lines.append(line)
        answer = " ".join(lines)

        print("\nNow rate (1–5):")
        det = int(input("Detection: "))
        rea = int(input("Reasoning: "))
        dep = int(input("Depth: "))
        cla = int(input("Clarity: "))
        scores.append([model, key, det, rea, dep, cla])
        print("✅ Recorded!\n")

print("\n=== Scoring Summary ===")
print(tabulate(scores, headers=["Model", "Prompt", "Detection", "Reasoning", "Depth", "Clarity"]))
