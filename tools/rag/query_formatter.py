# Converts detector results and timeline evidence into a RAG query

def build_rag_query(detectors: dict, timeline: dict = None, timeline_builder: dict = None):
    lines = ["Detected problems and evidence:"]

    for name, result in detectors.items():
        if result.get("detected"):
            lines.append(f"- {name}: {result}")

    if timeline:
        lines.append("\nTimeline highlights:")
        for key, value in timeline.items():
            lines.append(f"- {key}: {value}")

    if timeline_builder:
        lines.append("\nChronological story (timeline builder):")
        first_anomaly = timeline_builder.get("first_anomaly")
        if first_anomaly:
            lines.append(f"- First anomaly: {first_anomaly}")
        for item in timeline_builder.get("timeline", []):
            lines.append(f"- {item}")
        if timeline_builder.get("root_cause_signal"):
            lines.append(f"- Root cause signal: {timeline_builder.get('root_cause_signal')}")

    return "\n".join(lines)
