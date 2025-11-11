import json, re, argparse, sys

def load_runbook(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return json.load(f)

def read_text(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        raw = f.read()
    low = raw.lower()

    # Extract attribute values like Type="DiskMetrics"
    attr_vals = re.findall(r'(\w+)="([^"]+)"', low)
    attr_words = " ".join([v for (_k, v) in attr_vals])

    # Remove tags
    no_tags = re.sub(r"<[^>]+>", " ", low)

    # Combine and clean
    text = no_tags + " " + attr_words
    text = re.sub(r"\s+", " ", text)
    return text

def match_runbook(runbook, anomaly_text, debug=False):
    matches = []
    for entry in runbook:
        pats = [p.lower() for p in entry.get("patterns_any", [])]
        score = sum(1 for p in pats if p and p in anomaly_text)
        if score > 0:
            matches.append((entry, score))
        if debug:
            hits = [p for p in pats if p and p in anomaly_text]
            if hits:
                print(f"[DEBUG] Entry '{entry['id']}' hits: {hits}", file=sys.stderr)
    matches.sort(key=lambda x: x[1], reverse=True)
    return matches

def main():
    parser = argparse.ArgumentParser(description="Match anomalies to runbook entries.")
    parser.add_argument("--runbook", required=True, help="Path to runbook.json")
    parser.add_argument("--anomaly", required=True, help="Path to anomaly log file")
    parser.add_argument("--debug", action="store_true", help="Print debug info")
    args = parser.parse_args()

    runbook = load_runbook(args.runbook)
    text = read_text(args.anomaly)

    if args.debug:
        print(f"[DEBUG] Checking presence of common tokens in {args.anomaly}:", file=sys.stderr)
        for tok in ["diskmetrics", "chaosmetrics", "latencymetrics", "version lag", "tlog", "resolver"]:
            print(f"  - {tok}: {'YES' if tok in text else 'no'}", file=sys.stderr)

    matches = match_runbook(runbook, text, debug=args.debug)

    if not matches:
        print("⚠️  No runbook match found.")
    else:
        print(f"✅ Matches found for {args.anomaly}:\n")
        for entry, score in matches:
            print(f"🔹 {entry['name']} (score={score})")
            print(f"   Solution: {entry['solution']}")
            print(f"   Severity: {entry['severity']}")
            print(f"   Rationale: {entry['rationale']}\n")

if __name__ == "__main__":
    main()
