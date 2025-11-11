import json, re

data = json.load(open("ai_analysis/recovery_dataset_prod.json"))

for d in data:
    # Remove any CodeCoverage lines from context (defensive)
    d["context_no_reason"] = re.sub(
        r'<Event[^>]*Type="CodeCoverage"[^>]*>', '', d["context"]
    ).strip()

json.dump(
    data,
    open("ai_analysis/recovery_dataset_prod_noreason.json", "w"),
    indent=2,
)

print(f"✅ Created stripped dataset with {len(data)} entries")