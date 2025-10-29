
import json
from difflib import SequenceMatcher
import time
from google import genai
from openai import OpenAI
client = OpenAI()


data = json.load(open("/Users/mirayrdm/Documents/Courses/EC528/DB-LogAnalyzer/ai_analysis/recovery_dataset_prod_noreason.json"))

# Prepare model

from openai import OpenAI
client = OpenAI()

response = client.responses.create(
    model="gpt-4o-mini",
    input="Write a one-sentence bedtime story about a unicorn."
)

print(response.output_text)

def similarity(a, b):
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

results = []
correct = 0

for i, d in enumerate(data):
    context = d["context_no_reason"]  # first 8 lines
    true_reason = d["canonical_reason"].strip()

    prompt = f"""
    You are an expert in FoundationDB recovery analysis.

Given the following recovery log excerpt, classify the MOST LIKELY root cause into one and only one of these canonical labels:

- transaction_commit_stall
- disk_write_pressure
- master_role_failure
- configuration_missing
- tlog_failure
- storage_server_failure
- network_partition
- metadata_state_stall
- recruitment_delay
- unknown

Rules:
- Respond with ONLY the canonical label.
- Do not explain your reasoning.
- Do not output anything else.
LOG CONTEXT:
    {context}

    Reason:
    """
    print(prompt)
    try:

        resp = client.responses.create(
        model="gpt-4o-mini",
        input=f"{prompt}"
    )
    except Exception as e:
        print(f"Retrying item {i} after 2s due to: {e}")
        time.sleep(30)
        resp = client.responses.create(
        model="gpt-40-mini",
        input=f"{prompt}"
    )

    print(resp.output_text)
    pred = resp.output_text

    sim = similarity(pred, true_reason)
    is_correct = sim > 0.6  

    if is_correct:
        correct += 1

    results.append({
        "true": true_reason,
        "pred": pred,
        "sim": sim
    })

# Final metric
acc = correct / len(results)
print(f"Accuracy: {acc:.2%}")

# Save outputs
json.dump(results, open("gemini_eval_prod_results.json", "w"), indent=2)


