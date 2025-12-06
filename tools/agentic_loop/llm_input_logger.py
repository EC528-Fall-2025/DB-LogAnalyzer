"""LLM input logger to persist prompt text to disk."""

from datetime import datetime
from pathlib import Path


def write_llm_input(prompt_text: str, output_dir: str = "data", prefix: str = "llm_input"):
    """Write the full LLM prompt text to a timestamped file."""
    if not prompt_text:
        return None
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    path = Path(output_dir) / f"{prefix}_{ts}.txt"
    with open(path, "w") as f:
        f.write(prompt_text)
    return str(path)


def write_llm_output(output_text: str, output_dir: str = "data", prefix: str = "llm_output"):
    """Write the LLM output JSON/text to a timestamped file."""
    if not output_text:
        return None
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    path = Path(output_dir) / f"{prefix}_{ts}.txt"
    with open(path, "w") as f:
        f.write(output_text)
    return str(path)
