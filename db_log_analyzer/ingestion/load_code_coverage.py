import json
from pathlib import Path
from langchain_core.documents import Document   # for your installed version

def load_code_coverage_jsonl(path: str):
    """
    Handles:
      - JSON array
      - true JSONL
      - multi-line JSON objects
    """
    p = Path(path)
    text = p.read_text(encoding="utf-8").strip()

    # --- parse any of the 3 formats ---
    def _load_multiline_objects(t: str):
        entries, buf, depth, in_str, esc = [], [], 0, False, False
        def flush():
            nonlocal buf
            if buf:
                obj_txt = "".join(buf).strip()
                if obj_txt:
                    entries.append(json.loads(obj_txt))
                buf = []
        for ch in t:
            buf.append(ch)
            if in_str:
                if esc: esc = False
                elif ch == "\\": esc = True
                elif ch == '"': in_str = False
            else:
                if ch == '"': in_str = True
                elif ch == "{": depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        flush()
        return entries

    if not text:
        return []

    if text.startswith("["):
        entries = json.loads(text)
    else:
        try:
            entries = [json.loads(line) for line in text.splitlines() if line.strip()]
        except json.JSONDecodeError:
            entries = _load_multiline_objects(text)

    # --- build Documents ---
    docs = []
    for entry in entries:
        cluster = entry.get("cluster")
        label = entry.get("canonical_label", "")
        summary = entry.get("representative_comment", "")
        examples = entry.get("examples", [])

        text_block = (
            f"Cluster: {cluster}\n"
            f"Label: {label}\n"
            f"Summary: {summary}\n\n"
            "Examples:\n" + "\n".join(f"- {e}" for e in examples)
        )

        docs.append(
            Document(
                page_content=text_block,
                metadata={
                    "type": "code_coverage",
                    "cluster": cluster,
                    "label": label,
                    "summary": summary,              # ✅ add clean summary
                    "examples": examples,
                    "source_file": str(p),
                }
            )
        )
    return docs
