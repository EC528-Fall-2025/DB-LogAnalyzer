import re
import json
import glob
import os

# Match each CodeCoverage <Event ... /> tag (self-closing)
COV_TAG = re.compile(r'(<Event[^>]*\bType="CodeCoverage"[^>]*/>)', re.IGNORECASE)

# Attribute extractors (order-agnostic)
ATTR = {
    "time": re.compile(r'\bTime="([^"]+)"'),
    "comment": re.compile(r'\bComment="([^"]+)"'),
    "severity": re.compile(r'\bSeverity="([^"]+)"'),
    "src_file": re.compile(r'\bFile="([^"]+)"'),
    "line": re.compile(r'\bLine="([^"]+)"'),
}

out_path_jsonl = "all_codecoverage.jsonl"
total = 0

with open(out_path_jsonl, "w") as out:
    for path in glob.glob("samples/*.xml"):
        filename = os.path.basename(path)
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            xml_text = f.read()

        # iterate over ALL CodeCoverage events
        for m in COV_TAG.finditer(xml_text):
            raw = m.group(1)

            # extract attributes independently
            def get(attr, cast=None):
                mm = ATTR[attr].search(raw)
                if not mm:
                    return None
                val = mm.group(1)
                if cast:
                    try:
                        return cast(val)
                    except Exception:
                        return None
                return val

            time_val = get("time", cast=float)
            comment  = get("comment")
            severity = get("severity")
            srcfile  = get("src_file")
            line     = get("line")

            rec = {
                "file": filename,
                "comment": comment if comment is not None else "unknown",
                "time": time_val,
                "raw": raw
            }
            # optionally include extra metadata
            if severity is not None:
                try: rec["severity"] = int(float(severity))
                except: rec["severity"] = None
            if srcfile is not None:
                rec["src_file"] = srcfile
            if line is not None:
                rec["src_line"] = line

            out.write(json.dumps(rec) + "\n")
            total += 1

print(f"✅ Extracted {total} CodeCoverage events → {out_path_jsonl}")