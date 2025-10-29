import glob
import json

def is_codecov(line):
    return 'Type="CodeCoverage"' in line

OUT = "ai_analysis/codecoverage_extracted.jsonl"

def main():
    out = open(OUT, "a")

    for path in glob.glob("ai_analysis/samples/*.xml"):
        print(f"Processing {path}")
        lines = open(path).read().splitlines()

        cleaned = []
        for line in lines:
            if is_codecov(line):
                # store the event with the filename
                out.write(json.dumps({
                    "file": path.split("/")[-1],
                    "raw": line
                }) + "\n")
            else:
                cleaned.append(line)

        # write cleaned version
        clean_path = path.replace(".xml", "_clean.xml")
        with open(clean_path, "w") as f:
            f.write("\n".join(cleaned))

    out.close()
    print("✅ Done! Extracted CodeCoverage events are in:", OUT)

if __name__ == "__main__":
    main()