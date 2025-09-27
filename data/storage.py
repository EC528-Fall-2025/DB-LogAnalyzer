import duckdb
import json
import datetime
from datetime import datetime

def preprocess_json(in_path, out_path):
    with open(in_path) as f:
        lines = f.readlines()

    out_lines = []
    for line in lines:
        obj = json.loads(line)

        # Normalize DateTime if present
        if "DateTime" in obj:
            try:
                # Parse the ISO format with trailing Z (UTC)
                dt =  datetime.strptime(obj['DateTime'], "%Y-%m-%dT%H:%M:%SZ")
                # Store in normalized form without Z (DuckDB friendly)
                obj["DateTimeParsed"] = dt.strftime("%Y-%m-%d %H:%M:%S")
            except Exception as e:
                # if parsing fails, keep original
                obj["DateTimeParsed"] = obj["DateTime"]
        print(obj['DateTimeParsed'])
        out_lines.append(json.dumps(obj))

    with open(out_path, "w") as f:
        f.write("\n".join(out_lines))

# Example usage:
def init_db(path="fdb_logs.duckdb"):
    db = duckdb.connect(path)
    db.execute(open("./data/schema.sql").read())
    return db

    
def load_logs(db, json_path):
    # Preprocess first
    preprocessed = "./data/sample_log_preprocessed.json"
    preprocess_json(json_path, preprocessed)

    # Stage the logs (exploded JSON)
    db.execute("""
        CREATE OR REPLACE TABLE _staging_raw AS
        SELECT *
        FROM read_json_auto(?, format='newline_delimited');
    """, [preprocessed])

    # Insert into events
    db.execute("""
    INSERT INTO events
    SELECT
      row_number() OVER () AS event_id,
      CAST(json_extract_string(json, '$.DateTimeParsed') AS TIMESTAMP) AS ts,
      TRY_CAST(json_extract_string(json, '$.Severity') AS INT) AS severity,
      json_extract_string(json, '$.Type') AS event,
      json_extract_string(json, '$.Process') AS process,
      json_extract_string(json, '$.Role') AS role,
      TRY_CAST(json_extract_string(json, '$.PID') AS INT) AS pid,
      json_extract_string(json, '$.Machine') AS machine_id,
      json_extract_string(json, '$.Address') AS address,
      json_extract_string(json, '$.TraceFile') AS trace_file,
      TRY_CAST(json_extract_string(json, '$.SrcLine') AS INT) AS src_line,
      json AS raw_json,
      json AS fields_json,

    FROM _staging_raw
""")
        # ✅ Fetch and print the first 5 rows that were just added
    rows = db.execute("SELECT * FROM events ORDER BY event_id DESC LIMIT 5").fetchdf()
    print("✅ Recently inserted rows:")
    print(rows)