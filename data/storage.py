import json
from datetime import datetime
from pathlib import Path
import duckdb


def preprocess_json(in_path: str, out_path: str) -> None:
    """
    Reads NDJSON, parses DateTime (ISO8601 with trailing Z),
    and writes NDJSON with DateTimeParsed in 'YYYY-mm-dd HH:MM:SS'.
    If parsing fails, keeps the original string.
    """
    in_path = str(in_path)
    out_path = str(out_path)

    with open(in_path, encoding="utf-8") as f:
        lines = f.readlines()

    out_lines = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)

        if "DateTime" in obj:
            try:
                dt = datetime.strptime(obj["DateTime"], "%Y-%m-%dT%H:%M:%SZ")
                obj["DateTimeParsed"] = dt.strftime("%Y-%m-%d %H:%M:%S")
            except Exception:
                # fall back to whatever was provided
                obj["DateTimeParsed"] = obj.get("DateTime")

        out_lines.append(json.dumps(obj, ensure_ascii=False))

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(out_lines))


def init_db(path: str = "fdb_logs.duckdb"):
    """
    Opens/creates a DuckDB at `path` and applies schema.sql
    """
    db = duckdb.connect(path)
    with open("./data/schema.sql", "r", encoding="utf-8") as f:
        db.execute(f.read())
    return db


def load_logs(db, json_path: str) -> None:
    """
    Preprocesses NDJSON (adds DateTimeParsed) and ingests by:
      1) reading each line as text,
      2) casting to JSON in a single column `json`,
      3) extracting fields with json_* functions into `events`.
    This avoids auto schema inference problems for optional/missing fields.
    """
    in_p = Path(json_path)
    preprocessed_p = in_p.with_name(in_p.stem + "_preprocessed.json")
    preprocess_json(str(in_p), str(preprocessed_p))

    # Read each line as text, then CAST to JSON to get a single `json` column
    db.execute(
        """
        CREATE OR REPLACE TABLE _lines AS
        SELECT *
        FROM read_csv(
            ?,               -- path
            columns = {'line':'VARCHAR'},
            delim   = '\n',
            header  = false
        );
        """,
        [str(preprocessed_p)],
    )

    db.execute(
        """
        CREATE OR REPLACE TABLE _staging_raw AS
        SELECT CAST(line AS JSON) AS json
        FROM _lines;
        """
    )

    # Insert into events (everything pulled from the `json` column)
    db.execute(
        """
        INSERT INTO events (
            event_id, ts, severity, event, process, role, pid, machine_id, address,
            trace_file, src_line, raw_json, fields_json
        )
        SELECT
            row_number() OVER () AS event_id,
            TRY_CAST(json_extract_string(json, '$.DateTimeParsed') AS TIMESTAMP) AS ts,
            TRY_CAST(json_extract_string(json, '$.Severity')       AS INT)       AS severity,
            json_extract_string(json, '$.Type')                                  AS event,
            json_extract_string(json, '$.Process')                               AS process,
            json_extract_string(json, '$.Role')                                  AS role,
            TRY_CAST(json_extract_string(json, '$.PID')           AS INT)        AS pid,
            COALESCE(
              json_extract_string(json, '$.Machine'),
              json_extract_string(json, '$.MachineID')
            )                                                                     AS machine_id,
            json_extract_string(json, '$.Address')                                AS address,
            json_extract_string(json, '$.TraceFile')                              AS trace_file,
            TRY_CAST(json_extract_string(json, '$.SrcLine')      AS INT)         AS src_line,
            json                                                              AS raw_json,
            json                                                              AS fields_json
        FROM _staging_raw;
        """
    )

    # Optional: preview
    try:
        print(
            db.execute(
                "SELECT event_id, ts, severity, event, machine_id FROM events ORDER BY event_id DESC LIMIT 5"
            ).fetchdf()
        )
    except Exception:
        pass
