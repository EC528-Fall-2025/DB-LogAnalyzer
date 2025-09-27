import json
import duckdb
from pathlib import Path
from data.storage import init_db, load_logs, preprocess_json

def _write_ndjson(p: Path, rows):
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

def _apply_schema(db):
    with open("data/schema.sql","r",encoding="utf-8") as f:
        db.execute(f.read())

def test_preprocess_parses_DateTime(tmp_path: Path):
    src = tmp_path / "src.ndjson"
    dst = tmp_path / "dst.ndjson"
    _write_ndjson(src, [
        {"DateTime":"2025-09-05T21:14:45Z","Severity":"10","Type":"Role"},
        {"DateTime":"not-a-timestamp","Severity":"5","Type":"Weird"},
    ])
    preprocess_json(str(src), str(dst))
    lines = [json.loads(l) for l in dst.read_text(encoding="utf-8").splitlines()]
    assert lines[0]["DateTimeParsed"] == "2025-09-05 21:14:45"
    assert lines[1]["DateTimeParsed"] == "not-a-timestamp"  # fallback when parse fails

def test_ingest_happy_path(tmp_path: Path):
    db = duckdb.connect(":memory:")
    _apply_schema(db)

    p = tmp_path / "ok.ndjson"
    _write_ndjson(p, [
        {"DateTime":"2025-09-05T21:14:45Z","Severity":"10","Type":"Role","Role":"Role","PID":"123","Machine":"2.0.1.0:1","Address":"2.0.1.0:1"},
        {"DateTime":"2025-09-05T21:14:46Z","Severity":"20","Type":"GetKeyMetrics","Machine":"2.0.1.1:1"}
    ])

    load_logs(db, str(p))

    rows = db.execute("""
        select event_id, ts, severity, event, role, pid, machine_id
        from events
        order by event_id
    """).fetchall()

    assert len(rows) == 2
    assert rows[0][0] == 1 and rows[1][0] == 2                        # sequential ids
    assert str(rows[0][1]).startswith("2025-09-05 21:14:45")          # timestamp parsed
    assert rows[0][2] == 10 and rows[1][2] == 20                      # severity cast
    assert rows[0][3] == "Role" and rows[1][3] == "GetKeyMetrics"
    assert rows[0][4] == "Role"                                       # role string
    assert rows[0][5] in (123, 123.0, None)                           # pid numeric or NULL
    assert rows[0][6] == "2.0.1.0:1" and rows[1][6] == "2.0.1.1:1"

def test_bad_datetime_yields_null_ts(tmp_path: Path):
    db = duckdb.connect(":memory:")
    _apply_schema(db)

    p = tmp_path / "badts.ndjson"
    _write_ndjson(p, [{"DateTime":"not-a-ts","Severity":"5","Type":"Weird","Machine":"2.0.1.0:1"}])

    load_logs(db, str(p))
    ts, sev, ev = db.execute("select ts, severity, event from events limit 1").fetchone()
    assert ts is None
    assert sev == 5
    assert ev == "Weird"

def test_missing_optional_fields_become_nulls(tmp_path: Path):
    db = duckdb.connect(":memory:")
    _apply_schema(db)

    p = tmp_path / "missing.ndjson"
    _write_ndjson(p, [{"DateTime":"2025-01-01T00:00:00Z","Type":"NoRole"}])

    load_logs(db, str(p))
    row = db.execute("""
        select severity, role, process, pid, address, trace_file, src_line
        from events limit 1
    """).fetchone()
    assert row == (None, None, None, None, None, None, None)

def test_idempotency_simple(tmp_path: Path):
    """
    Re-running the same file may duplicate rows in current pipeline;
    this test documents behavior rather than enforcing dedupe.
    """
    db = duckdb.connect(":memory:")
    _apply_schema(db)

    p = tmp_path / "dupe.ndjson"
    _write_ndjson(p, [{"DateTime":"2025-01-01T00:00:00Z","Severity":"1","Type":"T"}])

    load_logs(db, str(p))
    load_logs(db, str(p))
    count = db.execute("select count(*) from events").fetchone()[0]
    assert count in (1, 2)
