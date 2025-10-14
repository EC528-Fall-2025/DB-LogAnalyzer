"""
Experiment A: Chunk Size Effect Demo for FoundationDB Logs (DuckDB version)
Usage:
    python experiment_a_demo.py fdb_logs.duckdb
"""
"""
Experiment A: Chunk Size Effect Demo for FoundationDB Logs (DuckDB version)
Usage:
    python experiment_a_demo.py fdb_logs.duckdb
"""
import sys
import duckdb
import json
from datetime import datetime, timedelta

def export_windows(db_path, anomaly_ts, windows, machine=None):
    con = duckdb.connect(db_path)

    for w in windows:
        start_ts = anomaly_ts - timedelta(seconds=w)
        end_ts = anomaly_ts
        basefile = f"window_{w}s"
        csv_outfile = f"{basefile}.csv"
        json_outfile = f"{basefile}.json"

        print("\n" + "=" * 80)
        print(f"🪶 Exporting {w}s window → {csv_outfile} / {json_outfile}")
        print(f"   Time range: {start_ts} → {end_ts}")

        # --- Build query with timestamp casting ---
        query = f"""
            SELECT ts, event, role, machine_id, severity
            FROM events
            WHERE CAST(ts AS TIMESTAMP)
                  BETWEEN '{start_ts}' AND '{end_ts}'
        """
        if machine:
            query += f" AND machine_id = '{machine}'"

        # Debug
        print("   Running query:")
        print(query.strip())

        # --- Fetch results ---
        result = con.execute(query)
        rows = result.fetchall()
        cols = [d[0] for d in result.description]

        if not rows:
            print("⚠️  No rows found in this time window.")
            continue

        # --- Export to CSV ---
        con.execute(f"COPY ({query}) TO '{csv_outfile}' (HEADER, DELIMITER ',');")
        print(f"✅  Exported {len(rows)} rows to {csv_outfile}")

        # --- Export to JSON ---
        data = [dict(zip(cols, row)) for row in rows]
        with open(json_outfile, "w") as f:
            json.dump(data, f, indent=2, default=str)
        print(f"✅  Exported {len(rows)} rows to {json_outfile}")

    con.close()
    print("\n✅ All windows exported successfully!")


def main():
    if len(sys.argv) < 2:
        print("Usage: python experiment_a_demo.py <db_path>")
        sys.exit(1)

    db_path = sys.argv[1]

    # --- Configuration for your anomaly ---
    anomaly_ts = datetime(2025, 10, 3, 1, 22, 15)
    machine = None   # set to "2.0.1.0:1" if you want to filter
    window_sizes = [10, 20, 30, 60]

    print("=" * 80)
    print("EXPERIMENT A: Manual Chunking Around Anomaly")
    print("=" * 80)
    print(f"Database: {db_path}")
    print(f"Anomaly Timestamp: {anomaly_ts}")
    print(f"Machine Filter: {machine}")
    print(f"Window Sizes: {window_sizes} seconds")
    print("=" * 80)

    export_windows(db_path, anomaly_ts, window_sizes, machine)


if __name__ == "__main__":
    main()
