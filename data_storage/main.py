# Run with:  python -m data.main
from tools.storage import init_db, load_logs

def main():
    db = init_db("fdb_logs.duckdb")

    # check if events already has rows
    rows = db.execute("SELECT COUNT(*) FROM events").fetchone()[0]

    if rows == 0:
        print("Loading logs for the first time...")
        load_logs(db, "./data/sample_log.json")
    else:
        print("Events already loaded, skipping.")

if __name__ == "__main__":
    main()
