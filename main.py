from data.storage import init_db, load_into_db
def main():
    log_path = 'trace.0.0.0.0.288994.1758137995.CbOixg.0.9.xml'
    db = init_db("fdb_logs.duckdb")

    load_into_db(db, log_path)

if __name__ == "__main__":
    main()