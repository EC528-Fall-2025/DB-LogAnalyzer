import pandas as pd
import sqlite3
import json
import glob

# Paths
parquet_dir = "./data"
sqlite_path = "fdb_logs.sqlite"

# Connect to SQLite
con = sqlite3.connect(sqlite_path)

# Loop over all parquet files in the directory
for parquet_file in glob.glob(f"{parquet_dir}/*.parquet"):
    table_name = parquet_file.split("/")[-1].replace(".parquet", "")
    print(f"📥 Loading {parquet_file} into table {table_name}")

    df = pd.read_parquet(parquet_file)

    # Convert dicts/lists into JSON strings so SQLite accepts them
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].apply(lambda x: json.dumps(x) if isinstance(x, (dict, list)) else x)

    # Write into SQLite
    df.to_sql(table_name, con, if_exists="replace", index=False)

con.close()
print(f"✅ Export complete! Open {sqlite_path} in DB Browser for SQLite")
