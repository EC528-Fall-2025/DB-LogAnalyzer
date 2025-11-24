import duckdb
import matplotlib.pyplot as plt

con = duckdb.connect("data/fdb_logs.duckdb")

df = con.execute("""
    SELECT event_id, durability_lag_s
    FROM events_wide
    WHERE durability_lag_s IS NOT NULL
    ORDER BY event_id
""").df()

plt.figure(figsize=(10, 4))
plt.plot(df["event_id"], df["durability_lag_s"])
plt.xlabel("Timestamp")
plt.ylabel("durability_lag_s")
plt.title("durability_lag_s Over Time")
plt.tight_layout()
plt.show()