import json
import os
from collections import defaultdict

# Directory with chunks
chunk_dir = '/Users/mirayrdm/Documents/Courses/EC528/DB-LogAnalyzer/chunks_10s'
chunk_files = [f'chunk_{i}.json' for i in range(5)]

# Sentinel value for invalid metrics
INVALID = -1.75945e+09

anomalies = []

def check_anomaly(event, chunk_id):
    event_type = event['event']
    fields_str = event.get('fields_json', '{}')
    try:
        fields = json.loads(fields_str)
    except:
        fields = {}
    if event_type == 'ReadLatencyMetrics':
        min_val = float(fields.get('Min', 0))
        if min_val == INVALID:
            anomalies.append(f"Chunk {chunk_id}: Invalid read latencies (Min={min_val}) in {event_type}")
    elif event_type == 'StorageMetrics':
        bytes_input_str = fields.get('BytesInput', '0 0 0')
        bytes_durable_str = fields.get('BytesDurable', '0 0 0')
        if ' ' in bytes_input_str:
            bytes_input = float(bytes_input_str.split()[2])
            bytes_durable = float(bytes_durable_str.split()[2])
            if bytes_durable < bytes_input * 0.9:  # If durable is less than 90% of input, lag
                anomalies.append(f"Chunk {chunk_id}: Durability lag - BytesInput: {bytes_input}, BytesDurable: {bytes_durable}")
        wrong_shard_str = fields.get('WrongShardServer', '0')
        if ' ' in wrong_shard_str:
            wrong_shard = int(wrong_shard_str.split()[-1])
        else:
            wrong_shard = int(wrong_shard_str)
        if wrong_shard > 0:
            anomalies.append(f"Chunk {chunk_id}: WrongShardServer count: {wrong_shard}")
    elif event_type == 'ProxyMetrics':
        conflicts_str = fields.get('TxnConflicts', '0')
        if ' ' in conflicts_str:
            conflicts = int(conflicts_str.split()[-1])
        else:
            conflicts = int(conflicts_str)
        if conflicts > 0:
            anomalies.append(f"Chunk {chunk_id}: Transaction conflicts: {conflicts}")
    elif event_type == 'ChaosMetrics':
        delays = int(fields.get('DiskDelays', '0'))
        flips = int(fields.get('BitFlips', '0'))
        if delays > 0 or flips > 0:
            anomalies.append(f"Chunk {chunk_id}: Chaos events - DiskDelays: {delays}, BitFlips: {flips}")

for i, chunk_file in enumerate(chunk_files):
    path = os.path.join(chunk_dir, chunk_file)
    with open(path, 'r') as f:
        data = json.load(f)
        for event in data:
            check_anomaly(event, i)

# Print anomalies
for a in anomalies:
    print(a)