"""
service/chunker.py

Implements multiple chunking strategies for logs:
- TimeChunker: split by time windows
- RoleChunker: split by process role
- SemanticChunker: cluster by text similarity (optional)
- HybridChunker: combine time + semantic grouping
"""

from collections import defaultdict
import pandas as pd
from math import floor
import tiktoken

# Optional: for semantic chunking
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.cluster import KMeans
except ImportError:
    SentenceTransformer = None
    KMeans = None


class TimeChunker:
    """Split events into time-based chunks"""
    def __init__(self, interval_seconds=300):
        self.interval = interval_seconds
        self.encoder = tiktoken.encoding_for_model("gpt-4")

    def chunk(self, events):
        # Convert list of Event objects → DataFrame
        df = pd.DataFrame([{"ts": e.ts, "event": e.event} for e in events if e.ts])
        if df.empty:
            return []
        start_ts = df['ts'].min()

        def round_ts(ts):
            delta = (ts - start_ts).total_seconds()
            bucket = floor(delta / self.interval)
            return start_ts + pd.Timedelta(seconds=bucket * self.interval)

        df['chunk_time'] = df['ts'].apply(round_ts)
        grouped = df.groupby('chunk_time')

        chunks = []
        for ts, group in grouped:
            text = "\n".join(group['event'].astype(str).tolist())
            tokens = len(self.encoder.encode(text))
            chunks.append(Chunk(ts, text, tokens))
        return chunks


class RoleChunker:
    """Group logs by their process role (e.g., storage, tlog, proxy)"""
    def __init__(self):
        self.encoder = tiktoken.encoding_for_model("gpt-4")

    def chunk(self, events):
        grouped = defaultdict(list)
        for e in events:
            grouped[e.role].append(e)
        chunks = []
        for role, group in grouped.items():
            text = "\n".join([g.event for g in group])
            tokens = len(self.encoder.encode(text))
            chunks.append(Chunk(role, text, tokens))
        return chunks


class SemanticChunker:
    """Cluster events by semantic similarity using embeddings"""
    def __init__(self, n_clusters=10):
        if SentenceTransformer is None:
            raise ImportError("Install sentence-transformers for SemanticChunker.")
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.n_clusters = n_clusters
        self.encoder = tiktoken.encoding_for_model("gpt-4")

    def chunk(self, events):
        texts = [e.event for e in events]
        embeddings = self.model.encode(texts)
        clusters = KMeans(n_clusters=self.n_clusters).fit_predict(embeddings)

        df = pd.DataFrame({"cluster": clusters, "text": texts})
        chunks = []
        for cid, group in df.groupby("cluster"):
            text = "\n".join(group["text"])
            tokens = len(self.encoder.encode(text))
            chunks.append(Chunk(cid, text, tokens))
        return chunks


class HybridChunker:
    """Combine time-based and role-based grouping."""
    def __init__(self, interval_seconds=300):
        self.interval = interval_seconds
        self.encoder = tiktoken.encoding_for_model("gpt-4")

    def chunk(self, events):
        import pandas as pd
        from math import floor

        # Convert to DataFrame
        df = pd.DataFrame([
            {"ts": e.ts, "event": e.event, "role": getattr(e, "role", "Unknown")}
            for e in events if e.ts
        ])
        if df.empty:
            return []

        start_ts = df["ts"].min()

        def round_ts(ts):
            delta = (ts - start_ts).total_seconds()
            bucket = floor(delta / self.interval)
            return start_ts + pd.Timedelta(seconds=bucket * self.interval)

        # Add chunk time
        df["chunk_time"] = df["ts"].apply(round_ts)

        # Group by both time window and role
        grouped = df.groupby(["chunk_time", "role"])

        chunks = []
        for (chunk_time, role), group in grouped:
            text = "\n".join(group["event"].astype(str).tolist())
            tokens = len(self.encoder.encode(text))
            label = f"{role} @ {chunk_time}"
            chunks.append(Chunk(label, text, tokens))

        return chunks



class Chunk:
    """Represents one chunk of log text."""
    def __init__(self, label, text, estimated_tokens):
        self.label = label               # could be timestamp, role, or cluster ID
        self.text = text                 # the joined log lines
        self.estimated_tokens = estimated_tokens
