# cluster_codecoverage.py
import json, re, math
from pathlib import Path
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import MiniBatchKMeans
import matplotlib.pyplot as plt
import umap

# Optional HDBSCAN
try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except Exception:
    HDBSCAN_AVAILABLE = False

IN_PATH = Path("all_codecoverage.jsonl")
OUT_DIR = Path("cluster_outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Load dataset
rows = []
with open(IN_PATH) as f:
    for line in f:
        line = line.strip()
        if line:
            rows.append(json.loads(line))
df = pd.DataFrame(rows)

# Clean Text
df["comment"] = df.get("comment", "").fillna("").astype(str)
df["comment_lc"] = (
    df["comment"]
    .str.lower()
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
)
RECOVERY_WHITELIST = re.compile(
 r"\b(commit|transaction|tlog|spill|io_error|workerfailed|sharedtlogfailed|"
 r"accepting_commits|writing_coordinated_state|recruit|resolver|grv|proxy|"
 r"tag throttle|configuration_never_created|recovery|too_old|unknown_result)\b",
 re.I)

df = df[df["comment_lc"].str.contains(RECOVERY_WHITELIST, na=False)]
df = df[df["comment_lc"].str.len() >= 10].drop_duplicates("comment_lc")

noise_patterns = re.compile(
    r"\b(?:simulation start|start simulation|begin test|health monitor|"
    r"knob|force reuse tenant id prefix|"
    r"location_cache_failed_endpoint_retry_interval|"
    r"blobstore_max_delay_connection_failed|"
    r"cache_refresh_interval_when_all_alternatives_failed|"
    r"mark_failed_unstable_connections|"
    r"dbinfo_failed_delay|"
    r"priority_team_failed)\b",
    re.I
)

df["is_noise_candidate"] = df["comment_lc"].str.contains(noise_patterns, na=False)
df = df[~df["is_noise_candidate"]].copy()
# Embedding Model
print("Encoding sentences...")
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(df["comment_lc"].tolist(), show_progress_bar=True, normalize_embeddings=True)
X_all = np.array(embeddings)

# ✅ Remove invalid (zero/NaN) embeddings
norms = np.linalg.norm(X_all, axis=1)
valid_mask = (norms > 1e-6) & np.isfinite(norms)
removed = (~valid_mask).sum()
print(f"Removed {removed} invalid embeddings")

df = df[valid_mask].reset_index(drop=True)
X_all = X_all[valid_mask]

def cluster_hdbscan_or_kmeans(X, n_fallback=15, seed=42):
    labels = None
    algo = None

    if HDBSCAN_AVAILABLE:
        try:
            mcs = max(5, min(20, int(len(X) * 0.01)))
            clusterer = hdbscan.HDBSCAN(min_cluster_size=mcs, metric='euclidean')
            labels = clusterer.fit_predict(X)
            if (labels >= 0).sum() > 0:
                algo = f"HDBSCAN(min_cluster_size={mcs})"
        except:
            labels = None

    if labels is None or (labels >= 0).sum() == 0:
        k = max(4, min(n_fallback, int(math.sqrt(len(X)))))
        km = MiniBatchKMeans(n_clusters=k, random_state=seed, n_init=10, batch_size=256)
        labels = km.fit_predict(X)
        algo = f"MiniBatchKMeans(k={k})"

    return labels, algo

labels_all, algo_all = cluster_hdbscan_or_kmeans(X_all)
df["cluster"] = labels_all

# ---- UMAP 2D projection + plot ----
print("Computing UMAP (cosine)…")
reducer = umap.UMAP(n_neighbors=30, min_dist=0.0, metric="cosine", random_state=42)
X_2d = reducer.fit_transform(X_all)

df_umap = pd.DataFrame({
    "x": X_2d[:, 0],
    "y": X_2d[:, 1],
    "cluster": labels_all,
    "comment": df["comment"].astype(str)
})

# Save a CSV for inspection
(OUT_DIR / "viz").mkdir(exist_ok=True, parents=True)
df_umap.to_csv(OUT_DIR / "viz" / "umap_points.csv", index=False)

# Save a static PNG
plt.figure(figsize=(10, 8))
plt.scatter(df_umap["x"], df_umap["y"], c=df_umap["cluster"], s=10, alpha=0.8, cmap="Spectral")
plt.title("CodeCoverage clusters (UMAP, cosine)")
plt.xlabel("UMAP-1"); plt.ylabel("UMAP-2")
plt.colorbar(label="cluster")
plt.tight_layout()
plt.savefig(OUT_DIR / "viz" / "umap_clusters.png", dpi=300)
plt.close()

print(f"UMAP saved: {OUT_DIR / 'viz' / 'umap_clusters.png'}")
print(f"Points CSV: {OUT_DIR / 'viz' / 'umap_points.csv'}")

# ✅ Representative selection using cosine similarity
def get_cluster_representatives(X, labels, min_cluster_size=3):
    reps = {}
    for c in sorted(set(labels)):
        idx = np.where(labels == c)[0]
        if len(idx) < min_cluster_size:
            continue

        sub = X[idx]
        centroid = np.mean(sub, axis=0)

        if np.linalg.norm(centroid) < 1e-6:
            continue

        # Safe distance normalization
        sims = np.dot(sub, centroid) / (
            np.linalg.norm(sub, axis=1) * np.linalg.norm(centroid)
        )
        sims = np.nan_to_num(sims, nan=-1.0, posinf=-1.0, neginf=-1.0)
        rep_idx = idx[int(np.argmax(sims))]
        reps[c] = rep_idx
    return reps

reps = get_cluster_representatives(X_all, labels_all)

# ✅ Create cluster summary
summary_rows = []
for cluster in sorted(set(labels_all)):
    idx = np.where(labels_all == cluster)[0]
    if len(idx) == 0:
        continue

    rep_idx = reps.get(cluster)
    rep_comment = df.iloc[rep_idx]["comment"] if rep_idx is not None else ""

    summary_rows.append({
        "cluster": int(cluster),
        "size": len(idx),
        "sample_comment": rep_comment[:200]
    })

summary_df = pd.DataFrame(summary_rows).sort_values("cluster")

# ✅ Save Outputs
summary_df.to_csv(OUT_DIR / "codecoverage_clusters_summary.csv", index=False)
df.to_csv(OUT_DIR / "all_codecoverage_clustered.csv", index=False)

print("✅ DONE")
print("Clustering Algo:", algo_all)
print("Outputs saved in:")
print(" - cluster_outputs/codecoverage_clusters_summary.csv")
print(" - cluster_outputs/all_codecoverage_clustered.csv")