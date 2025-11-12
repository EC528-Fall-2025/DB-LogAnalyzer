import duckdb
import numpy as np
from collections import defaultdict
from pathlib import Path
import sys

# ---------- utils ----------
def l2norm(X: np.ndarray) -> np.ndarray:
    """Normalize each vector to unit length."""
    return X / (np.linalg.norm(X, axis=-1, keepdims=True) + 1e-12)

def cosine_sim_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between two sets of L2-normalized vectors."""
    return A @ B.T

# ---------- centroid builder ----------
def build_centroids(train_emb: np.ndarray, train_labels: list[str]):
    """Compute mean vector (centroid) for each label."""
    labels = np.array(train_labels)
    by_label = defaultdict(list)
    for e, lab in zip(train_emb, labels):
        by_label[lab].append(e)

    centroid_ids, centroid_mat = [], []
    for lab, vecs in by_label.items():
        mu = np.mean(np.vstack(vecs), axis=0, keepdims=True)
        mu = l2norm(mu)[0]
        centroid_ids.append(lab)
        centroid_mat.append(mu)

    return np.array(centroid_ids), np.vstack(centroid_mat).astype("float32")

# ---------- nearest-cluster assigner ----------
class NearestClusterAssigner:
    def __init__(self, train_emb: np.ndarray, train_labels: list[str],
                 tau_centroid: float = 0.74, tau_knn: float = 0.70, k: int = 7):
        assert train_emb.ndim == 2, "Training embeddings must be 2D"
        self.train_emb = l2norm(train_emb.astype("float32"))
        self.train_labels = np.array(train_labels)
        self.tau_centroid = tau_centroid
        self.tau_knn = tau_knn
        self.k = k

        self.centroid_ids, self.centroid_mat = build_centroids(self.train_emb, train_labels)

    def _topk(self, sims: np.ndarray, k: int):
        """Return indices of top-k highest similarities."""
        k = min(k, sims.shape[0])
        if k == sims.shape[0]:
            return np.argsort(-sims)
        top = np.argpartition(-sims, k-1)[:k]
        return top[np.argsort(-sims[top])]

    def assign(self, e_new: np.ndarray):
        """Assign one embedding to the nearest cluster."""
        e = l2norm(e_new.astype("float32")[None, :])  # shape (1, d)

        # --- 1) nearest centroid ---
        sims_cent = cosine_sim_matrix(e, self.centroid_mat).ravel()
        j = int(np.argmax(sims_cent))
        best_label = self.centroid_ids[j]
        best_sim = float(sims_cent[j])

        if best_sim >= self.tau_centroid:
            return {"label": best_label, "confidence": best_sim,
                    "method": "centroid", "neighbors": []}

        # --- 2) fallback k-NN ---
        sims_knn = cosine_sim_matrix(e, self.train_emb).ravel()
        top_idx = self._topk(sims_knn, self.k)
        top_sims = sims_knn[top_idx]
        top_labels = self.train_labels[top_idx]

        weights = np.maximum(top_sims, 0.0) + 1e-6
        totals = defaultdict(float)
        for lab, w in zip(top_labels, weights):
            totals[lab] += w
        knn_label, knn_score = max(totals.items(), key=lambda kv: kv[1])

        if float(top_sims[0]) >= self.tau_knn:
            return {"label": knn_label, "confidence": float(top_sims[0]),
                    "method": "knn",
                    "neighbors": list(zip(map(str, top_labels.tolist()),
                                          map(float, top_sims.tolist())))}

        # --- 3) low similarity (new cluster) ---
        return {"label": "NEW_CLUSTER_CANDIDATE",
                "confidence": float(max(best_sim, float(top_sims[0]))),
                "method": "low_sim",
                "neighbors": list(zip(map(str, top_labels.tolist()),
                                      map(float, top_sims.tolist())))}

# ---------- main pipeline ----------
def main():
    # --- connect to DB (absolute path) ---
    DB_PATH = Path(__file__).resolve().parents[1] / "data" / "fdb_embeddings.duckdb"
    print(f"🔗 Connecting to: {DB_PATH}")

    if not DB_PATH.exists():
        sys.exit("❌ Database file not found. Run `python -m tests.generate` first.")

    con = duckdb.connect(str(DB_PATH))
    tables = [t[0] for t in con.execute("SHOW TABLES").fetchall()]
    print(f"📋 Tables found: {tables}")

    if "training_embeddings" not in tables or "input_embeddings" not in tables:
        sys.exit("❌ Missing required tables. Run `python -m tests.generate` first.")

    # --- load data ---
    df_train = con.execute("SELECT label, embedding FROM training_embeddings").df()
    train_labels = df_train["label"].to_numpy()
    train_emb = np.vstack(df_train["embedding"].to_numpy())

    df_new = con.execute("SELECT chunk_id, embedding FROM input_embeddings").df()

    # --- build assigner ---
    assigner = NearestClusterAssigner(train_emb, train_labels)

    # --- assign new chunks ---
    print("\n🔍 Cluster assignments:\n")
    for _, row in df_new.iterrows():
        e_new = np.array(row["embedding"])
        result = assigner.assign(e_new)
        print(f"Chunk {row['chunk_id']} → {result['label']} "
              f"(conf={result['confidence']:.3f}, method={result['method']})")

    con.close()

if __name__ == "__main__":
    main()
