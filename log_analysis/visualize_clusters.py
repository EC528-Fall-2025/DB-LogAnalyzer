import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

# Load the labeled + clustered dataset
df = pd.read_csv("cluster_outputs/codecoverage_clusters_filtered.csv")

# Embeddings column is string → convert back to list of floats
df["embedding"] = df["embedding"].apply(eval).apply(np.array)

X = np.vstack(df["embedding"].to_numpy())
labels = df["cluster"].to_numpy()

# Reduce dimensionality to 2D for visualization
tsne = TSNE(n_components=2, init="pca", random_state=42, perplexity=30)
X_2d = tsne.fit_transform(X)

df["x"] = X_2d[:, 0]
df["y"] = X_2d[:, 1]

plt.figure(figsize=(12, 8))
sns.scatterplot(
    data=df,
    x="x",
    y="y",
    hue="cluster",
    palette="tab20",
    s=30,
    alpha=0.8,
    legend="full"
)
plt.title("CodeCoverage Clusters (t-SNE Projection)")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.tight_layout()
plt.show()