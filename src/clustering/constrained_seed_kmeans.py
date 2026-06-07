import numpy as np
import time

# Implements constrained seed k-means
# - Labeled samples (seeds) initialize centroids
# - Seed assignments remain fixed
# - Unlabeled samples are reassigned
class ConstrainedSeedKMeans:
    def __init__(self, num_clusters, max_iter=100, tol=1e-4):
        self.num_clusters = num_clusters
        self.max_iter = max_iter
        self.tol = tol

    def fit_predict(self, embeddings, seed_indices, seed_labels):
        print(f"  embeddings shape: {embeddings.shape}")
        print(f"  embeddings dtype: {embeddings.dtype}")
        print(f"  n_seeds: {len(seed_indices)}")
        print(f"  n_clusters: {self.num_clusters}")
        print(f"  diff array would be: ({len(embeddings) - len(seed_indices)}, {self.num_clusters}, {embeddings.shape[1]})")
        print(f"  estimated memory: {(len(embeddings) * self.num_clusters * embeddings.shape[1] * 4) / 1e9:.2f} GB")
        embeddings = np.asarray(embeddings, dtype=np.float32)
        n_samples = embeddings.shape[0]

        seed_labels = seed_labels.numpy() if hasattr(seed_labels, 'numpy') else np.asarray(seed_labels)

        labels = np.full(n_samples, -1, dtype=int)
        labels[seed_indices] = seed_labels

        # Initialize centroids from seed means
        centroids = []
        for cluster_id in range(self.num_clusters):
            mask = (seed_labels == cluster_id)
            selected = [seed_indices[i] for i in range(len(seed_indices)) if mask[i]]
            centroids.append(embeddings[selected].mean(axis=0))
        centroids = np.vstack(centroids)

        seed_set = np.array(seed_indices)
        unlabeled_mask = np.ones(n_samples, dtype=bool)
        unlabeled_mask[seed_set] = False
        unlabeled_idx = np.where(unlabeled_mask)[0]

        for i in range(self.max_iter):
            print(f"  k-means iter {i+1}")
            t0 = time.time()
            old_centroids = centroids.copy()

            X = embeddings[unlabeled_idx]
            X_sq = np.sum(X ** 2, axis=1, keepdims=True)
            C_sq = np.sum(centroids ** 2, axis=1, keepdims=True)
            cross = X @ centroids.T
            distances = np.sqrt(np.maximum(X_sq - 2 * cross + C_sq.T, 0))
            labels[unlabeled_idx] = np.argmin(distances, axis=1)
            print(f"  iter {i+1} distances: {time.time()-t0:.3f}s")

            t1 = time.time()
            for k in range(self.num_clusters):
                members = embeddings[labels == k]
                if len(members) > 0:
                    centroids[k] = members.mean(axis=0)
            print(f"  iter {i+1} centroid update: {time.time()-t1:.3f}s")

            shift = np.linalg.norm(centroids - old_centroids)
            print(f"  iter {i+1} shift: {shift:.6f}")

            if shift < self.tol:
                break

        return labels