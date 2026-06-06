import numpy as np

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
        embeddings = np.asarray(embeddings)
        n_samples = embeddings.shape[0]

        labels = np.full(n_samples, -1, dtype=int)
        labels[seed_indices] = seed_labels

        centroids = []

        for cluster_id in range(self.num_clusters):
            mask = (seed_labels == cluster_id)

            selected_indices = [
                seed_indices[i]
                for i in range(len(seed_indices))
                if mask[i]
            ]

            members = embeddings[selected_indices]

            centroid = members.mean(axis=0)
            centroids.append(centroid)

        centroids = np.vstack(centroids)
        seed_set = set(seed_indices)

        for _ in range(self.max_iter):
            old_centroids = (centroids.copy())

            for i in range(n_samples):
                if i in seed_set:
                    continue

                distances = np.linalg.norm(
                    embeddings[i]
                    - centroids,
                    axis=1
                )

                labels[i] = np.argmin(distances)

            for k in range(self.num_clusters):
                members = embeddings[labels == k]

                if len(members) > 0:
                    centroids[k] = (members.mean(axis=0))

            shift = np.linalg.norm(centroids - old_centroids)

            if shift < self.tol:
                break

        return labels