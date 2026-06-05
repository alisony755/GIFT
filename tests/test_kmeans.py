import numpy as np
from src.clustering.constrained_seed_kmeans import ConstrainedSeedKMeans

# Run with python3 -m tests.test_kmeans

X = np.random.randn(6, 3)

seed_idx = np.array([0, 1])
seed_labels = np.array([0, 1])

kmeans = ConstrainedSeedKMeans(num_clusters=2)

labels = kmeans.fit_predict(X, seed_idx, seed_labels)

print("labels:", labels)