import numpy as np

from src.clustering.constrained_seed_kmeans import ConstrainedSeedKMeans
from src.text_representation.text_encoder import TextEncoder

# Run with python3 -m tests.test_full_pipeline

# Fake embeddings
Z_org = np.random.randn(6, 10)

seed_idx = np.array([0,1])
seed_labels = np.array([0,1])

kmeans = ConstrainedSeedKMeans(num_clusters=2)
weak = kmeans.fit_predict(Z_org, seed_idx, seed_labels)

print("weak labels:", weak)

encoder = TextEncoder()
Z_aug = np.random.randn(6, 10)

Z_final = encoder.build_final(Z_org, Z_aug)

print("Z_final:", Z_final.shape)