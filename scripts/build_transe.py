import numpy as np
import pickle

input_path = "data/external/NELL_KG/entity2vec.TransE"
output_path = "data/external/NELL_KG/transe.pkl"

embeddings = []

with open(input_path, "r") as f:
    for line in f:
        parts = line.strip().split()

        # Skip bad lines
        if len(parts) < 50:
            continue

        vec = list(map(float, parts))
        embeddings.append(vec)

embeddings = np.array(embeddings, dtype=np.float32)

with open(output_path, "wb") as f:
    pickle.dump(embeddings, f)

print("shape:", embeddings.shape)