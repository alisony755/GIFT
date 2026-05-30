import numpy as np

# Loads NELL TransE embeddings for entity graph node features
class TransELoader:
    def __init__(
        self,
        embedding_file,
        mapping_file
    ):
        self.embeddings = np.loadtxt(
            embedding_file,
            dtype=np.float32
        )

        self.entity_to_id = {}

        with open(mapping_file, "r") as f:
            for line in f:
                parts = line.strip().split()

                if len(parts) < 2:
                    continue

                entity = parts[0]
                idx = int(parts[1])

                self.entity_to_id[entity] = idx

        self.embedding_dim = self.embeddings.shape[1]

    def get_embedding(self, entity):
        idx = self.entity_to_id.get(entity)

        if idx is None:
            return np.zeros(
                self.embedding_dim,
                dtype=np.float32
            )

        return self.embeddings[idx]

    def build_feature_matrix(self, entities):
        return np.array(
            [self.get_embedding(entity) for entity in entities],
            dtype=np.float32
        )