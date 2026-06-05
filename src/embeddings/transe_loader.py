import pickle
import numpy as np


# Loads NELL TransE embeddings for entity graph node features
class TransELoader:

    # Load entity embeddings and entity-to-id mapping
    def __init__(self, embedding_file, mapping_file):

        # Load TransE embedding matrix
        with open(embedding_file, "rb") as f:
            self.embeddings = pickle.load(f)

        # Load entity -> embedding index mapping
        with open(mapping_file, "rb") as f:
            self.entity2id = pickle.load(f)

        # Store embedding dimensionality
        self.embedding_dim = self.embeddings.shape[1]

    # Retrieve embedding vector for a single entity
    def get_embedding(self, entity):

        # Normalize entity text
        entity = entity.lower()

        # Return 0 vector if entity is unknown
        if entity not in self.entity2id:
            return np.zeros(
                self.embedding_dim,
                dtype=np.float32
            )

        # Get embedding row index
        idx = self.entity2id[entity]

        # Return 0 vector if index is invalid
        if idx >= len(self.embeddings):
            return np.zeros(
                self.embedding_dim,
                dtype=np.float32
            )

        # Return TransE embedding vector
        return self.embeddings[idx]

    # Build feature matrix for a list of entities
    def build_feature_matrix(self, entities):

        # Stack entity embeddings into matrix
        return np.array(
            [self.get_embedding(entity) for entity in entities],
            dtype=np.float32
        )