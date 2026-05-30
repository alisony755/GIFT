import pickle
import numpy as np

# Loads pretrained GloVe embeddings for word graph node features
class GloveLoader:
    def __init__(self, embedding_path):
        with open(embedding_path, "rb") as f:
            self.embeddings = pickle.load(f)

        sample_vector = next(
            iter(self.embeddings.values())
        )

        self.embedding_dim = len(sample_vector)

    def get_embedding(self, word):
        return self.embeddings.get(
            word,
            np.zeros(
                self.embedding_dim,
                dtype=np.float32
            )
        )

    def build_feature_matrix(self, words):
        return np.array(
            [self.get_embedding(word) for word in words],
            dtype=np.float32
        )