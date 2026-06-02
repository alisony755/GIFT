import numpy as np
from src.graph.graph_data import GraphData
from src.embeddings.transe_loader import TransELoader

# Builds GIFT entity graph using NELL TransE embeddings and cosine similarity
# Produces G_e = {V_e, X_e, A_e}
class EntityGraphBuilder:
    def __init__(self, transe_path, mapping_path):
        self.transe = TransELoader(transe_path, mapping_path)

    def build(self, corpus_entities):
        entity_nodes = sorted(
            {
                entity
                for doc in corpus_entities
                for entity in doc
            }
        )

        X_e = self.transe.build_feature_matrix(entity_nodes)
        norms = np.linalg.norm(X_e, axis=1, keepdims=True)

        normalized = X_e / (norms + 1e-12)
        A_e = normalized @ normalized.T
        A_e = np.maximum(A_e, 0)
        np.fill_diagonal(A_e, 0)

        return GraphData(nodes=entity_nodes, X=X_e, A=A_e)