import numpy as np
from graph.graph_data import GraphData
from graph.pmi import PMIBuilder

# Builds GIFT POS graph using one-hot features and PMI adjacency
# Produces G_p = {V_p, X_p, A_p}
class POSGraphBuilder:
    def __init__(self):
        self.pmi = PMIBuilder()

    def build(self, corpus_pos_tags):
        pos_nodes = sorted(
            {
                tag
                for doc in corpus_pos_tags
                for tag in doc
            }
        )

        node_to_idx = {
            tag: idx
            for idx, tag in enumerate(pos_nodes)
        }

        X_p = np.eye(len(pos_nodes), dtype=np.float32)
        A_p = self.pmi.build(corpus_pos_tags, pos_nodes)

        return GraphData(nodes=pos_nodes, X=X_p, A=A_p)