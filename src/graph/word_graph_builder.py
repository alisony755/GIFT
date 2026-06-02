from src.graph.graph_data import GraphData
from src.graph.pmi import PMIBuilder
from src.embeddings.glove_loader import GloveLoader

# Builds GIFT word graph using GloVe features and PMI adjacency
# Produces G_w = {V_w, X_w, A_w}
class WordGraphBuilder:
    def __init__(self, glove_path):
        self.glove = GloveLoader(glove_path)
        self.pmi = PMIBuilder()

    def build(self, corpus_tokens):
        word_nodes = sorted(
            {
                token
                for doc in corpus_tokens
                for token in doc
            }
        )

        X_w = self.glove.build_feature_matrix(word_nodes)
        A_w = self.pmi.build(corpus_tokens, word_nodes)

        return GraphData(nodes=word_nodes, X=X_w, A=A_w)