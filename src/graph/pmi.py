import numpy as np
from collections import Counter
from itertools import combinations

# Computes PMI adjacency matrix for words or POS tags
class PMIBuilder:
    def build(self, documents, nodes):
        node_to_idx = {
            node: idx
            for idx, node in enumerate(nodes)
        }

        node_count = Counter()
        pair_count = Counter()

        total_docs = len(documents)

        window_size = 5

        for doc_nodes in documents:
            for i in range(len(doc_nodes)):
                window = doc_nodes[i:i + window_size]
                unique_nodes = set(window)
                for node in unique_nodes:
                    node_count[node] += 1
                for n1, n2 in combinations(sorted(unique_nodes), 2):
                    pair_count[(n1, n2)] += 1

                A = np.zeros(
                    (len(nodes), len(nodes)),
                    dtype=np.float32
                )

        for (n1, n2), cooccur in pair_count.items():
            p_i = node_count[n1] / total_docs
            p_j = node_count[n2] / total_docs
            p_ij = cooccur / total_docs

            pmi = np.log(
                p_ij / (p_i * p_j + 1e-12)
            )

            pmi = max(pmi, 0)

            i = node_to_idx[n1]
            j = node_to_idx[n2]

            A[i, j] = pmi
            A[j, i] = pmi

        return A