from dataclasses import dataclass
import numpy as np

# Stores graph nodes, features, and adjacency matrix
@dataclass
class GraphData:
    nodes: list
    X: np.ndarray
    A: np.ndarray