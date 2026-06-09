import torch
import torch.nn as nn
from .gcn_layer import GCNLayer

# Two-layer GCN used by GIFT for each component graph
# Encode graphs with GNNs, using node features themselves and relationships between nodes to learn better embeddings
# Produces H_w, H_e, H_p
class GCNEncoder(nn.Module):
    def __init__(self,input_dim, hidden_dim=128, dropout=0.9):
        super().__init__()

        self.gcn1 = GCNLayer(input_dim, hidden_dim)
        self.gcn2 = GCNLayer(hidden_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, X, A):
        H = self.gcn1(X, A)
        H = self.activation(H)
        H = self.dropout(H) # Apply dropout between layers
        H = self.gcn2(H, A)
        H = self.activation(H)

        return H