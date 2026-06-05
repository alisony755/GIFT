import torch
from src.text_representation.gcn_encoder import GCNEncoder

# Run with python3 -m tests.test_gcn

X = torch.randn(3, 16)
A = torch.eye(3)

model = GCNEncoder(input_dim=16, hidden_dim=8)

H = model(X, A)

print("H shape:", H.shape)