import torch
import torch.nn as nn

# Single GCN layer implementing graph convolution
class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()

        # Initializes linear transformation for node features
        self.linear = nn.Linear(
            in_features,
            out_features,
            bias=False
        )

    # Performs normalized graph convolution
    def forward(self, X, A):
        num_nodes = A.shape[0]

        I = torch.eye(num_nodes, device=A.device)
        A_hat = A + I
        degree = torch.sum(A_hat, dim=1)

        D_inv_sqrt = torch.diag(torch.pow(degree + 1e-12, -0.5))
        normalized_A = (D_inv_sqrt @ A_hat @ D_inv_sqrt)

        return normalized_A @ self.linear(X)