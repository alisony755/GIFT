import torch
import torch.nn as nn

# Implements GIFT classifier Υ(Z_org)
class Classifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.network(x)