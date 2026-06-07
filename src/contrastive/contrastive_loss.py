import torch
import torch.nn as nn
import torch.nn.functional as F

# Computes constrative loss
class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.5, batch_size=256):
        super().__init__()
        self.temperature = temperature
        self.batch_size = batch_size

    def forward(self, P_org, P_aug):
        N = P_org.shape[0]

        # Subsample if corpus is large
        if N > self.batch_size:
            idx = torch.randperm(N)[:self.batch_size]
            P_org = P_org[idx]
            P_aug = P_aug[idx]

        P_org = F.normalize(P_org, dim=1)
        P_aug = F.normalize(P_aug, dim=1)

        P = torch.cat([P_org, P_aug], dim=0)
        B = P_org.shape[0]
        batch_size = 2 * B

        sim = torch.matmul(P, P.T) / self.temperature

        # Mask self-similarity
        mask = torch.eye(batch_size, device=P.device).bool()
        sim = sim.masked_fill(mask, -1e9)

        # Positive pairs
        labels = torch.cat([
            torch.arange(B, 2 * B),
            torch.arange(0, B)
        ]).to(P.device)

        loss = F.cross_entropy(sim, labels)
        return loss