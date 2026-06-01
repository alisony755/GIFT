import torch
import torch.nn as nn
import torch.nn.functional as F

# Computes contrastive loss
class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, P_org, P_aug):
        P_org = F.normalize(P_org, dim=1)
        P_aug = F.normalize(P_aug, dim=1)

        P = torch.cat([P_org, P_aug], dim=0)

        batch_size = P.shape[0]
        N = batch_size // 2

        sim = torch.matmul(P, P.T)
        sim = sim / self.temperature

        mask = torch.eye(batch_size, device=P.device).bool()
        sim = sim.masked_fill(mask, -1e9)

        labels = torch.cat([
            torch.arange(N, 2 * N),
            torch.arange(0, N)
        ]).to(P.device)

        exp_sim = torch.exp(sim)

        denominator = exp_sim.sum(dim=1)

        positive_scores = exp_sim[
            torch.arange(batch_size),
            labels
        ]

        loss = -torch.log(positive_scores / denominator)

        return loss.mean()