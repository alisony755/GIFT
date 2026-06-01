import torch
import torch.nn as nn
import torch.nn.functional as F

# Implements cluster contrastive loss
class ClusterContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings, labels):
        embeddings = F.normalize(embeddings, dim=1)

        sim = torch.matmul(embeddings, embeddings.T)
        sim = sim / self.temperature

        batch_size = embeddings.size(0)

        eye_mask = torch.eye(
            batch_size,
            device=embeddings.device
        ).bool()

        exp_sim = torch.exp(sim)
        exp_sim = exp_sim.masked_fill(eye_mask, 0.0)

        losses = []

        for i in range(batch_size):
            positive_mask = (labels == labels[i])
            positive_mask[i] = False
            positive_indices = torch.where(positive_mask)[0]

            if len(positive_indices) == 0:
                continue

            denominator = exp_sim[i].sum()
            sample_losses = []

            for j in positive_indices:
                numerator = exp_sim[i, j]
                
                sample_loss = -torch.log(numerator / denominator)
                sample_losses.append(sample_loss)

            loss_i = torch.stack(sample_losses).mean()
            losses.append(loss_i)

        if len(losses) == 0:
            return torch.tensor(0.0, device=embeddings.device)

        return torch.stack(losses).mean()