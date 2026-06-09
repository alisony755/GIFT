import torch
import torch.nn as nn
import torch.nn.functional as F

# Implements cluster contrastive loss
class ClusterContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.5, batch_size=256):
        super().__init__()
        self.temperature = temperature
        self.batch_size = batch_size

    def forward(self, embeddings, labels):
        N = embeddings.shape[0]

        # Subsample if corpus is large
        if N > self.batch_size:
            idx = torch.randperm(N)[:self.batch_size]
            embeddings = embeddings[idx]
            labels = labels[idx]
            
        print(f"  CCL labels unique: {labels.unique()}")

        embeddings = F.normalize(embeddings, dim=1)
        B = embeddings.shape[0]

        sim = torch.matmul(embeddings, embeddings.T) / self.temperature  

        eye_mask = torch.eye(B, device=embeddings.device).bool()
        exp_sim = torch.exp(sim).masked_fill(eye_mask, 0.0)

        denominator = exp_sim.sum(dim=1, keepdim=True)

        # Vectorized positive mask — no Python loop
        labels = labels.to(embeddings.device)
        positive_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)) 
        positive_mask = positive_mask & ~eye_mask # Exclude self

        # Count positives per sample
        n_positives = positive_mask.sum(dim=1).float() 

        # Numerator for all positives at once
        log_prob = torch.log((exp_sim / denominator).clamp(min=1e-9))
        
        # Mean over positives per sample
        loss_per_sample = -(log_prob * positive_mask).sum(dim=1) / n_positives.clamp(min=1)

        # Only average over samples that have at least one positive
        valid = n_positives > 0
        
        if valid.sum() == 0:
            return torch.tensor(0.0, device=embeddings.device)

        return loss_per_sample[valid].mean()