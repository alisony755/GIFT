import torch
import torch.nn as nn

from src.contrastive.projection_head import ProjectionHead
from src.contrastive.contrastive_loss import ContrastiveLoss
from src.contrastive.cluster_contrastive_loss import ClusterContrastiveLoss
from src.model.classifier import Classifier

# Full GIFT objective
class GIFTModel(nn.Module):
    def __init__(
        self,
        input_dim,
        num_classes,
        hidden_dim=256,
        projection_dim=128,
        temperature=0.5,
        eta=0.5,
        zeta=0.5
    ):
        super().__init__()

        self.eta = eta
        self.zeta = zeta

        # Projection head (Φ)
        self.contrastive_projection = (
            ProjectionHead(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=projection_dim
            )
        )

        # Projection head (Ψ)
        self.cluster_projection = (
            ProjectionHead(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=projection_dim
            )
        )

        # Υ
        self.classifier = Classifier(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes
        )

        # L_cl
        self.contrastive_loss = (ContrastiveLoss(temperature=temperature))
        
        # L_ccl
        self.cluster_loss = (ClusterContrastiveLoss(temperature=temperature))
        
        # L_ce
        self.cross_entropy = (nn.CrossEntropyLoss())

    def forward(
        self,
        Z_org,
        Z_aug,
        cluster_labels,
        labeled_indices,
        true_labels
    ):
    
        # Eq. 6
        Z_cl = torch.cat([Z_org, Z_aug], dim=0)
        P = self.contrastive_projection(Z_cl)
        L_cl = self.contrastive_loss(P)

        # Eq. 7
        Q = self.cluster_projection(Z_org)
        L_ccl = self.cluster_loss(Q, cluster_labels)

        # Eq. 8
        logits = self.classifier(Z_org)
        labeled_logits = logits[labeled_indices]
        L_ce = self.cross_entropy(labeled_logits, true_labels)

        # Eq. 9
        total_loss = (
            self.eta * L_cl
            + self.zeta * L_ccl
            + L_ce
        )

        return {
            "loss": total_loss,
            "contrastive_loss": L_cl,
            "cluster_loss": L_ccl,
            "classification_loss": L_ce,
            "logits": logits
        }