import torch
import torch.nn as nn

from src.contrastive.projection_head import ProjectionHead
from src.contrastive.contrastive_loss import ContrastiveLoss
from src.contrastive.cluster_contrastive_loss import ClusterContrastiveLoss
from src.losses.cross_entropy_loss import cross_entropy
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
        zeta=0.5,
        batch_size=256,
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
        self.contrastive_loss_fn = ContrastiveLoss(temperature=temperature, batch_size=256)
        
        # L_ccl
        self.cluster_loss_fn = ClusterContrastiveLoss(temperature=temperature, batch_size=256)
        
        # L_ce
        self.cross_entropy = nn.CrossEntropyLoss()
            
    def forward(
        self,
        Z_org,
        Z_aug,
        cluster_labels,
        labeled_indices,
        true_labels
    ):
        # Eq. 6
        P_org = self.contrastive_projection(Z_org)
        P_aug = self.contrastive_projection(Z_aug)
        L_cl = self.contrastive_loss_fn(P_org, P_aug)
        print(f"  L_cl: {L_cl.item():.4f}")

        # Eq. 7
        Q = self.cluster_projection(Z_org)

        L_ccl = self.cluster_loss_fn(Q, cluster_labels)
        print(f"  L_ccl: {L_ccl.item():.4f}")

        # Eq. 8
        logits = self.classifier(Z_org)
        labeled_logits = logits[labeled_indices]
        labeled_true = true_labels[labeled_indices]
        L_ce = self.cross_entropy(labeled_logits, labeled_true)
        print(f"  L_ce: {L_ce.item():.4f}")

        # Eq. 9
        total_loss = (
            self.eta * L_cl
            + self.zeta * L_ccl
            + L_ce
        )
        print(f"  total_loss: {total_loss.item():.4f}")

        return {
            "loss": total_loss,
            "contrastive_loss": L_cl,
            "cluster_loss": L_ccl,
            "classification_loss": L_ce,
            "logits": logits
        }
        
    # Returns logits
    def classify(self, Z):
        return self.classifier(Z)