import torch
import torch.nn.functional as F

# Cross entropy loss
def cross_entropy(logits, labels):
    return F.cross_entropy(logits, labels)