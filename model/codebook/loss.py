"""Loss functions used while training the 3D codebook."""

import torch
import torch.nn as nn


class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-6, reduction="mean"):
        super().__init__()
        if reduction not in {"sum", "mean"}:
            raise ValueError("reduction must be 'sum' or 'mean'")
        self.eps = eps
        self.reduction = reduction

    def forward(self, prediction, target):
        loss = torch.sqrt((prediction - target) ** 2 + self.eps)
        if self.reduction == "sum":
            return torch.sum(loss)
        return torch.mean(loss)
