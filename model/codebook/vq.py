"""Minimal residual-vector-quantization base."""

import torch
import torch.nn as nn


class ResidualVectorQuantizer(nn.Module):
    """Minimal residual-VQ base used by the 3D block quantizer."""

    def __init__(
            self,
            n_e,
            e_dim,
            beta=0.25,
            LQ_stage=False,
            depth=6):
        super().__init__()
        self.n_e = int(n_e)
        self.e_dim = int(e_dim)
        self.LQ_stage = LQ_stage
        self.beta = beta
        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(
            -1.0 / self.n_e, 1.0 / self.n_e
        )
        self.depth = depth

    @staticmethod
    def dist(x, y):
        if x.shape == y.shape:
            return (x - y) ** 2
        return (
            torch.sum(x ** 2, dim=1, keepdim=True)
            + torch.sum(y ** 2, dim=1)
            - 2 * torch.matmul(x, y.t())
        )
