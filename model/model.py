from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from .codebook.network3d import Network3D
from .dynamic_channel_adaptation import DynamicChannelAdaptation
from .spatial_channel_prompt import SpatialChannelPrompt


def _state_dict_from_checkpoint(checkpoint):
    if (
            isinstance(checkpoint, dict)
            and "model_state_dict" in checkpoint):
        return checkpoint["model_state_dict"]
    return checkpoint


class Model(nn.Module):
    """Three-stage unfolding model recorded in the retained experiment log."""

    def __init__(
            self,
            Ch=8,
            stages=3,
            nc=32,
            codebook_checkpoint_path=None):
        super().__init__()
        del Ch
        self.s = stages
        self.upMode = "bilinear"
        self.nc = nc

        self.H4 = DynamicChannelAdaptation(
            4, 1, kernel_size=7, embedding_dim=320,
            scale=1, is_transpose=False
        )
        self.HT4 = DynamicChannelAdaptation(
            1, 4, kernel_size=7, embedding_dim=320,
            scale=1, is_transpose=True
        )
        self.D4 = DynamicChannelAdaptation(
            4, 4, kernel_size=7, embedding_dim=320,
            scale=4, is_transpose=False
        )
        self.DT4 = DynamicChannelAdaptation(
            4, 4, kernel_size=7, embedding_dim=320,
            scale=4, is_transpose=True
        )
        self.H8 = DynamicChannelAdaptation(
            8, 1, kernel_size=7, embedding_dim=320,
            scale=1, is_transpose=False
        )
        self.HT8 = DynamicChannelAdaptation(
            1, 8, kernel_size=7, embedding_dim=320,
            scale=1, is_transpose=True
        )
        self.D8 = DynamicChannelAdaptation(
            8, 8, kernel_size=7, embedding_dim=320,
            scale=4, is_transpose=False
        )
        self.DT8 = DynamicChannelAdaptation(
            8, 8, kernel_size=7, embedding_dim=320,
            scale=4, is_transpose=True
        )

        self.proxNet = SpatialChannelPrompt(
            dim=24,
            num_blocks=[2, 3, 3, 4],
            num_refinement_blocks=2,
            heads=[1, 2, 4, 8],
            ffn_expansion_factor=2.66,
            bias=False,
            LayerNorm_type="WithBias",
            decoder=True,
        )
        self.proxNetCodeBook = Network3D()
        if codebook_checkpoint_path is not None:
            checkpoint = torch.load(
                Path(codebook_checkpoint_path), map_location="cpu"
            )
            self.proxNetCodeBook.load_state_dict(
                _state_dict_from_checkpoint(checkpoint), strict=True
            )

        self.alpha = Parameter(0.1 * torch.ones(self.s, 1))
        self.alpha_F = Parameter(0.1 * torch.ones(self.s, 1))
        self.alpha_K = Parameter(0.1 * torch.ones(self.s, 1))
        self._initialize_weights()
        nn.init.normal_(self.alpha, mean=0.1, std=0.01)
        nn.init.normal_(self.alpha_F, mean=0.1, std=0.01)
        nn.init.normal_(self.alpha_K, mean=0.1, std=0.01)

        # The first-stage prior is fixed while training the unfolding model.
        self.proxNetCodeBook.requires_grad_(False)

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.xavier_normal_(module.weight.data)
                if module.bias is not None:
                    nn.init.constant_(module.bias.data, 0.0)
            elif isinstance(module, nn.ConvTranspose2d):
                nn.init.xavier_normal_(module.weight.data)
                if module.bias is not None:
                    nn.init.constant_(module.bias.data, 0.0)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight.data)
                if module.bias is not None:
                    nn.init.constant_(module.bias.data, 0.0)

    def forward(self, ms, pan, one_hot):
        channels = ms.shape[1]
        if channels not in (4, 8):
            raise ValueError(
                f"Expected a 4- or 8-band MS image, got {channels}"
            )
        restored = F.interpolate(
            ms, scale_factor=4, mode=self.upMode
        )
        prior = restored.clone()

        for stage in range(self.s):
            if channels == 4:
                gradient = (
                    self.DT4(self.D4(restored, one_hot) - ms, one_hot)
                    + self.HT4(
                        self.H4(restored, one_hot) - pan, one_hot
                    )
                    + self.alpha[stage] * (restored - prior)
                )
            else:
                gradient = (
                    self.DT8(self.D8(restored, one_hot) - ms, one_hot)
                    + self.HT8(
                        self.H8(restored, one_hot) - pan, one_hot
                    )
                    + self.alpha[stage] * (restored - prior)
                )

            restored = self.proxNet(
                restored - self.alpha_F[stage] * gradient
            )
            prior_middle = prior - self.alpha_K[stage] * (
                self.alpha[stage] * (prior - restored)
            )
            with torch.no_grad():
                prior = self.proxNetCodeBook(
                    prior_middle, one_hot
                )[0]

        return restored
