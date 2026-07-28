import torch
import torch.nn as nn

from .msab3d import MSAB3D
from .vq3d import BlockBasedResidualVectorQuantizer3D


CHANNELS = {16: 64, 32: 32, 64: 16}


class BasicBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks=6):
        super().__init__()
        self.block = nn.Sequential(
            MSAB3D(
                dim=out_channels,
                num_blocks=num_blocks,
                dim_head=out_channels // 4,
                heads=4,
            )
        )
        self.in_ch = in_channels
        self.out_ch = out_channels

    def forward(self, x):
        return self.block(x)


class VQModule(nn.Module):
    def __init__(
            self,
            n_e_shared=1024,
            n_e_task=256,
            e_dim_shared=256,
            depth=6,
            num_tasks=4):
        super().__init__()
        del n_e_shared, n_e_task, e_dim_shared, num_tasks
        self.quantize = BlockBasedResidualVectorQuantizer3D(
            n_shared=1024,
            n_task=256,
            e_dim=256,
            beta=0.25,
            LQ_stage=False,
            depth=depth,
            unfold_size=2,
            mlp_codebook=False,
        )

    def forward(self, x, one_hot):
        return self.quantize(x, one_hot)


class Network3D(nn.Module):
    """3D encoder, residual codebook and decoder used in both stages."""

    def __init__(self, n_e=1024, depth=6, num_block=(1, 1, 1)):
        super().__init__()
        del n_e
        current_resolution = max(CHANNELS)
        self.in_ch = 1
        self.out_ch = 1
        self.feature_channel = min(CHANNELS)
        self.feature_depths = len(CHANNELS)

        self.conv_in = nn.Conv3d(
            1, CHANNELS[current_resolution], 3, padding=1
        )
        self.encoder_conv1 = nn.Conv3d(
            CHANNELS[current_resolution],
            CHANNELS[current_resolution],
            3,
            padding=1,
        )
        self.encoder_256 = BasicBlock3D(
            CHANNELS[current_resolution],
            CHANNELS[current_resolution],
            num_blocks=num_block[0],
        )

        self.down1 = nn.Conv3d(
            CHANNELS[current_resolution],
            CHANNELS[current_resolution // 2],
            3,
            stride=(1, 2, 2),
            padding=1,
        )
        current_resolution //= 2
        self.encoder_conv2 = nn.Conv3d(
            CHANNELS[current_resolution],
            CHANNELS[current_resolution],
            3,
            padding=1,
        )
        self.encoder_128 = BasicBlock3D(
            CHANNELS[current_resolution],
            CHANNELS[current_resolution],
            num_blocks=num_block[1],
        )

        self.down2 = nn.Conv3d(
            CHANNELS[current_resolution],
            CHANNELS[current_resolution // 2],
            3,
            stride=(1, 2, 2),
            padding=1,
        )
        current_resolution //= 2
        self.encoder_conv3 = nn.Conv3d(
            CHANNELS[current_resolution],
            CHANNELS[current_resolution],
            3,
            padding=1,
        )
        self.encoder_64 = BasicBlock3D(
            CHANNELS[current_resolution],
            CHANNELS[current_resolution],
            num_blocks=num_block[2],
        )

        self.vq_64 = VQModule(depth=depth)
        self.decoder_conv1 = nn.Conv3d(
            CHANNELS[current_resolution],
            CHANNELS[current_resolution],
            3,
            padding=1,
        )
        self.decoder_64 = BasicBlock3D(
            CHANNELS[current_resolution],
            CHANNELS[current_resolution],
            num_blocks=num_block[2],
        )
        self.up2 = nn.Upsample(scale_factor=(1, 2, 2))
        current_resolution *= 2
        self.decoder_conv2 = nn.Conv3d(
            CHANNELS[current_resolution // 2],
            CHANNELS[current_resolution],
            3,
            padding=1,
        )
        self.decoder_128 = BasicBlock3D(
            CHANNELS[current_resolution],
            CHANNELS[current_resolution],
            num_blocks=num_block[1],
        )
        self.up3 = nn.Upsample(scale_factor=(1, 2, 2))
        current_resolution *= 2
        self.decoder_conv3 = nn.Conv3d(
            CHANNELS[current_resolution // 2],
            CHANNELS[current_resolution],
            3,
            padding=1,
        )
        self.decoder_256 = BasicBlock3D(
            CHANNELS[current_resolution],
            CHANNELS[current_resolution],
            num_blocks=num_block[0],
        )
        self.conv_out = nn.Conv3d(
            CHANNELS[current_resolution], 1, 3, padding=1
        )

    def encode(self, x):
        x = self.conv_in(x)
        feature1 = self.encoder_256(self.encoder_conv1(x))
        feature2 = self.encoder_128(
            self.encoder_conv2(self.down1(feature1))
        )
        feature3 = self.encoder_64(
            self.encoder_conv3(self.down2(feature2))
        )
        return feature1, feature2, feature3

    def decode(self, quantized):
        middle3 = self.decoder_conv1(quantized)
        decoded3 = self.decoder_64(middle3)
        middle2 = self.decoder_conv2(self.up2(decoded3))
        decoded2 = self.decoder_128(middle2)
        middle1 = self.decoder_conv3(self.up3(decoded2))
        decoded1 = self.decoder_256(middle1)
        return decoded1, middle1, middle2, middle3

    def forward(self, x, one_hot):
        x = x.unsqueeze(1)
        feature1, feature2, feature3 = self.encode(x)
        quantized, codebook_loss = self.vq_64(feature3, one_hot)
        decoded1, middle1, middle2, middle3 = self.decode(quantized)
        reconstructed = self.conv_out(decoded1).squeeze(1)
        return (
            reconstructed,
            codebook_loss,
            [feature1, feature2, feature3],
            [middle1, middle2, middle3],
        )
