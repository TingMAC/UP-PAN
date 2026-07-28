"""Spatial/spectral prompt proximal network used by the main model."""

import numbers

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def to_3d(x):
    return rearrange(x, "b c h w -> b (h w) c")


def to_4d(x, height, width):
    return rearrange(
        x, "b (h w) c -> b c h w", h=height, w=width
    )


class BiasFreeLayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        if len(normalized_shape) != 1:
            raise ValueError("LayerNorm expects a one-dimensional shape")
        self.weight = nn.Parameter(torch.ones(normalized_shape))

    def forward(self, x):
        variance = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(variance + 1e-5) * self.weight


class WithBiasLayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        if len(normalized_shape) != 1:
            raise ValueError("LayerNorm expects a one-dimensional shape")
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        variance = x.var(-1, keepdim=True, unbiased=False)
        return (
            (x - mean) / torch.sqrt(variance + 1e-5) * self.weight
            + self.bias
        )


class LayerNorm(nn.Module):
    def __init__(self, dim, layer_norm_type):
        super().__init__()
        if layer_norm_type == "BiasFree":
            self.body = BiasFreeLayerNorm(dim)
        else:
            self.body = WithBiasLayerNorm(dim)

    def forward(self, x):
        height, width = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), height, width)


class FeedForward(nn.Module):
    def __init__(self, dim, expansion_factor, bias):
        super().__init__()
        hidden_features = int(dim * expansion_factor)
        self.project_in = nn.Conv2d(
            dim, hidden_features * 2, kernel_size=1, bias=bias
        )
        self.dwconv = nn.Conv2d(
            hidden_features * 2,
            hidden_features * 2,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=hidden_features * 2,
            bias=bias,
        )
        self.project_out = nn.Conv2d(
            hidden_features, dim, kernel_size=1, bias=bias
        )

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        return self.project_out(F.gelu(x1) * x2)


class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim * 3,
            dim * 3,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=dim * 3,
            bias=bias,
        )
        self.project_out = nn.Conv2d(
            dim, dim, kernel_size=1, bias=bias
        )

    def forward(self, x):
        _, _, height, width = x.shape
        q, k, v = self.qkv_dwconv(self.qkv(x)).chunk(3, dim=1)
        q = rearrange(
            q, "b (head c) h w -> b head c (h w)",
            head=self.num_heads,
        )
        k = rearrange(
            k, "b (head c) h w -> b head c (h w)",
            head=self.num_heads,
        )
        v = rearrange(
            v, "b (head c) h w -> b head c (h w)",
            head=self.num_heads,
        )
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attention = (
            q @ k.transpose(-2, -1)
        ) * self.temperature
        attention = attention.softmax(dim=-1)
        output = attention @ v
        output = rearrange(
            output,
            "b head c (h w) -> b (head c) h w",
            head=self.num_heads,
            h=height,
            w=width,
        )
        return self.project_out(output)


class TransformerBlock(nn.Module):
    def __init__(
            self,
            dim,
            num_heads,
            ffn_expansion_factor,
            bias,
            layer_norm_type):
        super().__init__()
        self.norm1 = LayerNorm(dim, layer_norm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, layer_norm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.ffn(self.norm2(x))


class OverlapPatchEmbed(nn.Module):
    def __init__(self, input_channels, embed_dim, bias=False):
        super().__init__()
        self.proj = nn.Conv2d(
            input_channels,
            embed_dim,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=bias,
        )

    def forward(self, x):
        return self.proj(x)


class Downsample(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(
                features,
                features // 2,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.PixelUnshuffle(2),
        )

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(
                features,
                features * 2,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.PixelShuffle(2),
        )

    def forward(self, x):
        return self.body(x)


class SpaChaPromptGenBlock(nn.Module):
    def __init__(
            self,
            spatial_prompt_num=5,
            spectral_prompt_num=5,
            spatial_prompt_size=32,
            spectral_prompt_dim=64):
        super().__init__()
        self.spatial_prompt = nn.Parameter(
            torch.rand(
                1,
                spatial_prompt_num,
                spatial_prompt_size,
                spatial_prompt_size,
            )
        )
        self.spectral_prompt = nn.Parameter(
            torch.rand(spectral_prompt_num, spectral_prompt_dim)
        )
        self.linear_layer_spatial = nn.Linear(
            spectral_prompt_dim, spatial_prompt_num
        )
        self.linear_layer_spectral = nn.Linear(
            spectral_prompt_dim, spectral_prompt_num
        )
        self.conv = nn.Conv2d(
            spectral_prompt_dim * 2,
            spectral_prompt_dim,
            kernel_size=1,
            stride=1,
            bias=False,
        )

    def forward(self, x):
        batch_size = x.shape[0]
        embedding = x.mean(dim=(-2, -1))
        spatial_weights = F.softmax(
            self.linear_layer_spatial(embedding), dim=1
        )
        spectral_weights = F.softmax(
            self.linear_layer_spectral(embedding), dim=1
        )
        spatial_prompt = torch.sum(
            spatial_weights.unsqueeze(-1).unsqueeze(-1)
            * self.spatial_prompt.repeat(batch_size, 1, 1, 1),
            dim=1,
        )
        spectral_prompt = torch.sum(
            spectral_weights.unsqueeze(-1)
            * self.spectral_prompt.unsqueeze(0).repeat(
                batch_size, 1, 1
            ),
            dim=1,
        )
        spatial_feature = spatial_prompt.unsqueeze(1) * x
        spectral_feature = (
            spectral_prompt.unsqueeze(-1).unsqueeze(-1) * x
        )
        return self.conv(
            torch.cat((spatial_feature, spectral_feature), dim=1)
        ) + x


class SpatialChannelPrompt(nn.Module):
    def __init__(
            self,
            dim=48,
            num_blocks=(4, 6, 6, 8),
            num_refinement_blocks=4,
            heads=(1, 2, 4, 8),
            ffn_expansion_factor=2.66,
            bias=False,
            LayerNorm_type="WithBias",
            decoder=True):
        super().__init__()
        self.patch_embed4 = OverlapPatchEmbed(4, dim)
        self.patch_embed8 = OverlapPatchEmbed(8, dim)
        self.decoder = decoder

        self.encoder_prompt_dim0 = SpaChaPromptGenBlock(
            spatial_prompt_size=128, spectral_prompt_dim=dim
        )
        self.encoder_prompt_dim1 = SpaChaPromptGenBlock(
            spatial_prompt_size=64, spectral_prompt_dim=dim * 2
        )
        self.encoder_prompt_dim2 = SpaChaPromptGenBlock(
            spatial_prompt_size=32, spectral_prompt_dim=dim * 4
        )
        self.encoder_prompt_dim3 = SpaChaPromptGenBlock(
            spatial_prompt_size=16, spectral_prompt_dim=dim * 8
        )
        self.decoder_prompt_dim2 = SpaChaPromptGenBlock(
            spatial_prompt_size=32, spectral_prompt_dim=dim * 4
        )
        self.decoder_prompt_dim1 = SpaChaPromptGenBlock(
            spatial_prompt_size=64, spectral_prompt_dim=dim * 2
        )
        self.decoder_prompt_dim0 = SpaChaPromptGenBlock(
            spatial_prompt_size=128, spectral_prompt_dim=dim * 2
        )

        def blocks(level, head_index, block_count):
            return nn.Sequential(*[
                TransformerBlock(
                    dim=level,
                    num_heads=heads[head_index],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    layer_norm_type=LayerNorm_type,
                )
                for _ in range(block_count)
            ])

        self.encoder_level1 = blocks(dim, 0, num_blocks[0])
        self.down1_2 = Downsample(dim)
        self.encoder_level2 = blocks(dim * 2, 1, num_blocks[1])
        self.down2_3 = Downsample(dim * 2)
        self.encoder_level3 = blocks(dim * 4, 2, num_blocks[2])
        self.down3_4 = Downsample(dim * 4)
        self.latent = blocks(dim * 8, 3, num_blocks[3])

        self.up4_3 = Upsample(dim * 8)
        self.reduce_chan_level3 = nn.Conv2d(
            dim * 8, dim * 4, kernel_size=1, bias=bias
        )
        self.decoder_level3 = blocks(dim * 4, 2, num_blocks[2])
        self.up3_2 = Upsample(dim * 4)
        self.reduce_chan_level2 = nn.Conv2d(
            dim * 4, dim * 2, kernel_size=1, bias=bias
        )
        self.decoder_level2 = blocks(dim * 2, 1, num_blocks[1])
        self.up2_1 = Upsample(dim * 2)
        self.decoder_level1 = nn.Sequential(*[
            TransformerBlock(
                dim=dim * 2,
                num_heads=heads[0],
                ffn_expansion_factor=ffn_expansion_factor,
                bias=bias,
                layer_norm_type=LayerNorm_type,
            )
            for _ in range(num_blocks[0])
        ])
        self.refinement = nn.Sequential(*[
            TransformerBlock(
                dim=dim * 2,
                num_heads=heads[0],
                ffn_expansion_factor=ffn_expansion_factor,
                bias=bias,
                layer_norm_type=LayerNorm_type,
            )
            for _ in range(num_refinement_blocks)
        ])
        self.output4 = nn.Conv2d(
            dim * 2, 4, kernel_size=3, stride=1, padding=1, bias=bias
        )
        self.output8 = nn.Conv2d(
            dim * 2, 8, kernel_size=3, stride=1, padding=1, bias=bias
        )

    def forward(self, input_image):
        channels = input_image.shape[1]
        if channels == 4:
            level1 = self.patch_embed4(input_image)
        elif channels == 8:
            level1 = self.patch_embed8(input_image)
        else:
            raise ValueError(
                f"Expected 4 or 8 MS channels, received {channels}"
            )

        level1 = self.encoder_prompt_dim0(level1)
        encoded1 = self.encoder_level1(level1)
        level2 = self.encoder_prompt_dim1(self.down1_2(encoded1))
        encoded2 = self.encoder_level2(level2)
        level3 = self.encoder_prompt_dim2(self.down2_3(encoded2))
        encoded3 = self.encoder_level3(level3)
        level4 = self.encoder_prompt_dim3(self.down3_4(encoded3))
        latent = self.latent(level4)

        decoded3 = self.up4_3(latent)
        decoded3 = self.reduce_chan_level3(
            torch.cat([decoded3, encoded3], dim=1)
        )
        decoded3 = self.decoder_prompt_dim2(decoded3)
        decoded3 = self.decoder_level3(decoded3)

        decoded2 = self.up3_2(decoded3)
        decoded2 = self.reduce_chan_level2(
            torch.cat([decoded2, encoded2], dim=1)
        )
        decoded2 = self.decoder_prompt_dim1(decoded2)
        decoded2 = self.decoder_level2(decoded2)

        decoded1 = self.up2_1(decoded2)
        decoded1 = torch.cat([decoded1, encoded1], dim=1)
        decoded1 = self.decoder_prompt_dim0(decoded1)
        decoded1 = self.decoder_level1(decoded1)
        decoded1 = self.refinement(decoded1)

        if channels == 4:
            return self.output4(decoded1) + input_image
        return self.output8(decoded1) + input_image
