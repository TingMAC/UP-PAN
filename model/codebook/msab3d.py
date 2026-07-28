"""3D spectral-spatial attention blocks."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.dim = dim

    def forward(self, x):
        return self.fn(self.norm(x))


class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)


class FeedForward3D(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(dim, dim * mult, 1, bias=False),
            GELU(),
            nn.Conv3d(
                dim * mult,
                dim * mult,
                3,
                padding=1,
                groups=dim * mult,
                bias=False,
            ),
            GELU(),
            nn.Conv3d(dim * mult, dim, 1, bias=False),
        )
        self.dim = dim
        self.mult = mult

    def forward(self, x):
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        x = self.net(x)
        return x.permute(0, 2, 3, 4, 1).contiguous()


class MS_MSA3D(nn.Module):
    def __init__(self, dim, dim_head, heads):
        super().__init__()
        self.num_heads = heads
        self.dim_head = dim_head
        inner_dim = dim_head * heads
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.rescale = nn.Parameter(torch.ones(heads, 1, 1))
        self.proj = nn.Linear(inner_dim, dim, bias=True)
        self.pos_emb = nn.Sequential(
            nn.Conv3d(
                dim,
                dim,
                3,
                padding=1,
                bias=False,
                groups=dim,
            ),
            GELU(),
            nn.Conv3d(
                dim,
                dim,
                3,
                padding=1,
                bias=False,
                groups=dim,
            ),
        )

    def forward(self, x_input):
        batch, depth, height, width, channels = x_input.shape
        flattened = x_input.view(
            batch, depth * height * width, channels
        )
        q, k, v = map(
            lambda tensor: rearrange(
                tensor,
                "b n (h d) -> b h n d",
                h=self.num_heads,
            ),
            self.to_qkv(flattened).chunk(3, dim=-1),
        )
        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)
        q = F.normalize(q, dim=-1, p=2)
        k = F.normalize(k, dim=-1, p=2)
        attention = (k @ q.transpose(-2, -1)) * self.rescale
        attention = attention.softmax(dim=-1)
        output = attention @ v
        output = rearrange(
            output,
            "b h c (d hh ww) -> b (d hh ww) (h c)",
            d=depth,
            hh=height,
            ww=width,
        )
        output = self.proj(output).view(
            batch, depth, height, width, channels
        )
        positional = self.pos_emb(
            x_input.permute(0, 4, 1, 2, 3)
        ).permute(0, 2, 3, 4, 1)
        return output + positional


class MSAB3D(nn.Module):
    def __init__(self, dim, dim_head, heads, num_blocks):
        super().__init__()
        self.blocks = nn.ModuleList([
            nn.ModuleList([
                MS_MSA3D(dim, dim_head, heads),
                PreNorm(dim, FeedForward3D(dim)),
            ])
            for _ in range(num_blocks)
        ])

    def forward(self, x):
        x = x.permute(0, 2, 3, 4, 1).contiguous()
        for attention, feed_forward in self.blocks:
            x = attention(x) + x
            x = feed_forward(x) + x
        return x.permute(0, 4, 1, 2, 3).contiguous()
