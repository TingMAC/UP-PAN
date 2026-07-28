"""Wavelength-conditioned dynamic convolution."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.downscale_freq_shift = 1
        self.max_period = 10000

    def forward(self, wavelength):
        position = wavelength.float().unsqueeze(1)
        div_term = torch.exp(
            torch.arange(
                0,
                self.embedding_dim,
                2,
                device=wavelength.device,
                dtype=torch.float32,
            )
            * (-math.log(10000.0) / self.embedding_dim)
        )
        embedding = (
            torch.zeros_like(wavelength)
            .unsqueeze(1)
            .repeat(1, self.embedding_dim)
        )
        embedding[:, 0::2] = torch.sin(position * div_term)
        embedding[:, 1::2] = torch.cos(position * div_term)
        return embedding


class DynamicChannelAdaptation(nn.Module):
    """Generate sensor-aware convolution kernels from band wavelengths."""

    wavelengths = (
        torch.tensor([485, 555, 660, 830], dtype=torch.float64),
        torch.tensor([485, 560, 660, 830], dtype=torch.float64),
        torch.tensor(
            [425, 480, 545, 605, 660, 725, 832, 950],
            dtype=torch.float64,
        ),
        torch.tensor([480, 545, 672, 850], dtype=torch.float64),
    )

    def __init__(
            self,
            in_channels,
            out_channels,
            scale,
            kernel_size=3,
            embedding_dim=64,
            num_heads=1,
            dropout=0.1,
            is_transpose=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.embedding_dim = embedding_dim
        self.num_channels = max(out_channels, in_channels)
        self.L = self.num_channels
        self.scale = scale
        self.K = kernel_size
        self.is_transpose = is_transpose

        self.pos_encoder = SinusoidalTimeEmbedding(embedding_dim)
        wavelength_mlp = nn.Sequential(
            nn.Linear(embedding_dim, out_channels),
            nn.GELU(),
            nn.Linear(out_channels, out_channels),
        )
        self.wavelength_embedding = nn.ModuleList(
            [wavelength_mlp for _ in range(self.num_channels)]
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=out_channels,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=2
        )
        self.weight_query = nn.Parameter(
            torch.randn(self.L, out_channels)
        )
        self.bias_query = nn.Parameter(torch.randn(1, out_channels))

        generated_features = (
            min(in_channels, out_channels) * kernel_size * kernel_size
        )
        self.weight_generator = nn.Sequential(
            nn.Linear(out_channels, generated_features),
            nn.GELU(),
            nn.Linear(generated_features, generated_features),
        )
        self.bias_generator = nn.Sequential(
            nn.Linear(out_channels, out_channels),
            nn.GELU(),
            nn.Linear(out_channels, out_channels),
        )

    def forward(self, x, one_hot):
        outputs = []
        for batch_index in range(x.shape[0]):
            x_batch = x[batch_index].unsqueeze(0)
            task_id = one_hot[batch_index].argmax().item()
            wavelengths = self.wavelengths[task_id].to(x.device)
            wavelength_features = self.pos_encoder(
                wavelengths.view(-1)
            ).view(self.num_channels, -1).to(torch.float32)
            embedded = torch.zeros(
                self.num_channels,
                self.out_channels,
                device=x.device,
                dtype=wavelength_features.dtype,
            )
            for index, embedding in enumerate(
                    self.wavelength_embedding):
                embedded[index] = embedding(wavelength_features[index])

            transformer_input = torch.cat(
                [self.weight_query, embedded, self.bias_query], dim=0
            )
            transformer_output = self.transformer(transformer_input)
            weight_token = transformer_output[:self.L]
            bias_token = transformer_output[-1:]
            weights = self.weight_generator(weight_token + embedded)

            if self.is_transpose:
                weights = weights.view(
                    self.in_channels,
                    self.out_channels,
                    self.kernel_size,
                    self.kernel_size,
                ).contiguous()
            else:
                weights = weights.view(
                    self.out_channels,
                    self.in_channels,
                    self.kernel_size,
                    self.kernel_size,
                ).contiguous()
            bias = self.bias_generator(bias_token).view(-1)

            if self.is_transpose:
                padding, output_padding = self.auto_transpose_params(
                    input_size=x.shape[-1],
                    output_size=x.shape[-1] * self.scale,
                    kernel_size=self.K,
                    stride=self.scale,
                )
                output = F.conv_transpose2d(
                    x_batch,
                    weight=weights,
                    bias=bias,
                    stride=self.scale,
                    padding=padding,
                    output_padding=output_padding,
                )
            else:
                padding = self.auto_conv_params(
                    input_size=x.shape[-1],
                    output_size=x.shape[-1] // self.scale,
                    kernel_size=self.K,
                    stride=self.scale,
                )
                output = F.conv2d(
                    x_batch,
                    weight=weights,
                    bias=bias,
                    stride=self.scale,
                    padding=padding,
                )
            outputs.append(output.squeeze(0))
        return torch.stack(outputs, dim=0)

    @staticmethod
    def auto_transpose_params(
            input_size, output_size, kernel_size, stride):
        for padding in range(kernel_size):
            for output_padding in (0, 1):
                calculated = (
                    (input_size - 1) * stride
                    - 2 * padding
                    + kernel_size
                    + output_padding
                )
                if calculated == output_size:
                    return padding, output_padding
        raise ValueError("No valid transpose-convolution padding")

    @staticmethod
    def auto_conv_params(input_size, output_size, kernel_size, stride):
        for padding in range(kernel_size):
            calculated = (
                input_size + 2 * padding - kernel_size
            ) // stride + 1
            if calculated == output_size:
                return padding
        raise ValueError("No valid convolution padding")
