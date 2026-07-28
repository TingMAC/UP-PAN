import torch
import torch.nn as nn

from .vq import ResidualVectorQuantizer


class BlockBasedResidualVectorQuantizer3D(ResidualVectorQuantizer):
    """Residual VQ: shared codebook first, then task-specific codebook."""

    def __init__(
            self,
            n_shared=1024,
            n_task=256,
            e_dim=256,
            beta=0.25,
            LQ_stage=False,
            depth=6,
            unfold_size=2,
            mlp_codebook=False):
        del mlp_codebook
        super().__init__(1, 1, beta, LQ_stage, depth)
        self.unfold_size = unfold_size
        self.unfold = nn.Unfold(
            kernel_size=(unfold_size, unfold_size)
        )
        self.beta = beta
        self.e_dim = e_dim
        self.depth = depth
        self.n_shared = n_shared
        self.n_task = n_task

        self.shared_codebook = nn.Embedding(n_shared, e_dim)
        self.shared_codebook.weight.data.uniform_(
            -1.0 / n_shared, 1.0 / n_shared
        )
        self.task_codebooks = nn.ModuleList([
            nn.Embedding(n_task, e_dim) for _ in range(4)
        ])
        for codebook in self.task_codebooks:
            codebook.weight.data.uniform_(
                -1.0 / n_shared, 1.0 / n_shared
            )

    def forward(self, z, one_hot):
        batch_size, channels, _, height, width = z.shape
        quantized = torch.zeros_like(z)
        total_codebook_loss = 0.0

        for batch_index in range(batch_size):
            z_batch = z[batch_index].unsqueeze(0)
            flattened = z_batch.view(
                -1, channels, height, width
            )
            flattened = self.unfold(flattened).permute(0, 2, 1)
            flattened = flattened.reshape(-1, self.e_dim)
            quantized_batch = torch.zeros_like(flattened)
            residual = flattened

            shared_weights = self.shared_codebook.weight
            for _ in range(self.depth):
                distances = self.dist(residual, shared_weights)
                indices = torch.argmin(distances, dim=1)
                delta = self.shared_codebook(indices)
                quantized_batch = quantized_batch + delta
                residual = residual - delta

            task_id = one_hot[batch_index].argmax().item()
            task_codebook = self.task_codebooks[task_id]
            task_weights = task_codebook.weight
            for _ in range(self.depth // 2):
                distances = self.dist(residual, task_weights)
                indices = torch.argmin(distances, dim=1)
                delta = task_codebook(indices)
                quantized_batch = quantized_batch + delta
                residual = residual - delta

            quantized_batch = self.fold(
                quantized_batch, z_batch.shape
            )
            embedding_loss = torch.mean(
                (quantized_batch.detach() - z_batch) ** 2
            )
            codebook_loss = torch.mean(
                (quantized_batch - z_batch.detach()) ** 2
            )
            total_codebook_loss += (
                codebook_loss + embedding_loss * self.beta
            )
            quantized[batch_index] = (
                z_batch + (quantized_batch - z_batch).detach()
            )

        quantized = z + (quantized - z).detach()
        return quantized, total_codebook_loss / batch_size

    def fold(self, patches, output_shape):
        batch_size, channels, depth, height, width = output_shape
        patches = patches.view(
            batch_size * depth, -1, self.e_dim
        ).permute(0, 2, 1)
        fold = nn.Fold(
            output_size=(height, width),
            kernel_size=(self.unfold_size, self.unfold_size),
        )
        overlap_count = torch.ones(
            1,
            channels,
            height,
            width,
            device=patches.device,
            dtype=patches.dtype,
        )
        overlap_count = fold(self.unfold(overlap_count))
        return (fold(patches) / overlap_count).view(output_shape)
