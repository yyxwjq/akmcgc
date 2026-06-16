"""Shared tensor utilities used across diffusion, denoising, and models."""
import torch
from torch_scatter import scatter_mean


def remove_mean_batch(x: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Subtract the per-index mean from a tensor.

    Used for zero-center coordinate noise/updates per reaction sample. If
    ``indices`` is ``mask``, every batch graph is centered independently.
    """
    mean = scatter_mean(x, indices, dim=0)
    return x - mean[indices]


def move_by_com(pos: torch.Tensor) -> torch.Tensor:
    """Center one coordinate tensor by its global center of mass."""
    return pos - torch.mean(pos, dim=0)
