"""Small tensor utilities for diffusion losses and noise sampling."""
from typing import List

import math
import torch
from torch import Tensor
from torch_scatter import scatter_add

try:
    from ..utils.tensor import remove_mean_batch
except ImportError:  # pragma: no cover - support running from the repo root
    from utils.tensor import remove_mean_batch


def assert_mean_zero_with_mask(x, node_mask, eps=1e-10):
    """Assert that each mask group has approximately zero summed coordinates."""
    largest_value = x.abs().max().item()
    error = scatter_add(x, node_mask, dim=0).abs().max().item()
    rel_error = error / (largest_value + eps)
    assert rel_error < 1e-2, f"Mean is not zero, relative_error {rel_error}"


def sample_center_gravity_zero_gaussian_batch(
    size: List[int], indices: List[Tensor]
) -> Tensor:
    """Sample Gaussian coordinate noise and remove per-sample center of mass."""
    assert len(size) == 2
    x = torch.randn(size, device=indices[0].device)

    # This projection only works because Gaussian is rotation invariant
    # around zero and samples are independent!
    x_projected = remove_mean_batch(x, torch.cat(indices))
    return x_projected


def sum_except_batch(x, indices, dim_size):
    """Sum feature dimensions, then scatter sums by batch/sample index."""
    return scatter_add(x.sum(-1), indices, dim=0, dim_size=dim_size)


def cdf_standard_gaussian(x):
    """Standard normal CDF used by legacy likelihood code paths."""
    return 0.5 * (1.0 + torch.erf(x / math.sqrt(2)))


def sample_gaussian(size, device):
    """Plain standard Gaussian sampler."""
    x = torch.randn(size, device=device)
    return x


def num_nodes_to_batch_mask(n_samples, num_nodes, device):
    """Build a node-level batch mask from per-sample node counts."""
    assert isinstance(num_nodes, int) or len(num_nodes) == n_samples

    if isinstance(num_nodes, torch.Tensor):
        num_nodes = num_nodes.to(device)

    sample_inds = torch.arange(n_samples, device=device)

    return torch.repeat_interleave(sample_inds, num_nodes)
