"""Utility functions shared by EGNN/LEFTNet message-passing layers."""
import torch
from torch import Tensor

try:
    from ..utils.graph import get_edge_vectors_pbc
    from ..utils.tensor import move_by_com
except ImportError:  # pragma: no cover - support running from the repo root
    from utils.graph import get_edge_vectors_pbc
    from utils.tensor import move_by_com


def coord2cross(x, edge_index, norm_constant=1):
    """Return normalized cross products for optional reflection-sensitive terms."""
    row, col = edge_index
    cross = torch.cross(x[row], x[col], dim=1)
    norm = torch.linalg.norm(cross, dim=1, keepdim=True)
    cross = cross / (norm + norm_constant)
    return cross


def coord2diff(
    x,
    edge_index,
    norm_constant=1,
    cell=None,
    pbc=None,
    edge_shift=None,
    cell_offsets=None,
    fragment=None,
    mask=None,
):
    """
    Calculate coordinate differences between atoms.

    Periodic geometry is delegated to ``utils.graph.get_edge_vectors_pbc`` so
    the model uses one PBC convention throughout the codebase. New periodic
    code should generally pass ``cell_offsets``/``edge_shift`` and recover edge
    vectors through ``utils.graph.get_pbc_distances`` directly.

    Args:
        x: Atomic coordinates (n_atoms, 3)
        edge_index: Edge indices (2, n_edges)
        norm_constant: Normalization constant for division
        cell: Optional unit cell vectors (3, 3) for periodic systems
        pbc: Optional periodic boundary conditions [x, y, z] for periodic systems
        edge_shift: Optional integer lattice offsets for strict periodic graphs
        cell_offsets: Alias for edge_shift
        fragment: Optional fragment ids for stacked per-fragment cells
        mask: Optional batch ids for batched per-fragment cells

    Returns:
        radial: Squared distances (n_edges, 1)
        coord_diff: Normalized coordinate differences (n_edges, 3)
    """
    row, col = edge_index

    if cell is not None and pbc is not None:
        # Delegate PBC vector recovery to utils.graph so dataset and model use
        # the same cell_offsets convention.
        if edge_shift is None:
            edge_shift = cell_offsets
        coord_diff, distances = get_edge_vectors_pbc(
            pos=x,
            edge_index=edge_index,
            cell=cell,
            pbc=pbc,
            edge_shift=edge_shift,
            fragment=fragment,
            mask=mask,
            normalize=False,
        )
        radial = distances.pow(2).unsqueeze(1)
    else:
        coord_diff = x[row] - x[col]
        radial = torch.sum((coord_diff) ** 2, 1).unsqueeze(1)

    # Normalize the vector but keep the scalar squared distance separately for
    # invariant edge features.
    norm = torch.sqrt(radial + 1e-8)
    coord_diff = coord_diff / (norm + norm_constant)
    return radial, coord_diff


def unsorted_segment_sum(
    data, segment_ids, num_segments, normalization_factor, aggregation_method: str
):
    r"""Custom PyTorch op to replicate TensorFlow's `unsorted_segment_sum`.
    Normalization: 'sum' or 'mean'.
    """
    result_shape = (num_segments, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    # For graph message passing, segment_ids is usually edge_index[0]. This
    # scatters all edge messages back to their source/center node.
    result.scatter_add_(0, segment_ids, data)
    if aggregation_method == "sum":
        result = result / normalization_factor

    if aggregation_method == "mean":
        norm = data.new_zeros(result.shape)
        norm.scatter_add_(0, segment_ids, data.new_ones(data.shape))
        norm[norm == 0] = 1
        result = result / norm
    return result


def get_ji_bond_index(bond_atom_indices: Tensor) -> Tensor:
    r"""Get the index for e_ji
    for example, bond_atom_indices = [[0, 1], [1, 0]], returns [1, 0]

    Args:
        bond_atom_indices (Tensor): (2, n_bonds) for ij

    Returns:
        Tensor: index for ji
    """
    bond_atom_indices = torch.transpose(bond_atom_indices, 0, 1)
    _index = torch.tensor([1, 0], dtype=torch.long, device=bond_atom_indices.device)
    reverse_bond_atom_indices = bond_atom_indices[:, _index]
    bond_ji_index = []
    for ij in range(bond_atom_indices.shape[0]):
        reverse_index = torch.where(
            (bond_atom_indices == reverse_bond_atom_indices[ij]).all(dim=1)
        )[0]
        if reverse_index.numel() != 1:
            raise ValueError(
                "Every edge must have exactly one reverse edge to symmetrize"
            )
        bond_ji_index.append(reverse_index)
    return torch.concat(bond_ji_index).long()


def symmetrize_edge(edge_attr: Tensor, edge_ji_indices: Tensor) -> Tensor:
    return (edge_attr + edge_attr[edge_ji_indices]) / 2
