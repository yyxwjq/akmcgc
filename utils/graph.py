"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
from typing import Dict, Optional, Tuple

import torch
from ase import Atoms
from ase.neighborlist import neighbor_list

try:
    from .dtype import resolve_float_dtype
except ImportError:  # pragma: no cover - supports direct local imports in notebooks
    from dtype import resolve_float_dtype


def atoms_to_tensors(
    atoms: Atoms,
    device: torch.device,
    num_elements: int = 118,
    dtype: torch.dtype | str = torch.float64,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert ASE atoms to ``h``, positions, cell, and PBC tensors.

    The returned node feature ``h`` follows the feature order
    ``[pos, one_hot, charge]``.
    """
    dtype = resolve_float_dtype(dtype)
    atomic_numbers = torch.as_tensor(
        atoms.get_atomic_numbers(),
        dtype=torch.long,
        device=device,
    )
    feature_index = atomic_numbers - 1
    if torch.any(feature_index < 0) or torch.any(feature_index >= num_elements):
        invalid = atomic_numbers[(feature_index < 0) | (feature_index >= num_elements)]
        raise ValueError(
            f"Atomic numbers out of supported range 1..{num_elements}: "
            f"{invalid.detach().cpu().tolist()}"
        )

    one_hot = torch.zeros(
        (len(atomic_numbers), num_elements),
        dtype=dtype,
        device=device,
    )
    one_hot.scatter_(1, feature_index.unsqueeze(1), 1.0)

    pos = torch.as_tensor(atoms.get_positions(), dtype=dtype, device=device)
    cell = torch.as_tensor(atoms.get_cell().array, dtype=dtype, device=device)
    pbc = torch.as_tensor(atoms.get_pbc(), dtype=torch.bool, device=device)
    charge = _get_outer_shell_electrons(atomic_numbers).to(dtype=dtype).view(-1, 1)
    h = torch.cat([pos, one_hot, charge], dim=1)

    return h, pos, cell, pbc


def _get_outer_shell_electrons(atomic_numbers: torch.Tensor) -> torch.Tensor:
    """Return the number of electrons in the outermost shell.

    This matches the literal meaning of ``initial charge`` requested here:
    the electron count on the highest principal quantum-number shell.
    """
    return torch.tensor(
        [_outer_shell_electrons_single(int(z)) for z in atomic_numbers.tolist()],
        dtype=torch.long,
        device=atomic_numbers.device,
    )


_ORBITAL_ORDER = [
    (1, "s", 2),
    (2, "s", 2),
    (2, "p", 6),
    (3, "s", 2),
    (3, "p", 6),
    (4, "s", 2),
    (3, "d", 10),
    (4, "p", 6),
    (5, "s", 2),
    (4, "d", 10),
    (5, "p", 6),
    (6, "s", 2),
    (4, "f", 14),
    (5, "d", 10),
    (6, "p", 6),
    (7, "s", 2),
    (5, "f", 14),
    (6, "d", 10),
    (7, "p", 6),
]


_CONFIG_EXCEPTIONS = {
    24: [(1, "s", 2), (2, "s", 2), (2, "p", 6), (3, "s", 2), (3, "p", 6), (3, "d", 5), (4, "s", 1)],
    29: [(1, "s", 2), (2, "s", 2), (2, "p", 6), (3, "s", 2), (3, "p", 6), (3, "d", 10), (4, "s", 1)],
    41: [(1, "s", 2), (2, "s", 2), (2, "p", 6), (3, "s", 2), (3, "p", 6), (4, "s", 2), (3, "d", 10), (4, "p", 6), (4, "d", 4), (5, "s", 1)],
    42: [(1, "s", 2), (2, "s", 2), (2, "p", 6), (3, "s", 2), (3, "p", 6), (4, "s", 2), (3, "d", 10), (4, "p", 6), (4, "d", 5), (5, "s", 1)],
    44: [(1, "s", 2), (2, "s", 2), (2, "p", 6), (3, "s", 2), (3, "p", 6), (4, "s", 2), (3, "d", 10), (4, "p", 6), (4, "d", 7), (5, "s", 1)],
    45: [(1, "s", 2), (2, "s", 2), (2, "p", 6), (3, "s", 2), (3, "p", 6), (4, "s", 2), (3, "d", 10), (4, "p", 6), (4, "d", 8), (5, "s", 1)],
    46: [(1, "s", 2), (2, "s", 2), (2, "p", 6), (3, "s", 2), (3, "p", 6), (4, "s", 2), (3, "d", 10), (4, "p", 6), (4, "d", 10)],
    47: [(1, "s", 2), (2, "s", 2), (2, "p", 6), (3, "s", 2), (3, "p", 6), (4, "s", 2), (3, "d", 10), (4, "p", 6), (4, "d", 10), (5, "s", 1)],
    78: [(1, "s", 2), (2, "s", 2), (2, "p", 6), (3, "s", 2), (3, "p", 6), (4, "s", 2), (3, "d", 10), (4, "p", 6), (5, "s", 2), (4, "d", 10), (5, "p", 6), (6, "s", 1), (4, "f", 14), (5, "d", 9)],
    79: [(1, "s", 2), (2, "s", 2), (2, "p", 6), (3, "s", 2), (3, "p", 6), (4, "s", 2), (3, "d", 10), (4, "p", 6), (5, "s", 2), (4, "d", 10), (5, "p", 6), (6, "s", 1), (4, "f", 14), (5, "d", 10)],
}


def _outer_shell_electrons_single(atomic_number: int) -> int:
    if atomic_number <= 0 or atomic_number > 118:
        raise ValueError(f"Unsupported atomic number for outer-shell lookup: {atomic_number}")

    config = _CONFIG_EXCEPTIONS.get(atomic_number)
    if config is None:
        config = []
        remaining = atomic_number
        for n, orbital, capacity in _ORBITAL_ORDER:
            if remaining <= 0:
                break
            fill = min(remaining, capacity)
            if fill > 0:
                config.append((n, orbital, fill))
                remaining -= fill

    max_n = max(n for n, _, _ in config)
    return sum(count for n, _, count in config if n == max_n)


def radius_graph_ase(
    atoms: Atoms,
    cutoff: float,
    max_neigh: Optional[int] = 200,
    use_pbc: bool = True,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype | str = torch.float64,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build an ASE radius graph with MIC-aware neighbor search.

    Returns:
        edge_index: [2, n_edges]
        cell_offsets: [n_edges, 3], lattice-image offsets from ASE
    """
    dtype = resolve_float_dtype(dtype)
    atoms_for_graph = atoms.copy()
    if not use_pbc:
        atoms_for_graph.set_pbc(False)

    src, dst, cell_offsets, distance = neighbor_list(
        "ijSd",
        atoms_for_graph,
        cutoff=cutoff,
        self_interaction=False,
    )

    src = torch.as_tensor(src, dtype=torch.long, device=device)
    dst = torch.as_tensor(dst, dtype=torch.long, device=device)
    cell_offsets = torch.as_tensor(cell_offsets, dtype=dtype, device=device)
    distance = torch.as_tensor(distance, dtype=dtype, device=device)
    src, dst, cell_offsets = _limit_neighbors(
        src,
        dst,
        cell_offsets,
        distance,
        max_neigh,
    )

    return torch.stack([src, dst], dim=0), cell_offsets


def _limit_neighbors(
    src: torch.Tensor,
    dst: torch.Tensor,
    edge_shift: torch.Tensor,
    distance: torch.Tensor,
    max_neigh: Optional[int],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Keep at most ``max_neigh`` nearest outgoing edges for every source atom."""
    if max_neigh is None or max_neigh <= 0 or src.numel() == 0:
        return src, dst, edge_shift

    keep_parts = []
    for atom_index in torch.unique(src, sorted=True):
        edge_ids = torch.where(src == atom_index)[0]
        if edge_ids.numel() > max_neigh:
            nearest = torch.argsort(distance[edge_ids])[:max_neigh]
            edge_ids = edge_ids[nearest]
        keep_parts.append(edge_ids)

    keep = torch.cat(keep_parts, dim=0)
    keep = keep[torch.argsort(keep)]
    return src[keep], dst[keep], edge_shift[keep]


def get_pbc_distances(
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    cell: torch.Tensor,
    cell_offsets: torch.Tensor,
    neighbors: Optional[torch.Tensor] = None,
    pbc=None,
    fragment: Optional[torch.Tensor] = None,
    mask: Optional[torch.Tensor] = None,
    return_offsets: bool = False,
    return_distance_vec: bool = False,
    normalize: bool = False,
) -> Dict[str, torch.Tensor]:
    """AdsorbDiff-style periodic distance recovery for strict periodic graphs."""
    row, col = edge_index
    if cell.dim() == 4:
        if fragment is None or mask is None:
            raise ValueError("fragment and mask are required when cell is batched")
        offsets = get_pbc_offsets_batched(
            edge_index=edge_index,
            cell=cell,
            cell_offsets=cell_offsets,
            pbc=pbc,
            fragment=fragment,
            mask=mask,
        )
    else:
        offsets = get_pbc_offsets(
            edge_index=edge_index,
            cell=cell,
            cell_offsets=cell_offsets,
            pbc=pbc,
            fragment=fragment,
        )
    distance_vectors = pos[row] - pos[col] - offsets
    distances = distance_vectors.norm(dim=-1)

    out = {
        "edge_index": edge_index,
        "distances": distances,
    }
    if return_distance_vec:
        out["distance_vec"] = (
            distance_vectors / distances.clamp_min(1e-8).unsqueeze(1)
            if normalize
            else distance_vectors
        )
    if return_offsets:
        out["offsets"] = offsets
    if neighbors is not None:
        out["neighbors"] = neighbors
    return out


def get_edge_vectors_pbc(
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    cell: torch.Tensor,
    pbc,
    edge_shift: Optional[torch.Tensor] = None,
    fragment: Optional[torch.Tensor] = None,
    mask: Optional[torch.Tensor] = None,
    normalize: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compatibility wrapper returning periodic edge vectors and distances.

    By default this returns vectors in the `i - j_image` convention used by the
    current EGNN/LEFTNet code. If `edge_shift` is provided, it is interpreted in
    ASE's convention, where `pos_j + edge_shift @ cell - pos_i` is the physical
    edge vector from `i` to the chosen image of `j`.
    """
    if edge_shift is None:
        row, col = edge_index
        edge_vectors = pos[row] - pos[col]

        if cell.dim() != 2:
            raise ValueError("Automatic MIC without edge_shift expects a single [3,3] cell")

        inv_cell = torch.linalg.inv(cell)
        frac_vectors = edge_vectors @ inv_cell
        shifts = -torch.floor(frac_vectors + 0.5)

        pbc_tensor = torch.as_tensor(pbc, dtype=torch.bool, device=pos.device)
        shifts = torch.where(pbc_tensor.unsqueeze(0), shifts, torch.zeros_like(shifts))
        edge_vectors = edge_vectors + shifts @ cell

        distances = torch.linalg.norm(edge_vectors, dim=1)
        if normalize:
            edge_vectors = edge_vectors / distances.clamp_min(1e-8).unsqueeze(1)
        return edge_vectors, distances

    out = get_pbc_distances(
        pos=pos,
        edge_index=edge_index,
        cell=cell,
        cell_offsets=edge_shift,
        pbc=pbc,
        fragment=fragment,
        mask=mask,
        return_distance_vec=True,
        normalize=normalize,
    )
    return out["distance_vec"], out["distances"]


def get_pbc_offsets(
    edge_index: torch.Tensor,
    cell: torch.Tensor,
    cell_offsets: torch.Tensor,
    pbc,
    fragment: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    row, _ = edge_index
    shift = torch.as_tensor(cell_offsets, dtype=cell.dtype, device=cell.device)
    if cell.dim() == 4:
        if fragment is None:
            raise ValueError("fragment is required when cell is batched and stacked per fragment")
        raise ValueError("Use mask-aware get_pbc_offsets_batched for 4D cell tensors")
    if cell.dim() == 3:
        if fragment is None:
            raise ValueError("fragment is required when cell is stacked per fragment")
        edge_cell = cell[fragment[row]]
        edge_pbc = torch.as_tensor(pbc, dtype=torch.bool, device=cell.device)[fragment[row]]
        shift = torch.where(edge_pbc, shift, torch.zeros_like(shift))
        return torch.einsum("ni,nij->nj", shift, edge_cell)

    pbc_tensor = torch.as_tensor(pbc, dtype=torch.bool, device=cell.device)
    shift = torch.where(pbc_tensor.unsqueeze(0), shift, torch.zeros_like(shift))
    return shift @ cell


def get_pbc_offsets_batched(
    edge_index: torch.Tensor,
    cell: torch.Tensor,
    cell_offsets: torch.Tensor,
    pbc,
    fragment: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Return cartesian periodic offsets for batched joint graphs."""
    row, col = edge_index
    shift = torch.as_tensor(cell_offsets, dtype=cell.dtype, device=cell.device)
    edge_batch = mask[row]
    edge_fragment = fragment[row]
    edge_cell = cell[edge_batch, edge_fragment]
    edge_pbc = torch.as_tensor(pbc, dtype=torch.bool, device=cell.device)[edge_batch, edge_fragment]
    shift = torch.where(edge_pbc, shift, torch.zeros_like(shift))
    return torch.einsum("ni,nij->nj", shift, edge_cell)


def get_edge_target_positions(
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    cell: torch.Tensor,
    pbc,
    edge_shift: torch.Tensor,
    fragment: Optional[torch.Tensor] = None,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Return the periodic-image positions of edge targets."""
    _, col = edge_index
    if cell.dim() == 4:
        if fragment is None or mask is None:
            raise ValueError("fragment and mask are required when cell is batched")
        offsets = get_pbc_offsets_batched(
            edge_index=edge_index,
            cell=cell,
            cell_offsets=edge_shift,
            pbc=pbc,
            fragment=fragment,
            mask=mask,
        )
    else:
        offsets = get_pbc_offsets(
            edge_index=edge_index,
            cell=cell,
            cell_offsets=edge_shift,
            pbc=pbc,
            fragment=fragment,
        )
    return pos[col] + offsets


def get_neighbor_pairs(
    cell: torch.Tensor,
    pos: torch.Tensor,
    cutoff: float,
    pbc,
    max_neigh: Optional[int] = 200,
    all_neigh: bool = True,
    device: Optional[torch.device] = None,
):
    """Compatibility helper that builds radius pairs from tensors.

    New code should prefer ``radius_graph_ase`` when an ASE ``Atoms`` object is
    available. This helper is retained for modules that still import it.
    """
    del all_neigh
    device = device or pos.device
    n_atoms = pos.size(0)
    edge_src = []
    edge_dst = []
    edge_dist = []

    pbc_tensor = torch.as_tensor(pbc, dtype=torch.bool, device=pos.device)
    for src in range(n_atoms):
        vectors = pos[src].unsqueeze(0) - pos
        vectors = _apply_mic(vectors, cell, pbc_tensor)
        distances = torch.linalg.norm(vectors, dim=1)
        mask = (distances <= cutoff) & (distances > 0)
        dst = torch.where(mask)[0]
        if max_neigh is not None and max_neigh > 0 and dst.numel() > max_neigh:
            nearest = torch.argsort(distances[dst])[:max_neigh]
            dst = dst[nearest]
        edge_src.append(torch.full_like(dst, src))
        edge_dst.append(dst)
        edge_dist.append(distances[dst])

    if len(edge_src) == 0 or sum(part.numel() for part in edge_src) == 0:
        empty_long = torch.empty(0, dtype=torch.long, device=device)
        empty_float = torch.empty(0, dtype=pos.dtype, device=device)
        return empty_long, empty_long, empty_float, None

    src = torch.cat(edge_src).to(device=device)
    dst = torch.cat(edge_dst).to(device=device)
    distance = torch.cat(edge_dist).to(device=device)
    return src, dst, distance, None


def _apply_mic(
    vectors: torch.Tensor,
    cell: torch.Tensor,
    pbc: torch.Tensor,
) -> torch.Tensor:
    inv_cell = torch.linalg.inv(cell)
    frac_vectors = vectors @ inv_cell
    shifts = -torch.floor(frac_vectors + 0.5)
    shifts = torch.where(pbc.unsqueeze(0), shifts, torch.zeros_like(shifts))
    return vectors + shifts @ cell
