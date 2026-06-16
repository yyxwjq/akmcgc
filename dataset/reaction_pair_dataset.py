"""Dataset utilities for periodic reaction-pair joint graphs.

Read this file first when trying to understand how raw ``extxyz`` structures
become the tensors used by EGNN/Denoiser/Diffusion. A single dataset item is a
joint graph containing both the reactant/initial state and product/final state:

```
nodes = [reactant atoms, product atoms]
fragment = 0 for reactant nodes, 1 for product nodes
mask = batch id, filled during ``collate_fn``
```

The graph intentionally contains only intra-fragment edges. Reactant edges and
product edges are generated separately, then concatenated into one disconnected
joint graph.
"""
from pathlib import Path
from typing import Any, Dict, List

import ase.io
import torch
from torch.utils.data import Dataset

try:
    from ..utils.graph import atoms_to_tensors, get_pbc_distances, radius_graph_ase
    from ..utils.dtype import resolve_float_dtype
except ImportError:  # pragma: no cover - allows top-level local imports in notebooks
    from utils.graph import atoms_to_tensors, get_pbc_distances, radius_graph_ase
    from utils.dtype import resolve_float_dtype


class ReactionPairDataset(Dataset):
    """Reaction-pair dataset that returns one joint graph per sample.

    Reactant/initial-state atoms and product/final-state atoms are concatenated
    into a single node set. The graph is sparse and cutoff-based, but only
    contains intra-fragment edges. ``fragment`` identifies IS/FS membership
    while ``mask`` identifies batch membership after collation. Node feature
    ``h`` follows the order ``[pos, one_hot, charge]``. Periodic graph
    metadata includes `cell_offsets` and `neighbors` in an AdsorbDiff-like style.
    """

    def __init__(
        self,
        react_file: str,
        product_file: str,
        cutoff: float = 6.0,
        max_neigh: int = 200,
        r_fixed: bool = True,
        r_pbc: bool = True,
        device: str = "cpu",
        num_elements: int = 118,
        return_distances: bool = False,
        dtype: torch.dtype | str = torch.float64,
    ) -> None:
        """Store file paths and eagerly read all matching reaction frames.

        Args:
            react_file: Reactant/initial-state trajectory file.
            product_file: Product/final-state trajectory file with same frame count.
            cutoff: Neighbor cutoff passed to ASE neighbor list.
            max_neigh: Maximum outgoing neighbors per atom after distance sorting.
            r_pbc: Whether to keep periodic boundary conditions while building edges.
            return_distances: Debug option. Training normally recomputes distances
                from ``pos/cell/cell_offsets`` instead of storing static values.
            dtype: Floating dtype for ``h/pos/cell/cell_offsets``.
        """
        self.react_file = Path(react_file)
        self.product_file = Path(product_file)
        self.cutoff = cutoff
        self.max_neigh = max_neigh
        self.r_fixed = r_fixed
        self.r_pbc = r_pbc
        self.device = torch.device(device)
        self.num_elements = num_elements
        self.return_distances = return_distances
        self.dtype = resolve_float_dtype(dtype)

        if not self.react_file.exists():
            raise FileNotFoundError(f"Reactant file not found: {react_file}")
        if not self.product_file.exists():
            raise FileNotFoundError(f"Product file not found: {product_file}")

        self.react_traj = ase.io.read(self.react_file, index=":")
        self.product_traj = ase.io.read(self.product_file, index=":")

        if len(self.react_traj) != len(self.product_traj):
            raise ValueError(
                f"Frames mismatch: {len(self.react_traj)} reactant vs "
                f"{len(self.product_traj)} product"
            )

        self.n_frames = len(self.react_traj)

    def __len__(self) -> int:
        return self.n_frames

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get one reaction pair as a joint reactant+product graph.

        Returns a dictionary with the contract used throughout the model:

        ``h``: ``[N, 122]`` node state, ordered as ``[pos, one_hot, charge]``.
        ``pos``: ``[N, 3]`` coordinates, duplicated from ``h[:, :3]``.
        ``edge_index``: ``[2, E]`` sparse directed edges.
        ``cell_offsets``: ``[E, 3]`` periodic image offsets from ASE.
        ``fragment``: ``[N]`` reactant/product id inside this reaction sample.
        ``mask``: ``[N]`` batch id. For one sample it is all zero.
        ``cell``: ``[2, 3, 3]`` reactant/product cells.
        ``pbc``: ``[2, 3]`` reactant/product PBC flags.
        """
        react_atoms = self.react_traj[idx].copy()
        product_atoms = self.product_traj[idx].copy()

        if not self.r_pbc:
            react_atoms.set_pbc(False)
            product_atoms.set_pbc(False)

        h_is, pos_is, cell_is, pbc_is = atoms_to_tensors(
            react_atoms,
            device=self.device,
            num_elements=self.num_elements,
            dtype=self.dtype,
        )
        h_fs, pos_fs, cell_fs, pbc_fs = atoms_to_tensors(
            product_atoms,
            device=self.device,
            num_elements=self.num_elements,
            dtype=self.dtype,
        )

        n_is = h_is.size(0)
        n_fs = h_fs.size(0)
        n_total = n_is + n_fs

        edge_is, cell_offsets_is = radius_graph_ase(
            react_atoms,
            cutoff=self.cutoff,
            max_neigh=self.max_neigh,
            use_pbc=self.r_pbc,
            device=self.device,
            dtype=self.dtype,
        )
        edge_fs, cell_offsets_fs = radius_graph_ase(
            product_atoms,
            cutoff=self.cutoff,
            max_neigh=self.max_neigh,
            use_pbc=self.r_pbc,
            device=self.device,
            dtype=self.dtype,
        )
        # Product node ids start after all reactant nodes, so product edges need
        # an offset before both edge sets can live in the same joint graph.
        edge_index = torch.cat([edge_is, edge_fs + n_is], dim=1)
        cell_offsets = torch.cat([cell_offsets_is, cell_offsets_fs], dim=0)

        # fragment is local reaction identity, not batch identity:
        # 0 = initial/reactant, 1 = final/product.
        fragment = torch.cat(
            [
                torch.zeros(n_is, dtype=torch.long, device=self.device),
                torch.ones(n_fs, dtype=torch.long, device=self.device),
            ],
            dim=0,
        )

        sample = {
            "h": torch.cat([h_is, h_fs], dim=0),
            "pos": torch.cat([pos_is, pos_fs], dim=0),
            "edge_index": edge_index,
            "cell_offsets": cell_offsets,
            "neighbors": torch.tensor([edge_index.size(1)], dtype=torch.long, device=self.device),
            "fragment": fragment,
            # A single item is one graph, so mask is all zeros. collate_fn will
            # rewrite this field to 0, 1, 2, ... for a real batch.
            "mask": torch.zeros(n_total, dtype=torch.long, device=self.device),
            "cell": torch.stack([cell_is, cell_fs], dim=0),
            "pbc": torch.stack([pbc_is, pbc_fs], dim=0),
            "n_is": n_is,
            "n_fs": n_fs,
        }
        if self.return_distances:
            # Optional debug path: store edge vectors/distances produced from the
            # same PBC convention used by the model.
            pbc_geometry = get_pbc_distances(
                pos=sample["pos"],
                edge_index=edge_index,
                cell=sample["cell"],
                cell_offsets=cell_offsets,
                pbc=sample["pbc"],
                fragment=fragment,
                return_distance_vec=True,
            )
            sample["edge_vec"] = pbc_geometry["distance_vec"]
            sample["edge_dist"] = pbc_geometry["distances"]
        return sample

    @staticmethod
    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Collate joint reaction graphs into one disconnected batch graph.

        Each dataset item is already a reaction-level joint graph:
        reactant nodes and product nodes are concatenated, but only
        intra-fragment edges are present. Collation adds a second, purely
        computational concatenation: all reaction graphs in the mini-batch are
        packed into one disconnected graph so the GNN can process the whole
        mini-batch in a single tensorized forward pass. No cross-sample edges
        are created here.

        ``edge_index`` is shifted by the cumulative atom count. ``fragment``
        stays local to each reaction sample, while ``mask`` records which batch
        item every node came from. Batched cells become ``[B, 2, 3, 3]`` so PBC
        code can select ``cell[mask[row], fragment[row]]`` per edge.
        """
        if len(batch) == 0:
            raise ValueError("Cannot collate an empty batch")

        device = batch[0]["h"].device
        n_atoms_per_sample = [sample["n_is"] + sample["n_fs"] for sample in batch]

        edge_index_list = []
        cell_offsets_list = []
        offset = 0
        for sample in batch:
            # Every sample is built with local node ids starting at zero. After
            # concatenating node tensors, later samples must point to their new
            # global node ids in the big disconnected batch graph.
            edge_index_list.append(sample["edge_index"] + offset)
            # PBC cell offsets are lattice-image shifts, not node ids, so they
            # are concatenated unchanged.
            cell_offsets_list.append(sample["cell_offsets"])
            offset += sample["n_is"] + sample["n_fs"]

        out = {
            "h": torch.cat([sample["h"] for sample in batch], dim=0),
            "pos": torch.cat([sample["pos"] for sample in batch], dim=0),
            "edge_index": torch.cat(edge_index_list, dim=1),
            "cell_offsets": torch.cat(cell_offsets_list, dim=0),
            "neighbors": torch.tensor(
                [sample["edge_index"].size(1) for sample in batch],
                dtype=torch.long,
                device=device,
            ),
            "fragment": torch.cat([sample["fragment"] for sample in batch], dim=0),
            "mask": torch.cat(
                [
                    torch.full(
                        (n_atoms,),
                        batch_index,
                        dtype=torch.long,
                        device=device,
                    )
                    for batch_index, n_atoms in enumerate(n_atoms_per_sample)
                ],
                dim=0,
            ),
            "cell": torch.stack([sample["cell"] for sample in batch], dim=0),
            "pbc": torch.stack([sample["pbc"] for sample in batch], dim=0),
            "n_is": torch.as_tensor(
                [sample["n_is"] for sample in batch],
                dtype=torch.long,
                device=device,
            ),
            "n_fs": torch.as_tensor(
                [sample["n_fs"] for sample in batch],
                dtype=torch.long,
                device=device,
            ),
        }
        if "edge_vec" in batch[0]:
            out["edge_vec"] = torch.cat([sample["edge_vec"] for sample in batch], dim=0)
            out["edge_dist"] = torch.cat([sample["edge_dist"] for sample in batch], dim=0)
        return out
