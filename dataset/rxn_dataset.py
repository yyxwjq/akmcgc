"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
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


class RxnDataset(Dataset):
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
        """Get one reaction pair as a joint reactant+product graph."""
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
        edge_index = torch.cat([edge_is, edge_fs + n_is], dim=1)
        cell_offsets = torch.cat([cell_offsets_is, cell_offsets_fs], dim=0)

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
            "mask": torch.zeros(n_total, dtype=torch.long, device=self.device),
            "cell": torch.stack([cell_is, cell_fs], dim=0),
            "pbc": torch.stack([pbc_is, pbc_fs], dim=0),
            "n_is": n_is,
            "n_fs": n_fs,
        }
        if self.return_distances:
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
        """Collate joint reaction graphs into one disconnected batch graph."""
        if len(batch) == 0:
            raise ValueError("Cannot collate an empty batch")

        device = batch[0]["h"].device
        n_atoms_per_sample = [sample["n_is"] + sample["n_fs"] for sample in batch]

        edge_index_list = []
        cell_offsets_list = []
        offset = 0
        for sample in batch:
            edge_index_list.append(sample["edge_index"] + offset)
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


RPDataset = RxnDataset
RP_Dataset = RxnDataset
