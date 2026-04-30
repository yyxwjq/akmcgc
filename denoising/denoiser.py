from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
from torch import nn, Tensor

try:
    from ..model import EGNN
    from ..utils.tensor import remove_mean_batch
except ImportError:  # pragma: no cover
    from model import EGNN
    from utils.tensor import remove_mean_batch

from .base_denoiser import BaseDenoiser


class Denoiser(BaseDenoiser):
    """Predict diffusion noise on a periodic joint graph.

    This wrapper encodes raw node features, calls an equivariant model
    such as EGNN or LEFTNet, and decodes the result into ``eps_h``.
    """

    def __init__(
        self,
        model_config: Dict,
        fragment_names,
        node_nfs,
        edge_nf: int,
        condition_nf: int = 0,
        pos_dim: int = 3,
        update_pocket_coords: bool = True,
        condition_time: bool = True,
        edge_cutoff: Optional[float] = None,
        model: nn.Module = EGNN,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype | str = torch.float64,
        enforce_same_encoding=None,
        source: Optional[Dict] = None,
    ) -> None:
        super().__init__(
            model_config=model_config,
            fragment_names=fragment_names,
            node_nfs=node_nfs,
            edge_nf=edge_nf,
            condition_nf=condition_nf,
            pos_dim=pos_dim,
            update_pocket_coords=update_pocket_coords,
            condition_time=condition_time,
            edge_cutoff=edge_cutoff,
            model=model,
            device=device,
            dtype=dtype,
            enforce_same_encoding=enforce_same_encoding,
            source=source,
        )

    def forward(
        self,
        h: Tensor,
        edge_index: Tensor,
        t: Tensor,
        mask: Tensor,
        fragment: Tensor,
        cell: Tensor,
        pbc: Tensor,
        cell_offsets: Tensor,
        neighbors: Optional[Tensor] = None,
        conditions: Optional[Tensor] = None,
        edge_attr: Optional[Tensor] = None,
        return_hidden: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]] | Tuple[Tensor, Optional[Tensor], Tensor]:
        pos = h[:, : self.pos_dim].clone()
        hidden = self.encode_node_features(h)

        if self.edge_encoder is not None and edge_attr is not None:
            edge_attr = self.edge_encoder(edge_attr)

        hidden, condition_dim = self.augment_with_conditions(hidden, t, mask, conditions)

        update_coords_mask = None if self.update_pocket_coords else None
        model_out = self.model(
            hidden,
            pos,
            edge_index,
            edge_attr,
            node_mask=None,
            edge_mask=None,
            update_coords_mask=update_coords_mask,
            subgraph_mask=None,
            cell=cell,
            pbc=pbc,
            cell_offsets=cell_offsets,
            neighbors=neighbors,
            fragment=fragment,
            mask=mask,
        )

        if isinstance(model_out, tuple):
            hidden_out = model_out[0]
            pos_out = model_out[1]
            edge_attr_out = model_out[2] if len(model_out) > 2 else None
        else:
            hidden_out = model_out
            pos_out = pos
            edge_attr_out = None

        if condition_dim > 0 and hidden_out.size(1) >= condition_dim:
            hidden_out = hidden_out[:, :-condition_dim]

        vel = pos_out - pos
        if torch.any(torch.isnan(vel)):
            vel = torch.randn_like(vel)
        if torch.any(torch.isnan(hidden_out)):
            hidden_out = torch.randn_like(hidden_out)

        eps_h = torch.zeros_like(h)
        eps_h[:, : self.pos_dim] = remove_mean_batch(vel, mask)
        eps_h[:, self.pos_dim :] = self.decode_node_features(hidden_out)

        if (
            edge_attr_out is not None
            and self.edge_decoder is not None
            and edge_attr_out.size(1) > max(1, self.dist_dim)
        ):
            edge_attr_out = self.edge_decoder(edge_attr_out)
        else:
            edge_attr_out = None

        if return_hidden:
            return eps_h, edge_attr_out, hidden_out
        return eps_h, edge_attr_out
