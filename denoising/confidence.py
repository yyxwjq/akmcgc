"""Graph-level confidence predictor built from the denoising encoder stack."""
from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import nn, Tensor
from torch_scatter import scatter_mean

try:
    from ..model import EGNN
    from ..model.core import GatedMLP
except ImportError:  # pragma: no cover
    from model import EGNN
    from model.core import GatedMLP

from .base_denoiser import BaseDenoiser


class ConfidencePredictor(BaseDenoiser):
    """Graph-level confidence head built on the same equivariant encoder stack.

    Unlike ``Denoiser``, this module does not predict diffusion noise. It runs
    the graph through EGNN/LEFTNet, mean-pools node features by ``mask``, and
    predicts one scalar confidence per reaction sample.
    """

    def __init__(
        self,
        model_config: Dict,
        fragment_names,
        node_nfs,
        edge_nf: int,
        condition_nf: int = 0,
        pos_dim: int = 3,
        edge_cutoff: Optional[float] = None,
        model: nn.Module = EGNN,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype | str = torch.float64,
        enforce_same_encoding=None,
        source: Optional[Dict] = None,
        **kwargs,
    ) -> None:
        model_config = dict(model_config)
        model_config.update({"for_conf": True})
        super().__init__(
            model_config=model_config,
            fragment_names=fragment_names,
            node_nfs=node_nfs,
            edge_nf=edge_nf,
            condition_nf=condition_nf,
            pos_dim=pos_dim,
            update_pocket_coords=True,
            condition_time=True,
            edge_cutoff=edge_cutoff,
            model=model,
            device=device,
            dtype=dtype,
            enforce_same_encoding=enforce_same_encoding,
            source=source,
        )

        readout_in_dim = int(
            model_config.get("hidden_channels", model_config.get("in_node_nf", self.model_input_dim))
        )
        self.readout = GatedMLP(
            in_dim=readout_in_dim,
            out_dims=[readout_in_dim, readout_in_dim, 1],
            activation="swish",
            bias=True,
            last_layer_no_activation=True,
        )
        self.readout.to(dtype=self.dtype)

    def forward(
        self,
        batch: Dict,
        conditions: Optional[Tensor] = None,
    ) -> Tensor:
        """Return one confidence score per graph/sample in the batch."""
        h = batch["h"]
        # Reuse the same feature encoder and time-conditioning pathway as the
        # Denoiser. Confidence uses t=0 as a neutral fixed condition.
        pos = h[:, : self.pos_dim].clone()
        hidden = self.encode_node_features(h)
        hidden, _ = self.augment_with_conditions(
            hidden,
            t=torch.zeros(
                (int(batch["mask"].max().item()) + 1, 1),
                device=h.device,
                dtype=h.dtype,
            ),
            mask=batch["mask"],
            conditions=conditions,
        )

        model_out = self.model(
            hidden,
            pos,
            batch["edge_index"],
            edge_attr=None,
            node_mask=None,
            edge_mask=None,
            update_coords_mask=None,
            subgraph_mask=None,
            cell=batch["cell"],
            pbc=batch["pbc"],
            cell_offsets=batch["cell_offsets"],
            neighbors=batch.get("neighbors"),
            fragment=batch["fragment"],
            mask=batch["mask"],
        )

        if isinstance(model_out, tuple):
            node_features = model_out[0]
        else:
            node_features = model_out

        # Pool nodes into graph-level features before the final readout.
        graph_features = scatter_mean(node_features, index=batch["mask"], dim=0)
        return self.readout(graph_features).squeeze(-1)
