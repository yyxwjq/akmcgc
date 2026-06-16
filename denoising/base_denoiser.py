"""Shared machinery for denoising network wrappers.

``Denoiser`` is not the message-passing network itself. It wraps an equivariant
model such as EGNN/LEFTNet with:

1. a node feature encoder for ``h[:, pos_dim:]``;
2. optional time/condition concatenation;
3. a node feature decoder that maps hidden states back to predicted feature
   noise.

Coordinates stay in ``h[:, :pos_dim]`` and are passed directly to the equivariant
model.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import torch
from torch import nn

try:
    from ..model import MLP, EGNN
    from ..utils.dtype import resolve_float_dtype
except ImportError:  # pragma: no cover
    from model import MLP, EGNN
    from utils.dtype import resolve_float_dtype


class BaseDenoiser(nn.Module):
    def __init__(
        self,
        model_config: Dict,
        fragment_names: List[str],
        node_nfs: List[int] | int,
        edge_nf: int,
        condition_nf: int = 0,
        pos_dim: int = 3,
        update_pocket_coords: bool = True,
        condition_time: bool = True,
        edge_cutoff: Optional[float] = None,
        model: nn.Module = EGNN,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype | str = torch.float64,
        enforce_same_encoding: Optional[List] = None,
        source: Optional[Dict] = None,
    ) -> None:
        super().__init__()
        del enforce_same_encoding

        # node_nfs can be a single integer for the current joint-graph path or a
        # list for compatibility with older multi-fragment configs.
        if isinstance(node_nfs, int):
            node_nfs = [node_nfs]
        if len(node_nfs) == 0:
            raise ValueError("node_nfs must contain at least one feature dimension")

        self.node_nfs = node_nfs
        self.node_nf = int(node_nfs[0])
        if self.node_nf <= pos_dim:
            raise ValueError(f"node_nf must be > pos_dim, got {self.node_nf} and {pos_dim}")

        if "act_fn" not in model_config:
            model_config["act_fn"] = "swish"
        if "in_node_nf" not in model_config and "in_hidden_channels" not in model_config:
            raise KeyError("model_config must define `in_node_nf` or `in_hidden_channels`")
        if "in_node_nf" not in model_config:
            model_config["in_node_nf"] = model_config["in_hidden_channels"]

        self.model_config = model_config
        self.edge_nf = edge_nf
        self.condition_nf = condition_nf
        self.fragment_names = fragment_names
        self.pos_dim = pos_dim
        self.update_pocket_coords = update_pocket_coords
        self.condition_time = condition_time
        self.edge_cutoff = edge_cutoff
        self.device = device
        self.dtype = resolve_float_dtype(dtype)

        if model is None:
            model = EGNN
        # The inner model sees encoded scalar features plus any conditioning
        # columns. It predicts updated hidden features and updated coordinates.
        self.model = model(**model_config)
        self.model.to(dtype=self.dtype)
        if source is not None and "model" in source:
            self.model.load_state_dict(source["model"])

        self.dist_dim = getattr(self.model, "dist_dim", 0)
        self.model_input_dim = int(model_config["in_node_nf"])

        # ``model_input_dim`` is the actual input feature dimension expected by
        # EGNN/LEFTNet. Reserve columns for time/global conditions; the remaining
        # columns are produced by ``node_encoder``.
        self.embed_dim = self.model_input_dim
        if self.condition_time:
            self.embed_dim -= 1
        if self.condition_nf > 0:
            self.embed_dim -= self.condition_nf
        if self.embed_dim <= 0:
            raise ValueError("model input dimension is too small after conditioning")

        self.build_encoders_decoders(source)
        self.to(dtype=self.dtype)

    def build_encoders_decoders(self, source: Optional[Dict] = None) -> None:
        """Build feature encoders/decoders around the equivariant model.

        ``feat_dim = node_nf - pos_dim`` because position is handled separately.
        For the default data contract, ``node_nf=122`` and ``pos_dim=3``, so
        scalar/node feature dimension is 119.
        """
        feat_dim = self.node_nf - self.pos_dim
        self.node_encoder = MLP(
            in_dim=feat_dim,
            out_dims=[2 * feat_dim, self.embed_dim],
            activation=self.model_config["act_fn"],
            last_layer_no_activation=True,
        )
        self.node_decoder = MLP(
            in_dim=self.embed_dim,
            out_dims=[2 * feat_dim, feat_dim],
            activation=self.model_config["act_fn"],
            last_layer_no_activation=True,
        )

        if source is not None:
            if "encoders" in source:
                try:
                    self.node_encoder.load_state_dict(source["encoders"])
                except RuntimeError:
                    pass
            if "decoders" in source:
                try:
                    self.node_decoder.load_state_dict(source["decoders"])
                except RuntimeError:
                    pass

        # Edge encoders are only used when explicit edge features are supplied.
        # The current debug/training path usually uses edge_nf=0 and lets EGNN
        # construct distance-based edge features internally.
        self.edge_embed_dim = int(self.model_config.get("in_edge_nf", 0))
        if self.edge_nf > 0 and self.edge_embed_dim > 0:
            self.edge_encoder = MLP(
                in_dim=self.edge_nf,
                out_dims=[2 * self.edge_nf, self.edge_embed_dim],
                activation=self.model_config["act_fn"],
                last_layer_no_activation=True,
            )
            self.edge_decoder = MLP(
                in_dim=self.edge_embed_dim + self.dist_dim,
                out_dims=[2 * self.edge_nf, self.edge_nf],
                activation=self.model_config["act_fn"],
                last_layer_no_activation=True,
            )
        else:
            self.edge_encoder = None
            self.edge_decoder = None

    def encode_node_features(self, h: torch.Tensor) -> torch.Tensor:
        """Encode only non-coordinate node features from ``h``."""
        return self.node_encoder(h[:, self.pos_dim :].clone())

    def decode_node_features(self, hidden: torch.Tensor) -> torch.Tensor:
        """Decode hidden node states back to predicted feature noise."""
        return self.node_decoder(hidden)

    def augment_with_conditions(
        self,
        hidden: torch.Tensor,
        t: torch.Tensor,
        mask: torch.Tensor,
        conditions: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, int]:
        """Append sample-level time/global conditions to each node.

        ``t`` and ``conditions`` are shaped by graph/sample, not by node. ``mask``
        broadcasts them to nodes: all atoms from the same reaction sample receive
        the same time condition.
        """
        condition_dim = 0
        if self.condition_time:
            if t.dim() == 0:
                h_time = torch.full_like(hidden[:, :1], float(t.item()))
            elif t.dim() == 1:
                h_time = torch.full_like(hidden[:, :1], float(t[0].item()))
            else:
                h_time = t[mask]
            hidden = torch.cat([hidden, h_time], dim=1)
            condition_dim += 1

        if self.condition_nf > 0:
            if conditions is None:
                raise ValueError("conditions must be provided when condition_nf > 0")
            if conditions.dim() == 1:
                conditions = conditions.unsqueeze(1)
            hidden = torch.cat([hidden, conditions[mask]], dim=1)
            condition_dim += self.condition_nf
        return hidden, condition_dim

    def forward(self, *args, **kwargs):
        raise NotImplementedError
