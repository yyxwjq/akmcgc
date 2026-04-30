"""Base class for joint-graph denoising network wrappers."""
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
        self.model = model(**model_config)
        self.model.to(dtype=self.dtype)
        if source is not None and "model" in source:
            self.model.load_state_dict(source["model"])

        self.dist_dim = getattr(self.model, "dist_dim", 0)
        self.model_input_dim = int(model_config["in_node_nf"])

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
        return self.node_encoder(h[:, self.pos_dim :].clone())

    def decode_node_features(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.node_decoder(hidden)

    def augment_with_conditions(
        self,
        hidden: torch.Tensor,
        t: torch.Tensor,
        mask: torch.Tensor,
        conditions: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, int]:
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
