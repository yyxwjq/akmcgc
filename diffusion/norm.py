from __future__ import annotations

from typing import Dict, List, Tuple

import torch
from torch import Tensor, nn

FEATURE_MAPPING = ["pos", "one_hot", "charge"]


class Norm(nn.Module):
    def __init__(
        self,
        norm_values: Tuple = (1.0, 1.0, 1.0),
        norm_biases: Tuple = (0.0, 0.0, 0.0),
        pos_dim: int = 3,
    ) -> None:
        super().__init__()
        self.norm_values = norm_values
        self.norm_biases = norm_biases
        self.pos_dim = pos_dim

    def split_h(self, h: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        pos = h[:, : self.pos_dim]
        charge = h[:, -1:]
        one_hot = h[:, self.pos_dim : -1]
        return pos, one_hot, charge

    def join_h(self, pos: Tensor, one_hot: Tensor, charge: Tensor) -> Tensor:
        return torch.cat([pos, one_hot, charge], dim=1)

    def normalize_batch(self, batch: Dict) -> Dict:
        out = dict(batch)
        h = batch["h"].clone()
        pos, one_hot, charge = self.split_h(h)
        pos = (pos - self.norm_biases[0]) / self.norm_values[0]
        one_hot = (one_hot - self.norm_biases[1]) / self.norm_values[1]
        charge = (charge - self.norm_biases[2]) / self.norm_values[2]
        out["h"] = self.join_h(pos, one_hot, charge)
        out["pos"] = out["h"][:, : self.pos_dim].clone()
        return out

    def normalize_legacy(self, representations: List[Dict]) -> List[Dict]:
        for ii in range(len(representations)):
            for jj, feature_type in enumerate(FEATURE_MAPPING):
                representations[ii][feature_type] = (
                    representations[ii][feature_type] - self.norm_biases[jj]
                ) / self.norm_values[jj]
        return representations

    def normalize(self, inputs):
        if isinstance(inputs, dict):
            return self.normalize_batch(inputs)
        return self.normalize_legacy(inputs)

    def unnormalize(self, x: Tensor, ind: int) -> Tensor:
        return x * self.norm_values[ind] + self.norm_biases[ind]

    def unnormalize_h(self, h: Tensor) -> Tensor:
        pos, one_hot, charge = self.split_h(h)
        pos = self.unnormalize(pos, 0)
        one_hot = self.unnormalize(one_hot, 1)
        charge = self.unnormalize(charge, 2)
        return self.join_h(pos, one_hot, charge)

    def unnormalize_z(self, z_combined):
        if isinstance(z_combined, Tensor):
            return self.unnormalize_h(z_combined)
        for ii in range(len(z_combined)):
            z_combined[ii][:, : self.pos_dim] = self.unnormalize(
                z_combined[ii][:, : self.pos_dim], 0
            )
            z_combined[ii][:, self.pos_dim : -1] = self.unnormalize(
                z_combined[ii][:, self.pos_dim : -1], 1
            )
            z_combined[ii][:, -1:] = self.unnormalize(z_combined[ii][:, -1:], 2)
        return z_combined


GraphNorm = Norm
Normalizer = Norm
