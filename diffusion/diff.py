from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch
from torch import Tensor, nn

try:
    from . import diff_utils as utils
    from .norm import Norm
    from .diff_sched import DiffSchedule
except ImportError:  # pragma: no cover
    import diffusion.diff_utils as utils
    from diffusion.norm import Norm
    from diffusion.diff_sched import DiffSchedule


class Diffusion(nn.Module):
    """Joint-graph E(n) diffusion over periodic reaction-pair graphs."""

    def __init__(
        self,
        denoiser: Optional[nn.Module] = None,
        schdule: Optional[DiffSchedule] = None,
        normalizer: Optional[Norm] = None,
        size_histogram: Optional[Dict] = None,
        loss_type: str = "l2",
        pos_only: bool = False,
        fixed_idx: Optional[List] = None,
    ) -> None:
        super().__init__()
        assert loss_type in {"vlb", "l2"}
        if denoiser is None:
            raise ValueError("Diffusion requires a denoiser network")
        if schdule is None or normalizer is None:
            raise ValueError("Diffusion requires both schdule and normalizer")

        self.denoiser = denoiser
        self.schedule = schdule
        self.normalizer = normalizer
        self.size_histogram = size_histogram
        self.loss_type = loss_type
        self.pos_only = pos_only
        self.fixed_idx = fixed_idx or []

        self.pos_dim = denoiser.pos_dim
        self.node_nf = denoiser.node_nf
        self.T = schdule.gamma_module.timesteps
        self.norm_values = normalizer.norm_values
        self.norm_biases = normalizer.norm_biases

    def _num_graphs(self, mask: Tensor) -> int:
        return int(mask.max().item()) + 1 if mask.numel() > 0 else 0

    def _node_counts(self, mask: Tensor) -> Tensor:
        return torch.bincount(mask, minlength=self._num_graphs(mask))

    def forward(
        self,
        batch: Dict,
        conditions: Optional[Tensor] = None,
        return_pred: bool = False,
    ) -> Dict:
        batch = self.normalizer.normalize_batch(batch)
        h = batch["h"]
        mask = batch["mask"]
        num_graphs = self._num_graphs(mask)
        n_nodes = self._node_counts(mask)
        device = h.device

        delta_log_px = self.delta_log_px(n_nodes).to(dtype=h.dtype)

        lowest_t = 0 if self.training else 1
        t_int = torch.randint(
            lowest_t,
            self.T + 1,
            size=(num_graphs, 1),
            device=device,
        ).to(dtype=h.dtype)
        s_int = t_int - 1
        t_is_zero = (t_int == 0).float().squeeze(1)
        t_is_not_zero = 1.0 - t_is_zero

        s = s_int / self.T
        t = t_int / self.T

        gamma_s = self.schedule.inflate_batch_array(self.schedule.gamma_module(s), h)
        gamma_t = self.schedule.inflate_batch_array(self.schedule.gamma_module(t), h)

        z_t, eps_xh = self.noised_representation(h, mask, gamma_t)
        net_eps_xh, _ = self.denoiser(
            h=z_t,
            edge_index=batch["edge_index"],
            t=t,
            conditions=conditions,
            mask=mask,
            fragment=batch["fragment"],
            cell=batch["cell"],
            pbc=batch["pbc"],
            cell_offsets=batch["cell_offsets"],
            neighbors=batch.get("neighbors"),
            edge_attr=None,
        )

        if return_pred:
            return {"eps_xh": eps_xh, "net_eps_xh": net_eps_xh}

        if self.pos_only:
            net_eps_xh = net_eps_xh.clone()
            net_eps_xh[:, self.pos_dim :] = 0.0

        coord_error = utils.sum_except_batch(
            (eps_xh[:, : self.pos_dim] - net_eps_xh[:, : self.pos_dim]) ** 2,
            mask,
            dim_size=num_graphs,
        )
        coord_error = coord_error * t_is_not_zero
        feature_error = torch.zeros_like(coord_error)

        x_pred = self.compute_x_pred(net_eps_xh, z_t, gamma_t, mask)
        loss_0_x = utils.sum_except_batch(
            (x_pred[:, : self.pos_dim] - h[:, : self.pos_dim]) ** 2,
            mask,
            dim_size=num_graphs,
        ) * t_is_zero

        zeros = torch.zeros_like(loss_0_x)
        return {
            "delta_log_px": delta_log_px,
            "error_t": coord_error + feature_error,
            "SNR_weight": (1 - self.schedule.SNR(gamma_s - gamma_t)).squeeze(1),
            "loss_0_x": loss_0_x,
            "loss_0_cat": zeros,
            "loss_0_charge": zeros,
            "neg_log_constants": zeros,
            "kl_prior": zeros,
            "log_pN": zeros,
            "t_int": t_int.squeeze(1),
            "net_eps_xh": net_eps_xh,
            "eps_xh": eps_xh,
        }

    def delta_log_px(self, num_nodes: Tensor) -> Tensor:
        return -self.subspace_dimensionality(num_nodes).float() * np.log(self.norm_values[0])

    def subspace_dimensionality(self, input_size: Tensor) -> Tensor:
        return (input_size - 1) * self.pos_dim

    def noised_representation(
        self,
        h: Tensor,
        mask: Tensor,
        gamma_t: Tensor,
    ) -> tuple[Tensor, Tensor]:
        alpha_t = self.schedule.alpha(gamma_t, h)
        sigma_t = self.schedule.sigma(gamma_t, h)
        eps_xh = self.sample_combined_position_feature_noise(h, mask)
        z_t = alpha_t[mask] * h + sigma_t[mask] * eps_xh
        return z_t, eps_xh

    def sample_combined_position_feature_noise(self, h: Tensor, mask: Tensor) -> Tensor:
        eps_x = utils.sample_center_gravity_zero_gaussian_batch(
            size=(h.size(0), self.pos_dim),
            indices=[mask],
        ).to(dtype=h.dtype)
        feat_dim = h.size(1) - self.pos_dim
        if self.pos_only:
            eps_feat = torch.zeros((h.size(0), feat_dim), device=h.device, dtype=h.dtype)
        else:
            eps_feat = utils.sample_gaussian((h.size(0), feat_dim), device=h.device).to(h.dtype)
        eps_xh = torch.cat([eps_x, eps_feat], dim=1)
        for idx in self.fixed_idx:
            fixed_mask = h.new_zeros(h.size(0), dtype=torch.bool)
            fixed_mask[idx] = True
            eps_xh[fixed_mask] = 0
        return eps_xh

    def sample_normal(
        self,
        mu: Tensor,
        sigma: Tensor,
        mask: Tensor,
        fix_noise: bool = False,
    ) -> Tensor:
        del fix_noise
        eps_xh = self.sample_combined_position_feature_noise(mu, mask)
        return mu + sigma[mask] * eps_xh

    def compute_x_pred(
        self,
        net_eps_xh: Tensor,
        zt_xh: Tensor,
        gamma_t: Tensor,
        mask: Tensor,
    ) -> Tensor:
        sigma_t = self.schedule.sigma(gamma_t, target_tensor=net_eps_xh)
        alpha_t = self.schedule.alpha(gamma_t, target_tensor=net_eps_xh)
        return (zt_xh - sigma_t[mask] * net_eps_xh) / alpha_t[mask]

    def sample_p_zs_given_zt(
        self,
        batch: Dict,
        zt_xh: Tensor,
        s: Tensor,
        t: Tensor,
        conditions: Optional[Tensor] = None,
        fix_noise: bool = False,
    ) -> Tensor:
        gamma_s = self.schedule.gamma_module(s)
        gamma_t = self.schedule.gamma_module(t)
        sigma2_t_given_s, sigma_t_given_s, alpha_t_given_s = self.schedule.sigma_and_alpha_t_given_s(
            gamma_t, gamma_s, zt_xh
        )
        sigma_s = self.schedule.sigma(gamma_s, target_tensor=zt_xh)
        sigma_t = self.schedule.sigma(gamma_t, target_tensor=zt_xh)

        net_eps_xh, _ = self.denoiser(
            h=zt_xh,
            edge_index=batch["edge_index"],
            t=t,
            conditions=conditions,
            mask=batch["mask"],
            fragment=batch["fragment"],
            cell=batch["cell"],
            pbc=batch["pbc"],
            cell_offsets=batch["cell_offsets"],
            neighbors=batch.get("neighbors"),
            edge_attr=None,
        )
        if self.pos_only:
            net_eps_xh = net_eps_xh.clone()
            net_eps_xh[:, self.pos_dim :] = 0.0

        mu = zt_xh / alpha_t_given_s[batch["mask"]] - net_eps_xh * (
            sigma2_t_given_s / alpha_t_given_s / sigma_t
        )[batch["mask"]]
        sigma = sigma_t_given_s * sigma_s / sigma_t
        zs_xh = self.sample_normal(mu, sigma, batch["mask"], fix_noise=fix_noise)
        zs_xh[:, : self.pos_dim] = utils.remove_mean_batch(
            zs_xh[:, : self.pos_dim], batch["mask"]
        )
        return zs_xh

    def sample_p_xh_given_z0(
        self,
        batch: Dict,
        z0_xh: Tensor,
        conditions: Optional[Tensor] = None,
        fix_noise: bool = False,
    ) -> Tensor:
        n_graphs = self._num_graphs(batch["mask"])
        t_zeros = torch.zeros(size=(n_graphs, 1), device=z0_xh.device, dtype=z0_xh.dtype)
        gamma_0 = self.schedule.gamma_module(t_zeros)
        sigma_x = self.schedule.SNR(-0.5 * gamma_0)

        net_eps_xh, _ = self.denoiser(
            h=z0_xh,
            edge_index=batch["edge_index"],
            t=t_zeros,
            conditions=conditions,
            mask=batch["mask"],
            fragment=batch["fragment"],
            cell=batch["cell"],
            pbc=batch["pbc"],
            cell_offsets=batch["cell_offsets"],
            neighbors=batch.get("neighbors"),
            edge_attr=None,
        )
        if self.pos_only:
            net_eps_xh = net_eps_xh.clone()
            net_eps_xh[:, self.pos_dim :] = 0.0

        mu_x = self.compute_x_pred(net_eps_xh, z0_xh, gamma_0, batch["mask"])
        x0_xh = self.sample_normal(mu_x, sigma_x, batch["mask"], fix_noise=fix_noise)
        if self.pos_only:
            x0_xh[:, self.pos_dim :] = batch["h"][:, self.pos_dim :]
        return x0_xh

    @torch.no_grad()
    def sample(
        self,
        batch: Dict,
        conditions: Optional[Tensor] = None,
        return_frames: int = 1,
        timesteps: Optional[int] = None,
    ) -> Dict:
        del return_frames
        timesteps = self.T if timesteps is None else timesteps
        batch = self.normalizer.normalize_batch(batch)
        h0 = batch["h"].clone()
        zt_xh = self.sample_combined_position_feature_noise(h0, batch["mask"])
        if self.pos_only:
            zt_xh[:, self.pos_dim :] = h0[:, self.pos_dim :]

        for s in reversed(range(0, timesteps)):
            s_array = torch.full(
                (self._num_graphs(batch["mask"]), 1),
                fill_value=s,
                device=h0.device,
                dtype=h0.dtype,
            )
            t_array = s_array + 1
            s_array = s_array / timesteps
            t_array = t_array / timesteps
            zt_xh = self.sample_p_zs_given_zt(
                batch=batch,
                zt_xh=zt_xh,
                s=s_array,
                t=t_array,
                conditions=conditions,
                fix_noise=False,
            )
            if self.pos_only:
                zt_xh[:, self.pos_dim :] = h0[:, self.pos_dim :]

        xh = self.sample_p_xh_given_z0(batch, zt_xh, conditions=conditions)
        xh = self.normalizer.unnormalize_h(xh)
        out = dict(batch)
        out["h"] = xh
        out["pos"] = xh[:, : self.pos_dim].clone()
        return out

    @torch.no_grad()
    def inpaint(
        self,
        batch: Dict,
        conditions: Optional[Tensor] = None,
        return_frames: int = 1,
        resamplings: int = 1,
        jump_length: int = 1,
        timesteps: Optional[int] = None,
        frag_fixed: Optional[List[int]] = None,
    ) -> Dict:
        del return_frames, resamplings, jump_length
        timesteps = self.T if timesteps is None else timesteps
        frag_fixed = [0] if frag_fixed is None else frag_fixed

        batch = self.normalizer.normalize_batch(batch)
        h_fixed = batch["h"].clone()
        fixed_mask = torch.zeros_like(batch["fragment"], dtype=torch.bool)
        for idx in frag_fixed:
            fixed_mask |= batch["fragment"] == idx

        zt_xh = self.sample_combined_position_feature_noise(h_fixed, batch["mask"])
        if self.pos_only:
            zt_xh[:, self.pos_dim :] = h_fixed[:, self.pos_dim :]

        for s in reversed(range(0, timesteps)):
            s_array = torch.full(
                (self._num_graphs(batch["mask"]), 1),
                fill_value=s,
                device=h_fixed.device,
                dtype=h_fixed.dtype,
            )
            t_array = s_array + 1
            s_array = s_array / timesteps
            t_array = t_array / timesteps
            zs_xh = self.sample_p_zs_given_zt(
                batch=batch,
                zt_xh=zt_xh,
                s=s_array,
                t=t_array,
                conditions=conditions,
                fix_noise=False,
            )
            gamma_s = self.schedule.inflate_batch_array(self.schedule.gamma_module(s_array), h_fixed)
            z_known, _ = self.noised_representation(h_fixed, batch["mask"], gamma_s)
            zs_xh[fixed_mask, : self.pos_dim] = z_known[fixed_mask, : self.pos_dim]
            zs_xh[fixed_mask, self.pos_dim :] = h_fixed[fixed_mask, self.pos_dim :]
            if self.pos_only:
                zs_xh[:, self.pos_dim :] = h_fixed[:, self.pos_dim :]
            zt_xh = zs_xh

        xh = self.sample_p_xh_given_z0(batch, zt_xh, conditions=conditions)
        xh[fixed_mask] = h_fixed[fixed_mask]
        xh = self.normalizer.unnormalize_h(xh)
        out = dict(batch)
        out["h"] = xh
        out["pos"] = xh[:, : self.pos_dim].clone()
        return out

    @torch.no_grad()
    def inpaint_fixed(self, *args, **kwargs) -> Dict:
        return self.inpaint(*args, **kwargs)


PBCDiffusion = Diffusion
EnVariationalDiffusion = Diffusion
