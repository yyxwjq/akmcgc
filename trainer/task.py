"""PyTorch Lightning training modules for diffusion and confidence tasks.

This file connects the lower-level pieces:

``ReactionPairDataset -> Denoiser/ConfidencePredictor -> Diffusion -> Lightning Trainer``.

Read ``DiffModule`` for denoising diffusion training, and ``ConfModule`` for a
graph-level confidence prediction task.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import copy
import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, StepLR
from torch.utils.data import DataLoader
from pytorch_lightning import LightningModule
from torchmetrics import MeanAbsoluteError, PearsonCorrCoef, SpearmanCorrCoef
from torchmetrics.classification import BinaryAUROC, BinaryAccuracy, BinaryF1Score, BinaryPrecision

try:
    from ..dataset import ReactionPairDataset
    from ..denoising import ConfidencePredictor, Denoiser
    from ..diffusion import DiffSchedule, Diffusion, Norm, PredefinedNoiseSchedule
    from ..trainer.metrics import average_over_batch_metrics, pretty_print
    from ..utils.dtype import resolve_float_dtype
    from ..utils.train_utils import Queue, get_grad_norm
except ImportError:  # pragma: no cover
    from dataset import ReactionPairDataset
    from denoising import ConfidencePredictor, Denoiser
    from diffusion import DiffSchedule, Diffusion, Norm, PredefinedNoiseSchedule
    from trainer.metrics import average_over_batch_metrics, pretty_print
    from utils.dtype import resolve_float_dtype
    from utils.train_utils import Queue, get_grad_norm


LR_SCHEDULER = {
    "cos": CosineAnnealingWarmRestarts,
    "step": StepLR,
}


def _build_dataset(training_config: Dict, split: str) -> ReactionPairDataset:
    """Construct a split-specific ``ReactionPairDataset`` from training_config paths."""
    react_key = f"{split}_react_file"
    product_key = f"{split}_product_file"
    if react_key not in training_config or product_key not in training_config:
        raise KeyError(f"training_config must define `{react_key}` and `{product_key}`")
    return ReactionPairDataset(
        react_file=training_config[react_key],
        product_file=training_config[product_key],
        cutoff=training_config.get("cutoff", 6.0),
        max_neigh=training_config.get("max_neigh", 200),
        r_fixed=training_config.get("r_fixed", True),
        r_pbc=training_config.get("r_pbc", True),
        device=training_config.get("device", "cpu"),
        num_elements=training_config.get("num_elements", 118),
        dtype=training_config.get("dtype", torch.float64),
    )


def _default_conditions(batch: Dict, condition_nf: int) -> Optional[torch.Tensor]:
    """Create zero graph-level conditions when a model expects condition columns."""
    if condition_nf <= 0:
        return None
    num_graphs = int(batch["mask"].max().item()) + 1
    return torch.zeros((num_graphs, condition_nf), dtype=batch["h"].dtype, device=batch["h"].device)


def _product_rmsd(batch: Dict, pred_pos: torch.Tensor, fragment_id: int = 1) -> np.ndarray:
    """Compute per-sample RMSD on the product/final-state fragment."""
    rmsds = []
    batch_size = int(batch["mask"].max().item()) + 1
    for batch_idx in range(batch_size):
        node_mask = (batch["mask"] == batch_idx) & (batch["fragment"] == fragment_id)
        if torch.count_nonzero(node_mask) == 0:
            rmsds.append(np.nan)
            continue
        target = batch["pos"][node_mask]
        pred = pred_pos[node_mask]
        rmsd = torch.sqrt(torch.mean(torch.sum((pred - target) ** 2, dim=1)))
        rmsds.append(float(rmsd.item()))
    return np.asarray(rmsds, dtype=float)


class DiffModule(LightningModule):
    """LightningModule that trains the denoising diffusion model."""

    def __init__(
        self,
        model_config: Dict,
        optimizer_config: Dict,
        training_config: Dict,
        node_nfs,
        edge_nf: int = 0,
        condition_nf: int = 0,
        fragment_names=None,
        pos_dim: int = 3,
        update_pocket_coords: bool = True,
        condition_time: bool = True,
        edge_cutoff: Optional[float] = None,
        norm_values=(1.0, 1.0, 1.0),
        norm_biases=(0.0, 0.0, 0.0),
        noise_schedule: str = "cosine",
        timesteps: int = 1000,
        precision: float = 1e-5,
        loss_type: str = "l2",
        pos_only: bool = True,
        process_type: Optional[str] = None,
        model: nn.Module = None,
        enforce_same_encoding=None,
        scales=None,
        eval_epochs: int = 20,
        source: Optional[Dict] = None,
        fixed_idx=None,
    ) -> None:
        super().__init__()
        _ = process_type
        fragment_names = fragment_names or ["IS", "FS"]
        scales = [1.0] if scales is None else scales
        dtype = resolve_float_dtype(training_config.get("dtype", torch.float64))

        # Build the neural noise predictor first. Diffusion will call this
        # Denoiser at every training/sampling timestep.
        denoiser = Denoiser(
            model_config=model_config,
            node_nfs=node_nfs,
            edge_nf=edge_nf,
            condition_nf=condition_nf,
            fragment_names=fragment_names,
            pos_dim=pos_dim,
            update_pocket_coords=update_pocket_coords,
            condition_time=condition_time,
            edge_cutoff=edge_cutoff,
            model=model,
            enforce_same_encoding=enforce_same_encoding,
            source=source,
            dtype=dtype,
        )

        # Norm owns the affine scaling for h=[pos, one_hot, charge].
        norm = Norm(
            norm_values=norm_values,
            norm_biases=norm_biases,
            pos_dim=pos_dim,
        )
        # Training schedule controls alpha_t/sigma_t for forward diffusion.
        gamma_module = PredefinedNoiseSchedule(
            noise_schedule=noise_schedule,
            timesteps=timesteps,
            precision=precision,
        )
        schedule = DiffSchedule(gamma_module=gamma_module, norm_values=norm_values)

        self.ddpm = Diffusion(
            denoiser=denoiser,
            schdule=schedule,
            normalizer=norm,
            size_histogram=None,
            loss_type=loss_type,
            pos_only=pos_only,
            fixed_idx=fixed_idx,
        )

        self.model_config = model_config
        self.optimizer_config = optimizer_config
        self.training_config = training_config
        # LightningModule already owns a ``dtype`` property through nn.Module.
        # Keep the dataset/model dtype under a project-specific name.
        self.data_dtype = dtype
        self.loss_type = loss_type
        self.pos_only = pos_only
        self.condition_nf = condition_nf
        self.scales = scales

        # Sampling can use fewer timesteps than training for faster evaluation.
        sampling_gamma_module = PredefinedNoiseSchedule(
            noise_schedule=noise_schedule,
            timesteps=training_config.get("sampling_timesteps", 150),
            precision=precision,
        )
        self.sampling_schedule = DiffSchedule(
            gamma_module=sampling_gamma_module,
            norm_values=norm_values,
        )
        self.eval_epochs = eval_epochs

        self.clip_grad = training_config.get("clip_grad", False)
        if self.clip_grad:
            self.gradnorm_queue = Queue()
            self.gradnorm_queue.add(3000)
        self._train_epoch_outputs = []
        self._val_epoch_outputs = []
        self.save_hyperparameters(ignore=["model"])

    def configure_optimizers(self):
        """Create optimizer and optional LR scheduler for Lightning."""
        optimizer = torch.optim.AdamW(self.ddpm.parameters(), **self.optimizer_config)
        schedule_type = self.training_config.get("lr_schedule_type")
        if schedule_type is not None:
            scheduler_func = LR_SCHEDULER[schedule_type]
            scheduler = scheduler_func(
                optimizer=optimizer, **self.training_config["lr_schedule_config"]
            )
            return [optimizer], [scheduler]
        return optimizer

    def setup(self, stage: Optional[str] = None):
        """Create datasets lazily when Lightning enters fit/test stages."""
        if stage in ("fit", None):
            self.train_dataset = _build_dataset(self.training_config, "train")
            self.val_dataset = _build_dataset(self.training_config, "val")
        if stage in ("test", None):
            self.test_dataset = _build_dataset(self.training_config, "test")
        elif stage not in ("fit", "test", None):
            raise NotImplementedError

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            self.training_config["bz"],
            shuffle=True,
            num_workers=self.training_config.get("num_workers", 0),
            collate_fn=self.train_dataset.collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            self.training_config["bz"],
            shuffle=False,
            num_workers=self.training_config.get("num_workers", 0),
            collate_fn=self.val_dataset.collate_fn,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            self.training_config["bz"],
            shuffle=False,
            num_workers=self.training_config.get("num_workers", 0),
            collate_fn=self.test_dataset.collate_fn,
        )

    def compute_loss(self, batch):
        """Run Diffusion.forward and combine returned terms into one loss."""
        conditions = _default_conditions(batch, self.condition_nf)
        loss_terms = self.ddpm.forward(batch, conditions)
        num_nodes = (batch["n_is"] + batch["n_fs"]).float()

        # Normalize coordinate error by coordinate dimensionality and graph size
        # so graphs with more atoms do not automatically dominate the loss.
        error_t_normalized = loss_terms["error_t"] / (self.ddpm.pos_dim * num_nodes)
        if self.loss_type == "l2" and self.training:
            loss_t = error_t_normalized
            loss_0_x = loss_terms["loss_0_x"] / (self.ddpm.pos_dim * num_nodes)
            loss_0_cat = loss_terms["loss_0_cat"]
            loss_0_charge = loss_terms["loss_0_charge"]
            loss_0 = loss_0_x + loss_0_cat + loss_0_charge
        else:
            loss_t = -self.ddpm.T * 0.5 * loss_terms["SNR_weight"] * loss_terms["error_t"]
            loss_0_x = loss_terms["loss_0_x"]
            loss_0_cat = loss_terms["loss_0_cat"]
            loss_0_charge = loss_terms["loss_0_charge"]
            loss_0 = loss_0_x + loss_0_cat + loss_0_charge + loss_terms["neg_log_constants"]

        nll = loss_t + loss_0 + loss_terms["kl_prior"]
        if not (self.loss_type == "l2" and self.training):
            nll = nll - loss_terms["delta_log_px"] - loss_terms["log_pN"]

        info = {
            "coord_loss": error_t_normalized.mean().item(),
            "loss_0_x": loss_0_x.mean().item(),
            "loss_0_cat": loss_0_cat.mean().item(),
            "loss_0_charge": loss_0_charge.mean().item(),
        }
        return nll, info

    def eval_inplaint_batch(
        self,
        batch: Dict,
        resamplings: int = 1,
        jump_length: int = 1,
        frag_fixed=None,
    ):
        """Run inpainting on a batch and report product-fragment RMSD."""
        del resamplings, jump_length
        frag_fixed = [0] if frag_fixed is None else frag_fixed
        # Use a copy so evaluation can swap to a shorter sampling schedule
        # without mutating the training object.
        sampling_ddpm = copy.deepcopy(self.ddpm)
        sampling_ddpm.schedule = self.sampling_schedule
        sampling_ddpm.T = self.sampling_schedule.gamma_module.timesteps
        sampling_ddpm.eval()

        conditions = _default_conditions(batch, self.condition_nf)
        with torch.no_grad():
            out_batch = sampling_ddpm.inpaint(
                batch=batch,
                conditions=conditions,
                frag_fixed=frag_fixed,
            )
        rmsds = _product_rmsd(batch, out_batch["pos"], fragment_id=1)
        return float(np.nanmean(rmsds)), float(np.nanmedian(rmsds))

    def training_step(self, batch, batch_idx):
        """Lightning training step for one batch."""
        nll, info = self.compute_loss(batch)
        loss = nll.mean(0)
        self.log("train-totloss", loss, rank_zero_only=True)
        for k, v in info.items():
            self.log(f"train-{k}", v, rank_zero_only=True)

        if (self.current_epoch + 1) % self.eval_epochs == 0 and batch_idx == 0:
            rmsd_mean, rmsd_median = self.eval_inplaint_batch(batch, frag_fixed=[0])
            info["rmsd"], info["rmsd-median"] = rmsd_mean, rmsd_median
        else:
            info["rmsd"], info["rmsd-median"] = np.nan, np.nan
        info["loss"] = loss
        self._train_epoch_outputs.append(info)
        return info

    def _shared_eval(self, batch, batch_idx, prefix, *args):
        """Shared validation/test step."""
        del args
        nll, info = self.compute_loss(batch)
        loss = nll.mean(0)
        info["totloss"] = loss.item()
        if (self.current_epoch + 1) % self.eval_epochs == 0 and batch_idx == 0:
            info["rmsd"], info["rmsd-median"] = self.eval_inplaint_batch(batch, frag_fixed=[0])
        else:
            info["rmsd"], info["rmsd-median"] = np.nan, np.nan

        info_prefix = {}
        for k, v in info.items():
            info_prefix[f"{prefix}-{k}"] = v
        if prefix == "val":
            self._val_epoch_outputs.append(info_prefix)
        return info_prefix

    def validation_step(self, batch, batch_idx, *args):
        return self._shared_eval(batch, batch_idx, "val", *args)

    def test_step(self, batch, batch_idx, *args):
        return self._shared_eval(batch, batch_idx, "test", *args)

    def on_train_epoch_start(self) -> None:
        self._train_epoch_outputs = []

    def on_validation_epoch_start(self) -> None:
        self._val_epoch_outputs = []

    def on_validation_epoch_end(self) -> None:
        val_epoch_metrics = average_over_batch_metrics(self._val_epoch_outputs)
        if self.trainer.is_global_zero:
            pretty_print(self.current_epoch, val_epoch_metrics, prefix="val")
        val_epoch_metrics.update({"epoch": self.current_epoch})
        for k, v in val_epoch_metrics.items():
            self.log(k, v, sync_dist=True)

    def on_train_epoch_end(self) -> None:
        epoch_metrics = average_over_batch_metrics(
            self._train_epoch_outputs,
            allowed=["rmsd", "rmsd-median"],
        )
        self.log("train-rmsd", epoch_metrics.get("rmsd", np.nan), sync_dist=True)
        self.log("train-rmsd-median", epoch_metrics.get("rmsd-median", np.nan), sync_dist=True)

    def configure_gradient_clipping(
        self,
        optimizer,
        optimizer_idx=None,
        gradient_clip_val=None,
        gradient_clip_algorithm=None,
    ):
        """Adaptive gradient clipping based on recent gradient norms."""
        del optimizer_idx, gradient_clip_val, gradient_clip_algorithm
        if not self.clip_grad:
            return
        max_grad_norm = 1.5 * self.gradnorm_queue.mean() + 3 * self.gradnorm_queue.std()
        params = [p for g in optimizer.param_groups for p in g["params"]]
        grad_norm = get_grad_norm(params)
        self.clip_gradients(
            optimizer, gradient_clip_val=max_grad_norm, gradient_clip_algorithm="norm"
        )
        self.gradnorm_queue.add(float(min(grad_norm.item(), max_grad_norm)))


class ConfModule(LightningModule):
    """LightningModule for graph-level confidence prediction."""

    def __init__(
        self,
        model_config: Dict,
        optimizer_config: Dict,
        training_config: Dict,
        node_nfs,
        edge_nf: int = 0,
        condition_nf: int = 0,
        fragment_names=None,
        pos_dim: int = 3,
        edge_cutoff: Optional[float] = None,
        process_type: Optional[str] = None,
        model: nn.Module = None,
        enforce_same_encoding=None,
        source: Optional[Dict] = None,
        classification: bool = True,
        name_temp: str = "conf_hold",
        target_key: str = "target",
    ) -> None:
        super().__init__()
        _ = process_type, name_temp
        fragment_names = fragment_names or ["IS", "FS"]
        dtype = resolve_float_dtype(training_config.get("dtype", torch.float64))
        self.confidence = ConfidencePredictor(
            model_config=model_config,
            node_nfs=node_nfs,
            edge_nf=edge_nf,
            condition_nf=condition_nf,
            fragment_names=fragment_names,
            pos_dim=pos_dim,
            edge_cutoff=edge_cutoff,
            model=model,
            enforce_same_encoding=enforce_same_encoding,
            source=source,
            dtype=dtype,
        )
        self.optimizer_config = optimizer_config
        self.training_config = training_config
        # LightningModule already owns a ``dtype`` property through nn.Module.
        # Keep the dataset/model dtype under a project-specific name.
        self.data_dtype = dtype
        self.condition_nf = condition_nf
        self.classification = classification
        self.target_key = target_key
        self.clip_grad = training_config.get("clip_grad", False)
        if self.clip_grad:
            self.gradnorm_queue = Queue()
            self.gradnorm_queue.add(3000)
        self._val_epoch_outputs = []
        self.save_hyperparameters(ignore=["model"])

        if classification:
            self.AccEval = BinaryAccuracy(threshold=0.5)
            self.AUCEval = BinaryAUROC()
            self.PrecisionEval = BinaryPrecision(threshold=0.5)
            self.F1Eval = BinaryF1Score(threshold=0.5)
            self.loss_fn = nn.BCELoss()
        else:
            self.loss_fn = nn.MSELoss()
            self.MAEEval = MeanAbsoluteError()
            self.PearsonEval = PearsonCorrCoef()
            self.SpearmanEval = SpearmanCorrCoef()

    def configure_optimizers(self):
        """Create optimizer and optional LR scheduler for confidence model."""
        optimizer = torch.optim.AdamW(self.confidence.parameters(), **self.optimizer_config)
        schedule_type = self.training_config.get("lr_schedule_type")
        if schedule_type is not None:
            scheduler_func = LR_SCHEDULER[schedule_type]
            scheduler = scheduler_func(
                optimizer=optimizer, **self.training_config["lr_schedule_config"]
            )
            return [optimizer], [scheduler]
        return optimizer

    def setup(self, stage: Optional[str] = None):
        """Create split datasets for confidence training/evaluation."""
        if stage in ("fit", None):
            self.train_dataset = _build_dataset(self.training_config, "train")
            self.val_dataset = _build_dataset(self.training_config, "val")
        if stage in ("test", None):
            self.test_dataset = _build_dataset(self.training_config, "test")
        elif stage not in ("fit", "test", None):
            raise NotImplementedError

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            self.training_config["bz"],
            shuffle=True,
            num_workers=self.training_config.get("num_workers", 0),
            collate_fn=self.train_dataset.collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            self.training_config["bz"],
            shuffle=False,
            num_workers=self.training_config.get("num_workers", 0),
            collate_fn=self.val_dataset.collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            self.training_config["bz"],
            shuffle=False,
            num_workers=self.training_config.get("num_workers", 0),
            collate_fn=self.test_dataset.collate_fn,
        )

    def compute_loss(self, batch):
        """Compute confidence prediction loss and metrics.

        The default ``ReactionPairDataset`` does not include labels, so a real confidence
        task must provide ``batch[target_key]`` through a labeled dataset wrapper.
        """
        if self.target_key not in batch:
            raise KeyError(
                f"Confidence training expects `{self.target_key}` in the batch. "
                "Provide a dataset that appends labels to the joint graph batch."
            )
        conditions = _default_conditions(batch, self.condition_nf)
        targets = batch[self.target_key]
        preds = self.confidence(batch, conditions).to(targets.device)

        if self.classification:
            preds = torch.sigmoid(preds)
            info = {
                "acc": self.AccEval(preds, targets).item(),
                "AUC": self.AUCEval(preds, targets).item(),
                "mean": torch.mean(preds.round()).item(),
                "precision": self.PrecisionEval(preds, targets).item(),
                "F1": self.F1Eval(preds, targets).item(),
            }
        else:
            info = {
                "MAE": self.MAEEval(preds, targets).item(),
                "Pearson": self.PearsonEval(preds, targets).item(),
                "mean": torch.mean(preds).item(),
                "Spearman": self.SpearmanEval(preds, targets).item(),
            }
        loss = self.loss_fn(preds, targets.to(device=preds.device, dtype=preds.dtype))
        return loss, info

    def training_step(self, batch, batch_idx):
        del batch_idx
        loss, info = self.compute_loss(batch)
        self.log("train-totloss", loss, rank_zero_only=True)
        for k, v in info.items():
            self.log(f"train-{k}", v, rank_zero_only=True)
        return loss

    def _shared_eval(self, batch, batch_idx, prefix, *args):
        del batch_idx, args
        loss, info = self.compute_loss(batch)
        info["totloss"] = loss.item()
        info_prefix = {f"{prefix}-{k}": v for k, v in info.items()}
        if prefix == "val":
            self._val_epoch_outputs.append(info_prefix)
        return info_prefix

    def validation_step(self, batch, batch_idx, *args):
        return self._shared_eval(batch, batch_idx, "val", *args)

    def test_step(self, batch, batch_idx, *args):
        return self._shared_eval(batch, batch_idx, "test", *args)

    def on_validation_epoch_start(self) -> None:
        self._val_epoch_outputs = []

    def on_validation_epoch_end(self) -> None:
        val_epoch_metrics = average_over_batch_metrics(self._val_epoch_outputs)
        if self.trainer.is_global_zero:
            pretty_print(self.current_epoch, val_epoch_metrics, prefix="val")
        val_epoch_metrics.update({"epoch": self.current_epoch})
        for k, v in val_epoch_metrics.items():
            self.log(k, v, sync_dist=True)


RPDiffModule = DiffModule
RPConfModule = ConfModule
DDPMModule = DiffModule
