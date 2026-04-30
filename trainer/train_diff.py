from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks.progress import TQDMProgressBar

try:
    from .task import DiffModule
    from ..model import EGNN, LEFTNet
except ImportError:  # pragma: no cover
    from task import DiffModule
    from model import EGNN, LEFTNet


model_type = "leftnet"
version = "joint-periodic-v1"
project = "akmcgc"

egnn_config = dict(
    in_node_nf=8,
    in_edge_nf=0,
    hidden_nf=256,
    edge_hidden_nf=64,
    act_fn="swish",
    n_layers=9,
    attention=True,
    out_node_nf=None,
    tanh=True,
    coords_range=15.0,
    norm_constant=1.0,
    inv_sublayers=1,
    sin_embedding=True,
    normalization_factor=1.0,
    aggregation_method="mean",
)
leftnet_config = dict(
    pos_require_grad=False,
    cutoff=10.0,
    num_layers=6,
    hidden_channels=196,
    num_radial=96,
    in_hidden_channels=8,
    reflect_equiv=True,
    legacy=True,
    update=True,
    pos_grad=False,
    single_layer_output=True,
    object_aware=True,
)

if model_type == "leftnet":
    model_config = leftnet_config
    model = LEFTNet
elif model_type == "egnn":
    model_config = egnn_config
    model = EGNN
else:
    raise KeyError("model type not implemented.")

optimizer_config = dict(
    lr=2.5e-4,
    betas=[0.9, 0.999],
    weight_decay=0,
    amsgrad=True,
)

repo_root = Path(__file__).resolve().parents[1]
sample_react = repo_root / "tests" / "data" / "h2o_react.extxyz"
sample_product = repo_root / "tests" / "data" / "h2o_product.extxyz"

training_config = dict(
    train_react_file=str(sample_react),
    train_product_file=str(sample_product),
    val_react_file=str(sample_react),
    val_product_file=str(sample_product),
    test_react_file=str(sample_react),
    test_product_file=str(sample_product),
    cutoff=4.0,
    max_neigh=32,
    r_fixed=True,
    r_pbc=True,
    device="cpu",
    dtype="float64",
    num_elements=118,
    bz=2,
    num_workers=0,
    clip_grad=True,
    gradient_clip_val=None,
    ema=False,
    ema_decay=0.999,
    lr_schedule_type=None,
    lr_schedule_config=dict(gamma=0.8, step_size=100),
    sampling_timesteps=150,
)

node_nf = 3 + training_config["num_elements"] + 1
node_nfs = [node_nf]
edge_nf = 0
condition_nf = 0
fragment_names = ["IS", "FS"]
pos_dim = 3
update_pocket_coords = True
condition_time = True
edge_cutoff = None
loss_type = "l2"
pos_only = True
eval_epochs = 10
norm_values = (1.0, 1.0, 1.0)
norm_biases = (0.0, 0.0, 0.0)
noise_schedule = "cosine"
timesteps = 5000
precision = 1e-5

run_name = f"{model_type}-{version}-" + str(uuid4()).split("-")[-1]

seed_everything(42, workers=True)
diff_mod = DiffModule(
    model_config=model_config,
    optimizer_config=optimizer_config,
    training_config=training_config,
    node_nfs=node_nfs,
    edge_nf=edge_nf,
    condition_nf=condition_nf,
    fragment_names=fragment_names,
    pos_dim=pos_dim,
    update_pocket_coords=update_pocket_coords,
    condition_time=condition_time,
    edge_cutoff=edge_cutoff,
    norm_values=norm_values,
    norm_biases=norm_biases,
    noise_schedule=noise_schedule,
    timesteps=timesteps,
    precision=precision,
    loss_type=loss_type,
    pos_only=pos_only,
    model=model,
    scales=[1.0],
    source=None,
    fixed_idx=None,
    eval_epochs=eval_epochs,
)

ckpt_path = repo_root / "checkpoint" / project / run_name
ckpt_path.mkdir(parents=True, exist_ok=True)
callbacks = [
    EarlyStopping(monitor="val-totloss", patience=2000, verbose=True, log_rank_zero_only=True),
    ModelCheckpoint(
        monitor="val-totloss",
        dirpath=str(ckpt_path),
        filename="ddpm-{epoch:03d}-{val-totloss:.2f}",
        every_n_epochs=1,
        save_top_k=-1,
    ),
    TQDMProgressBar(),
    LearningRateMonitor(logging_interval="step"),
]
if training_config["ema"]:
    try:
        from .ema import EMACallback
    except ImportError:  # pragma: no cover
        from ema import EMACallback
    callbacks.append(EMACallback(decay=training_config["ema_decay"]))

trainer = Trainer(
    max_epochs=2000,
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    deterministic=False,
    devices=1,
    log_every_n_steps=1,
    callbacks=callbacks,
    profiler=None,
    logger=False,
    accumulate_grad_batches=1,
    gradient_clip_val=training_config["gradient_clip_val"],
    limit_train_batches=200,
    limit_val_batches=20,
)

trainer.fit(diff_mod)
