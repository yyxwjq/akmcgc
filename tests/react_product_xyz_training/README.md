# React/Product XYZ Training Test

这个目录专门用于用下面两个文件训练一个初末态 reaction diffusion 模型：

```text
../data/react.xyz
../data/product.xyz
```

## 文件说明

```text
train_react_product_xyz.ipynb
```

主训练 notebook。它会：

1. 按 frame index 把 react/product 同步切成 70% train、10% validation、20% test。
2. 构建 `ReactionPairDataset` 和 `DiffModule`。
3. 运行 `Trainer.fit(...)`。
4. 保存 checkpoint、loss history、Lightning CSV 日志和 run config。
5. 训练后评估 test denoising loss 和条件生成的 inpainting RMSD。

```text
inpaint_from_checkpoint.ipynb
```

训练完成后的专用推理 notebook。它只做一件事：

```text
读取 run_config.json + .ckpt
-> 对一个 test sample 做一次 inpaint(frag_fixed=[0])
-> 导出 extxyz 和可视化 png
```

默认会写出：

```text
conditioned_reactant.extxyz
predicted_product.extxyz
target_product.extxyz
inpaint_visualization.png
inpaint_summary.json
```

## 默认输出

notebook 默认把所有本次测试输出写到：

```text
tests/react_product_xyz_training/
```

主要输出包括：

```text
splits/seed_42/
  train_react.xyz
  train_product.xyz
  val_react.xyz
  val_product.xyz
  test_react.xyz
  test_product.xyz
  split_indices.json

runs/react_product_egnn_70_10_20/
  run_config.json
  loss_history.csv
  checkpoints/
  lightning_logs/
```

## 怎么实时看训练

运行 notebook 里的 `Train` cell 时，可以从三个地方看训练情况：

1. Lightning/TQDM 进度条：显示 epoch 和 batch 进度。
2. cell 输出中的实时 loss：

```text
[train] epoch=0 batch=10 loss=...
```

3. CSV 日志：

```text
runs/react_product_egnn_70_10_20/lightning_logs/csv/version_*/metrics.csv
```

notebook 训练结束后还会打印：

```text
first 5 train losses
last 5 train losses
mean first 5
mean last 5
```

## 训练目标到底是什么

这个 notebook 默认训练的主任务是：

```text
给定 reactant -> 条件生成 product
```

也就是 `.ckpt` 最直接的用途是 `inpaint(frag_fixed=[0])`。这和当前仓库的 joint-graph 设计一致。

当前 `tests/react_product_xyz_training/` 这套测试资产只保留了 inpaint 路径，不再提供单独的命令行脚本。

## 怎么使用训练出来的模型文件

训练后会得到 checkpoint，例如：

```text
runs/react_product_egnn_70_10_20/checkpoints/last.ckpt
```

### 用法：打开 `inpaint_from_checkpoint.ipynb`

训练完成后，直接打开：

```text
tests/react_product_xyz_training/inpaint_from_checkpoint.ipynb
```

它默认读取：

```text
tests/react_product_xyz_training/runs/react_product_egnn_70_10_20/run_config.json
tests/react_product_xyz_training/runs/react_product_egnn_70_10_20/checkpoints/last.ckpt
```

然后在：

```text
tests/react_product_xyz_training/runs/react_product_egnn_70_10_20/notebook_inpaint/
```

写出：

```text
conditioned_reactant.extxyz
predicted_product.extxyz
target_product.extxyz
inpaint_visualization.png
inpaint_summary.json
```

这里最值得对比的是：

- `conditioned_reactant.extxyz`：固定不动的输入 reactant，带 `cell / pbc`
- `predicted_product.extxyz`：模型生成的 product，带 `cell / pbc`
- `target_product.extxyz`：测试集真实 product，带 `cell / pbc`
- `inpaint_visualization.png`：reactant / 预测 product / 真实 product 的并排图
- `inpaint_summary.json`：本次推理使用的 checkpoint、timesteps 和 RMSD 记录

可以用 ASE、OVITO、VMD 或其他结构可视化工具打开。

## 正式训练建议

notebook 默认是偏保守的小训练配置。确认流程无误后，可以把配置改成：

```python
MAX_EPOCHS = 100
LIMIT_TRAIN_BATCHES = None
LIMIT_VAL_BATCHES = None
HIDDEN_NF = 128
N_LAYERS = 3
INPAINT_TIMESTEPS = 32
```

如果 GPU 可用：

```python
ACCELERATOR = "gpu"
```

先看 `final test denoising coord loss` 是否低于 `initial test denoising coord loss`，再看 inpainting RMSD。短训练时 inpainting RMSD 可能很大，这是正常的。
