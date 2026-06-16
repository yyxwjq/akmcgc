# Understand Anything 使用指南

本指南面向当前项目 `akmcgc`，目标是帮助你用 `Understand Anything` 更快理解代码结构、训练流程和模块关系。

## 1. 它是什么

`Understand Anything` 不是普通的聊天命令集合，它的核心机制是：

1. 先分析代码库
2. 生成一个知识图文件
3. 再基于知识图做问答、解释、可视化和 onboarding

默认输出目录在项目根目录下：

```text
.understand-anything/
```

其中最重要的文件是：

```text
.understand-anything/knowledge-graph.json
```

如果这个文件还没生成，很多 `understand-*` 命令都不能正常工作。

## 2. 最常用工作流

对于一个普通代码仓库，建议按下面顺序使用：

### 第一步：进入项目目录

```bash
cd /Users/wx/Desktop/yyxwjq/akmcgc
```

### 第二步：先建图

建议第一次直接用中文输出：

```bash
understand --language zh
```

如果你想强制完整重建：

```bash
understand --full --language zh
```

执行后，工具会扫描代码并生成：

```text
/Users/wx/Desktop/yyxwjq/akmcgc/.understand-anything/knowledge-graph.json
```

### 第三步：开始提问或深挖

常用命令：

```bash
understand-chat training pipeline
understand-explain trainer/task.py
understand-explain diffusion/diff.py
understand-onboard
understand-dashboard
```

## 3. 各命令用途

### `understand`

作用：分析项目并生成知识图。

常用写法：

```bash
understand
understand --language zh
understand --full
understand --auto-update
understand /Users/wx/Desktop/yyxwjq/akmcgc
```

常用参数：

- `--full`
  强制全量重建，不使用已有图。
- `--language zh`
  让图中的摘要、标题、说明优先用中文生成。
- `--auto-update`
  打开自动更新配置。
- `--no-auto-update`
  关闭自动更新配置。
- `--review`
  跑一次图审查流程，适合你怀疑图质量不够时使用。

适合场景：

- 第一次分析一个仓库
- 仓库改动较大后重新生成结构图
- 希望后续问答都基于统一结构理解

### `understand-chat`

作用：基于知识图回答你关于代码库的问题。

常用写法：

```bash
understand-chat where is the training entrypoint
understand-chat how diffusion connects to denoiser
understand-chat periodic boundary condition
understand-chat what are the main modules
```

适合场景：

- 想快速定位入口文件
- 想知道某个功能分布在哪些模块
- 想追高层调用链，但不想自己一层层翻文件

建议提问方式：

- 好问题：`how DiffModule, Denoiser, and Diffusion connect`
- 好问题：`where dataset becomes graph tensors`
- 不够好的问题：`explain this repo`

尽量带上明确对象或主题词。

### `understand-explain`

作用：深入解释某个文件、函数或模块。

常用写法：

```bash
understand-explain trainer/task.py
understand-explain dataset/reaction_pair_dataset.py
understand-explain diffusion/diff.py
understand-explain trainer/task.py:DiffModule
```

它会：

- 先从知识图里找目标节点
- 再读取真实源码
- 结合上下游依赖解释它在项目中的角色

适合场景：

- 读某个核心文件时看不懂
- 想知道“这个文件为什么存在”
- 想结合上下游关系看单文件，不只看局部代码

### `understand-onboard`

作用：自动生成适合新成员阅读的项目上手说明。

常用写法：

```bash
understand-onboard
```

一般会组织出以下内容：

- 项目概览
- 架构层次
- 推荐阅读路径
- 关键文件说明
- 复杂度热点

适合场景：

- 你自己隔一段时间回来重新接项目
- 想给未来的自己或合作同学留一个高层入口

### `understand-dashboard`

作用：启动交互式 dashboard，看知识图的可视化结果。

常用写法：

```bash
understand-dashboard
understand-dashboard /Users/wx/Desktop/yyxwjq/akmcgc
```

适合场景：

- 想从图形界面看模块连接关系
- 想看 layer、tour、节点分组
- 想比纯文本更直观地浏览项目

### `understand-diff`

作用：分析代码改动的影响面。

适合场景：

- 改一个核心模块前，先看会影响哪些部分
- 做重构或 review 时看变更扩散范围
- 分析某次 diff 会波及哪些节点

说明：

这个命令更适合“你已经开始改代码”之后使用，不是第一阶段的主力工具。

### `understand-domain`

作用：提炼业务域、业务流程和过程图。

适合场景：

- Web 后端
- 业务系统
- API 服务
- 有明显业务流程的应用代码

对 `akmcgc` 这类科研/深度学习仓库，通常不是第一优先级。

### `understand-knowledge`

作用：分析 Karpathy 风格的 wiki/知识库，而不是普通源码仓库。

适合场景：

- 你维护了一个 markdown 知识库
- 想把 wiki 变成结构化知识图

对当前 `akmcgc` 不属于主用命令。

## 4. 针对 `akmcgc` 的推荐问法

下面这些问题最适合你当前项目：

### 先看整体结构

```bash
understand-chat what are the main modules
understand-chat what is the training pipeline
understand-chat where is the real training entrypoint
```

### 再看数据流

```bash
understand-chat how data flows from dataset to diffusion
understand-chat where extxyz becomes graph tensors
understand-chat how batch, fragment, and mask are used
```

### 再看核心模块

```bash
understand-explain trainer/task.py
understand-explain dataset/reaction_pair_dataset.py
understand-explain diffusion/diff.py
understand-explain denoising/denoiser.py
understand-explain model/leftnet.py
```

### 最后看 onboarding

```bash
understand-onboard
```

## 5. 推荐阅读顺序

对于 `akmcgc`，建议按这个顺序读：

1. `trainer/train_diff.py`
   看训练是怎么启动的
2. `trainer/task.py`
   看 LightningModule 是怎么把 dataset、denoiser、diffusion 串起来的
3. `dataset/reaction_pair_dataset.py`
   看原始结构如何变成 joint graph
4. `diffusion/diff.py`
   看扩散训练和采样逻辑
5. `denoising/denoiser.py`
   看网络预测接口
6. `model/`
   看具体 backbone，例如 `EGNN` 或 `LEFTNet`

对应的命令可以直接这样跑：

```bash
understand-explain trainer/train_diff.py
understand-explain trainer/task.py
understand-explain dataset/reaction_pair_dataset.py
understand-explain diffusion/diff.py
```

## 6. 什么时候需要重新运行 `understand`

建议在这些情况下重新跑：

- 你新增了较多文件
- 你重构了目录结构
- 你改了训练主链路
- 你发现问答内容和当前代码明显不一致

常用命令：

```bash
understand --full --language zh
```

如果只是小改动，普通：

```bash
understand --language zh
```

通常就够了。

## 7. 常见问题

### 问题 1：为什么 `understand-chat` 或 `understand-explain` 不能直接用？

因为它们依赖：

```text
.understand-anything/knowledge-graph.json
```

先运行：

```bash
understand --language zh
```

### 问题 2：为什么生成很慢？

因为第一步是在分析整个项目并构建结构化图谱。项目越大，首次分析越慢。后续通常会快一些。

### 问题 3：我应该一直用中文还是英文？

如果你主要是自己阅读和理解，建议：

```bash
understand --language zh
```

如果你后续要直接把生成内容给国际合作者看，或者你希望术语更贴近原始代码生态，可以用英文。

### 问题 4：它能完全代替自己读代码吗？

不能。

它最适合做：

- 入口定位
- 模块关系梳理
- 调用链和层次理解
- 为深读某个文件建立上下文

真正的数学实现、张量形状、细节 bug，最后还是要回到源码本身。

## 8. 最小实用方案

如果你不想记太多命令，实际只记下面 4 个就够了：

```bash
understand --language zh
understand-chat what is the training pipeline
understand-explain trainer/task.py
understand-onboard
```

这 4 个命令已经能覆盖：

- 建图
- 看整体
- 看核心文件
- 生成上手说明

## 9. 建议

对于你现在的目标，“理解并逐步标准化 `akmcgc`”，推荐使用策略是：

1. 先跑 `understand --language zh`
2. 再用 `understand-chat` 搞清楚训练主链路
3. 然后用 `understand-explain` 深读 `trainer/task.py`、`dataset/reaction_pair_dataset.py`、`diffusion/diff.py`
4. 最后用 `understand-onboard` 生成一份高层说明，作为你后续整理项目的基础

这比一开始就逐个文件硬啃，更有效率。
