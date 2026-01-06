# AGAR 项目说明（详细版）

本项目是一个最小化的目标检测训练工程：以 **COCO 格式数据**为输入，使用 **Hydra** 管理配置，使用 **Lightning Fabric** 做训练/评估/分布式启动约束，默认模型为 **torchvision Faster R-CNN**。项目同时包含一套偏工程化的 DDP 启动与排障脚本（`scripts/run_ddp.sh` 等）。

> 英文快速使用说明见 `README.md`；本文更偏“项目结构/配置/工作流/排障”的完整说明。

---

## 1. 目录结构与关键文件

- `src/agar/`
  - `train.py`：训练入口（Hydra 配置 `conf/`），负责：
    - 构建 `Fabric`、`DataLoader`、模型与优化器
    - 训练循环（`src/agar/engine/fabric_loop.py`）
    - 记录 `TensorBoard/MLflow`（可选）
    - 保存 checkpoint（`checkpoints/`）
  - `eval.py`：评估入口，支持 `ckpt=/path/to/*.pt` 加载权重并输出 mAP 等指标（依赖 COCO backend）。
  - `smoke.py`：冒烟测试入口；若 COCO 路径不存在会自动用 dummy 数据跑 1 次 step（用于快速验证“模型+优化器+Fabric 启动”）。
  - `launch_check.py`：仅检查/打印 launch 信息（rank、CUDA 可见卡数、world_size 等），用于排查“本应单卡但被误启动为 DDP”之类问题。
  - `run.py`：统一入口（推荐）。当 `train.devices>1` 时自动通过 `scripts/run_ddp.sh` 启动 `torchrun`，避免 Fabric 隐式 spawn。
  - `data/`
    - `datamodule.py`：根据 `cfg.data.source` 构建 dataloaders（当前实现为 `coco`）。
    - `agar_dataset.py`：COCO detection dataset（返回 torchvision detection contract）。
    - `fo_to_coco.py`：从 FiftyOne 数据集导出 COCO cache（可随机拆分或按 view/tag 拆分）。
    - `inspect_coco.py`：检查 COCO JSON 质量（数量、抽样 bbox、越界比例、分布等）。
  - `tools/`
    - `ddp_sanity.py`：最小 DDP/collective 检查（gloo/nccl）。
    - `parse_topo_affinity.py`、`pin_and_exec.py`：配合 `scripts/run_ddp.sh` 做 CPU affinity pinning。
- `conf/`（Hydra 配置）
  - `config.yaml`：全局 defaults（data/model/train/eval/experiment）。
  - `data/agar.yaml`：数据默认配置（包含 COCO 路径占位符）。
  - `model/fasterrcnn.yaml`：模型配置（`num_classes` 为前景类别数，不含背景）。
  - `train/default.yaml`：训练超参/设备/日志开关等。
  - `experiment/*.yaml`：实验配置（通常填真实数据路径、batch、devices 等）。
- `scripts/`
  - `run_ddp.sh`：DDP 启动包装（自动选择网卡 IFNAME、设置 NCCL/GLOO 环境、可选 CPU pinning、写 triage logs）。
  - `ddp_triage.sh`、`ddp_diag.sh`：更重的环境/拓扑/单卡 sanity + NCCL sanity 收集脚本。
- `outputs/`：Hydra 默认输出目录（按日期归档）。
- `logs/`：DDP triage/diag 以及 `run_ddp.sh` 写入的日志目录。
- `checkpoints/`、`tb/`：训练时写入（也会出现在某次 Hydra `outputs/.../` 里，取决于工作目录）。
- `tests/`：pytest 覆盖了若干关键行为（dataset contract、torchrun config 对齐、`agar.run` devices 解析等）。

---

## 2. 环境与安装

### 2.1 Python 与依赖

- Python：`>=3.11`（见 `pyproject.toml`）。
- 安装（开发模式）：

```bash
pip install -e . -U
```

也可以用 `requirements.txt` 安装一组固定版本（包含 `torch/torchvision/torchaudio` 与 `fiftyone`）。

### 2.2 COCO mAP 评估依赖（强烈建议）

`torchmetrics.detection.MeanAveragePrecision` 需要 COCO backend。二选一安装：

```bash
python -m pip install -U pycocotools
# 或（更快）
python -m pip install -U faster-coco-eval
```

验证：

```bash
python -c "from torchmetrics.detection.mean_ap import MeanAveragePrecision; MeanAveragePrecision(); print('OK')"
```

如果缺依赖：
- 训练会打印提示并跳过 eval（`train.eval_every=0` 也可显式关闭）。
- `python -m agar.eval ...` 会得到空指标或无法计算。

---

## 3. 数据工作流（COCO cache 为中心）

本项目训练当前以 `data.source: coco` 为主；FiftyOne 仅作为“离线导出 COCO cache”的工具链存在。

### 3.1 COCO cache 的最小要求

你需要提供：
- `data.images_dir`：图片根目录（训练时会用 `file_name` 拼接到该目录下）。
- `data.train_json`：COCO train 标注 JSON。
- `data.val_json`：COCO val 标注 JSON。

并确保标注满足 torchvision detection 任务需求（dataset 会输出）：
- `target["boxes"]`: `FloatTensor[N,4]`，xyxy
- `target["labels"]`: `LongTensor[N]`
- 允许 `N=0`（无框图像）

### 3.2 从 FiftyOne 导出 COCO cache

示例（随机拆分，不写 tag）：

```bash
python -m agar.data.fo_to_coco \
  --fo-dataset AGARlower \
  --label-field detections \
  --train-view "" \
  --val-view "" \
  --out-dir /abs/path/to/coco_cache/agar_lower_coco \
  --copy-images false \
  --image-root /abs/path/to/images_root \
  --classes "E.coli,S.aureus,P.aeruginosa,C.albicans,B.subtilis,Contamination,Defect" \
  --split-method random \
  --train-ratio 0.8 \
  --seed 42
```

注意：
- `--copy-images false` 时必须提供 `--image-root`，导出的 `file_name` 会是相对 `image-root` 的路径。
- `--copy-images true` 会在 `out_dir/images/` 下 hardlink（失败则 copy）一份图片（便于可移植，但占空间）。

### 3.3 检查 COCO cache（强烈建议）

```bash
python -m agar.data.inspect_coco \
  --json /abs/path/to/coco/train.json \
  --images-dir /abs/path/to/images_root \
  --check-all
```

该工具会抽样 bbox 并检查越界比例；即使有越界样本，训练侧也会对 bbox 做裁剪（避免崩溃），但最好在数据侧修正。

---

## 4. Hydra 配置体系与实验组织方式

### 4.1 defaults 与实验（experiment）覆盖

`conf/config.yaml` 默认会加载：
- `data: agar`
- `model: fasterrcnn`
- `train: default`
- `eval: default`
- `experiment: exp_agar_lower_from_coco_cache`

一般你会：
1) 把真实数据路径写到某个 `conf/experiment/*.yaml` 中（推荐新建一个自己的 experiment 文件）。
2) 运行时用 `experiment=xxx` 选择实验配置。

### 4.2 实验配置的建议写法

已有实验文件大多使用：

- `# @package _global_`：让配置覆盖发生在 root（而不是挂在 `experiment.*` 下面）。
- `defaults: - exp_agar_lower_from_coco_cache`：复用一个“包含真实路径/类别数”的基底实验，再局部覆盖 batch、steps 等。

例如：
- `conf/experiment/exp_agar_lower_from_coco_cache.yaml`：给出一组具体 COCO 路径与 `model.num_classes`。
- `conf/experiment/exp_quick_baseline.yaml`：在上面基础上改 batch/num_workers 并只训 1 epoch。
- `conf/experiment/exp_tiny_steps.yaml`：超快 sanity（`max_steps=5`、`eval_every=0`）。
- `conf/experiment/exp_ddp_smoke.yaml`：2 卡 DDP 冒烟（`train.devices=2`）。

### 4.3 命令行覆盖（overrides）示例

```bash
python -m agar.train experiment=exp_quick_baseline data.batch_size=12 train.max_steps=50 train.eval_every=0
```

查看合并后的配置：

```bash
python -m agar.train experiment=exp_quick_baseline --cfg job
```

---

## 5. 训练（单卡 / DDP）

### 5.1 单卡训练（推荐从这里开始）

```bash
python -m agar.train experiment=exp_quick_baseline train.devices=1
```

如果机器上有多张卡，但你只想单卡训练，务必显式设置 `train.devices=1`（本项目也会在非 torchrun 环境下拒绝 `train.devices>1`，避免 Fabric 隐式 spawn 引发不稳定）。

### 5.2 使用统一入口 `agar.run`（推荐）

单卡：

```bash
CUDA_VISIBLE_DEVICES=0 python -m agar.run train experiment=exp_quick_baseline train.max_steps=5 train.eval_every=0
```

多卡（`train.devices>1` 会自动 torchrun，并复用 `scripts/run_ddp.sh`）：

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m agar.run train experiment=exp_ddp_smoke train.devices=2 train.max_steps=5 train.eval_every=0
```

### 5.3 直接 torchrun（可选）

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 -m agar.train experiment=exp_ddp_smoke
```

更推荐（带 NCCL/NIC/日志能力）：

```bash
./scripts/run_ddp.sh torchrun --nproc_per_node=2 -m agar.train experiment=exp_ddp_smoke
```

CPU pinning（跨 NUMA/跨 PCIe 拓扑时可能更稳定）：

```bash
DDP_PIN_CPU=1 ./scripts/run_ddp.sh torchrun --nproc_per_node=2 -m agar.train experiment=exp_ddp_smoke
```

### 5.4 训练输出与日志

训练在 Hydra 的运行目录（默认 `outputs/YYYY-MM-DD/HH-MM-SS/`）里会写入：
- `checkpoints/`：`checkpoint_last.pt`、`checkpoint_best.pt`（取决于 `eval.metric_key` 与评估是否启用）
- `tb/`：TensorBoard event files（若 `train.tensorboard: true` 且 global rank 0）
- `data_stats.json`：训练/验证 COCO 的统计（images、annotations、类计数、bbox area 分位数等）

---

## 6. 评估

```bash
python -m agar.eval experiment=exp_agar_lower_from_coco_cache ckpt=/abs/path/to/checkpoint_best.pt
```

- 若 COCO JSON 中包含 `categories.name`，评估会打印 `category_id -> name` 映射。
- 指标基于 `torchmetrics` 的 COCO mAP 计算（需要安装 `pycocotools` 或 `faster-coco-eval`）。

---

## 7. 冒烟与启动检查工具

- 冒烟（优先验证“数据可读 + 训练链路可跑通”）：

```bash
python -m agar.smoke experiment=exp_tiny_steps
```

- 只检查 launch 信息（排查 devices/world_size/可见卡）：

```bash
python -m agar.launch_check experiment=exp_quick_baseline
```

- 最小 DDP sanity（排查 NCCL/GLOO 初始化与 all_reduce）：

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 -m agar.tools.ddp_sanity backend=nccl
```

---

## 8. DDP 常见问题与排障建议

### 8.1 现象：应为单卡却启动成多卡/或 rank1 崩溃

建议流程：
1) 显式设置 `train.devices=1`，并用 `python -m agar.launch_check ...` 确认 `world_size=1`。
2) 使用 `python -m agar.run ...` 启动多卡，确保由项目统一控制 torchrun。
3) 如果仍不稳定，优先跑 `ddp_sanity`，再跑 triage/diag 脚本收集信息。

### 8.2 一键收集（更重）

```bash
bash scripts/ddp_triage.sh
# 或
bash scripts/ddp_diag.sh
```

它们会收集：
- `nvidia-smi -L`、`nvidia-smi topo -m`、P2P/PCI 信息
- 单卡 matmul sanity（分别测 GPU0/GPU1）
- gloo/nccl 的 `ddp_sanity` 结果

---

## 9. 测试与开发

运行测试：

```bash
pytest -q
```

（可选）ruff：

```bash
ruff check .
```

---

## 10. 目前的空白/待实现项

- `src/agar/predict.py` 目前是占位：批量推理与可视化尚未实现。
- `data.source: fiftyone` 的在线 dataloader 目前未实现（当前推荐 FO -> COCO cache 的离线导出方式）。

