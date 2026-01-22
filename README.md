# agar

Minimal COCO-format detection training project (Hydra + Lightning Fabric).

## Install

```bash
pip install -e . -U
```

## Quick checks

```bash
python -m compileall -q src tests
```

## Train from standard COCO

```bash
python -m agar.train experiment=exp_agar_lower_from_coco_cache
```

Data config now uses standard COCO-style inputs:

- `data.train_ann` / `data.val_ann`: COCO annotation JSON(s)
- `data.images_root`: image root that `file_name` is relative to
- optional `data.coco_root`: base directory to resolve relative `*_ann` paths

Example for your provided dataset:

```bash
python -m agar.train experiment=exp_agar7dataset_standard_coco
python -m agar.train experiment=exp_coco2026_agar_countable_4k
```

## 2-GPU smoke (COCO2026 4k)

Prefer `agar.run` for multi-GPU: it auto-launches `torchrun` when `train.devices>1` (including when set inside the experiment yaml).

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m agar.run smoke experiment=exp_ddp_smoke_coco2026_agar_countable_4k
```

## Quick baseline (1 epoch)

```bash
python -m agar.train experiment=exp_quick_baseline
```

To try a larger batch temporarily, override on the CLI:

```bash
python -m agar.train experiment=exp_quick_baseline data.batch_size=12
```

## Tiny steps (fast sanity run)

```bash
python -m agar.train experiment=exp_tiny_steps
```

## MLflow

Default tracking uses SQLite to avoid the deprecated filesystem backend:

```bash
python -m agar.train experiment=exp_quick_baseline train.mlflow=true
```

Override the tracking URI if you want a different backend:

```bash
python -m agar.train experiment=exp_quick_baseline train.mlflow=true train.mlflow_tracking_uri=sqlite:////home/gh/mlflow.db
```

## Running

Unified entry (recommended):

```bash
CUDA_VISIBLE_DEVICES=0 python -m agar.run train experiment=exp_quick_baseline train.max_steps=5 train.eval_every=0
```

COCO2026 4k (2 GPUs):

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m agar.run train experiment=exp_coco2026_agar_countable_4k train.devices=2 train.max_steps=5 train.eval_every=0
```

Note: `train.mlflow=true` (not `ture`).

Multi-GPU (auto torchrun via `scripts/run_ddp.sh`):

```bash
CUDA_VISIBLE_DEVICES=0,1 DDP_PIN_CPU=1 python -m agar.run train experiment=exp_ddp_smoke train.devices=2 train.max_steps=5 train.eval_every=0
```

When `train.devices>1`, `agar.run` automatically launches torchrun and reuses `scripts/run_ddp.sh` for NCCL/NIC/CPU pinning.
If you still use manual `torchrun`, `agar.train`/`agar.eval`/`agar.smoke`/`agar.launch_check` will align `train.devices`/`train.strategy` in torchrun env.

Single GPU (explicit):

```bash
python -m agar.train experiment=exp_quick_baseline train.devices=1
```

If your machine has only 1 GPU, do not set `train.devices=2` (may segfault).

## DDP (torchrun)

Single GPU:

```bash
CUDA_VISIBLE_DEVICES=0 python -m agar.train experiment=exp_quick_baseline
```

Two GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 -m agar.train experiment=exp_ddp_smoke
```

Recommended: use `scripts/run_ddp.sh` to pin IFNAME + NCCL settings:

```bash
./scripts/run_ddp.sh torchrun --nproc_per_node=2 -m agar.train experiment=exp_ddp_smoke
```

Force a specific NIC if needed:

```bash
IFNAME=enp193s0f0 bash ./scripts/run_ddp.sh torchrun --nproc_per_node=2 -m agar.train experiment=exp_ddp_smoke
```

Enable CPU pinning for cross-NUMA GPUs (uses `nvidia-smi topo -m` CPU affinity):

```bash
DDP_PIN_CPU=1 bash ./scripts/run_ddp.sh torchrun --nproc_per_node=2 -m agar.train experiment=exp_ddp_smoke train.max_steps=10 train.eval_every=0
```

DDP triage logs (default enabled):

- Writes `logs/ddp_run_YYYYmmdd_HHMMSS/` on every `scripts/run_ddp.sh` run
- Disable: `DDP_LOG=0`
- Custom dir: `DDP_LOG_DIR=/abs/path`
- Save torchrun output: `DDP_LOG_STDOUT=1` (writes `torchrun.log`)

Full DDP training:

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 -m agar.train experiment=exp_agar_lower_from_coco_cache data.batch_size=2 data.num_workers=4
```

Verification (run inside conda env `agar_cv311`):

```bash
conda run -n agar_cv311 bash ./scripts/run_ddp.sh torchrun --nproc_per_node=2 -m agar.tools.ddp_sanity backend=nccl
conda run -n agar_cv311 DDP_PIN_CPU=1 bash ./scripts/run_ddp.sh torchrun --nproc_per_node=2 -m agar.tools.ddp_sanity backend=nccl
```

## DDP Debug

If your GPUs are cross-NUMA (topology shows `SYS`), CPU pinning can reduce cross-NUMA memory traffic and improve stability.
If you do not see `destroy_process_group: done` in `ddp_sanity` output, the process likely crashed or was killed (SIGKILL/segfault).

1) Basic debug:

```bash
NCCL_DEBUG=INFO TORCH_DISTRIBUTED_DEBUG=DETAIL CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 -m agar.tools.ddp_sanity backend=nccl
```

2) Disable IB:

```bash
NCCL_IB_DISABLE=1 NCCL_DEBUG=INFO TORCH_DISTRIBUTED_DEBUG=DETAIL CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 -m agar.tools.ddp_sanity backend=nccl
```

3) Disable SHM:

```bash
NCCL_SHM_DISABLE=1 NCCL_IB_DISABLE=1 NCCL_DEBUG=INFO TORCH_DISTRIBUTED_DEBUG=DETAIL CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 -m agar.tools.ddp_sanity backend=nccl
```

4) Disable P2P:

```bash
NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1 NCCL_IB_DISABLE=1 NCCL_DEBUG=INFO TORCH_DISTRIBUTED_DEBUG=DETAIL CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 -m agar.tools.ddp_sanity backend=nccl
```

## Train

Edit `conf/experiment/exp_agarv2_resnet50.yaml` to point to real data paths, then:

```bash
python -m agar.train experiment=exp_agarv2_resnet50
```

## Eval

```bash
python -m agar.eval ckpt=/path/to/checkpoints/checkpoint_best.pt
```

## COCO eval dependencies

`torchmetrics` mAP requires a COCO backend. Install one of:

```bash
python -m pip install -U pycocotools
```

Optional (faster evaluation):

```bash
python -m pip install -U faster-coco-eval
```

Verify:

```bash
python -c "from torchmetrics.detection.mean_ap import MeanAveragePrecision; MeanAveragePrecision(); print('OK')"
```
