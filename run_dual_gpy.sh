#!/bin/bash
#SBATCH -J agar_2gpu             # 任务名
#SBATCH -p dzagnormal            # 分区
#SBATCH -N 1                     # 1个节点
#SBATCH -n 16                    # 申请 16 个核心 (配合2张卡)
#SBATCH --gres=gpu:2             # 申请 2 张 A800
#SBATCH --output=train_%j.log    # 日志
#SBATCH --error=train_%j.err     # 报错日志

echo "开始时间: $(date)"
echo "节点: $(hostname)"

# --- 环境准备 ---
module purge
module load nvidia/cuda/12.2
source /work/home/zqy_885225/miniforge3/bin/activate agar_a800
cd /work/home/zqy_885225/gh/agar

# --- 检查显卡 ---
nvidia-smi

# --- 开始训练 ---
# trainer.devices=2: 启用双卡并行 (DDP)
# data.num_workers=8: 充分利用 CPU
python -m agar.train \
    experiment=exp_coco2026_agar_countable_4k \
    train.mlflow=true \
    train.mlflow_tracking_uri="sqlite:////work/home/zqy_885225/gh/agar/mlflow.db" \
    trainer.devices=2 \
    data.batch_size=32 \
    data.num_workers=8

echo "结束时间: $(date)"