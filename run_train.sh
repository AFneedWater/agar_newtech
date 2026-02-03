#!/bin/bash
#SBATCH -J agar_train            # 任务名
#SBATCH -p dzagnormal            # 分区
#SBATCH -N 1                     # 1个节点
#SBATCH -n 1                     # 1个任务
#SBATCH -c 8                     # 8个CPU核心 (性价比最高，容易排队)
#SBATCH --gres=gpu:1             # 1张显卡
#SBATCH --output=train_%j.log    # 你的日志会保存在这
#SBATCH --error=train_%j.err     # 报错信息保存在这

# 1. 加载环境
module load nvidia/cuda/12.2
source /work/home/zqy_885225/miniforge3/bin/activate agar_a800
cd /work/home/zqy_885225/gh/agar

# 2. 运行训练 (后台自动跑)
echo "Start time: $(date)"

python -m agar.train \
    experiment=exp_coco2026_agar_countable_4k \
    train.mlflow=true \
    train.mlflow_tracking_uri="sqlite:////work/home/zqy_885225/gh/agar/mlflow.db" \
    data.batch_size=32 \
    data.num_workers=8

echo "End time: $(date)"