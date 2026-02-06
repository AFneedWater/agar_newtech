#!/bin/bash
#SBATCH --job-name=agar_train         # 作业名称
#SBATCH --partition=dzagnormal        # 分区名称 (A800节点)
#SBATCH --nodes=1                     # 申请1个节点
#SBATCH --ntasks-per-node=1           # 运行1个主任务 (由torchrun接管分发)
#SBATCH --cpus-per-task=16            # 申请16核CPU (保证数据加载不卡顿)
#SBATCH --gres=gpu:2                  # 申请2张A800显卡
#SBATCH --output=logs/%x_%j.out       # 标准输出日志 (自动保存在logs文件夹)
#SBATCH --error=logs/%x_%j.err        # 错误日志
#SBATCH --time=24:00:00               # 限制运行时间 (24小时)

# 1. 环境清理与加载
module purge
module load nvidia/cuda/12.2
module load compiler/gcc/9.3.0  # 既然之前需要高版本GCC，加载上更稳

# 2. 激活 Conda 环境
source /work/home/zqy_885225/miniforge3/bin/activate agar_a800

# 3. 定位到项目目录
cd /work/home/zqy_885225/gh/agar

# 4. 确保 Python 能找到源码 (非常重要！)
export PYTHONPATH=$PYTHONPATH:$(pwd)/src

# 5. 创建日志目录
mkdir -p logs

# 6. 打印节点信息 (调试用)
echo "Running on node: $(hostname)"
echo "GPUs available: $CUDA_VISIBLE_DEVICES"
nvidia-smi

# 7. 启动训练
# 注意：
# - 既然代码里修复了缩放，这里 batch_size=16 应该很稳
# - 保持 pin_memory=false 以防万一
# - 使用 srun 启动 torchrun 可以让 Slurm 更好地监控进程
srun torchrun --nproc_per_node=2 -m agar.train \
    experiment=exp_coco2026_agar_countable_4k \
    train.mlflow=true \
    train.mlflow_tracking_uri="sqlite:////work/home/zqy_885225/gh/agar/mlflow.db" \
    train.devices=2 \
    data.batch_size=16 \
    data.num_workers=8 \
    data.pin_memory=false