#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$ROOT/logs/ddp_diag_${TS}"
mkdir -p "$LOG_DIR"

SUMMARY="$LOG_DIR/summary.txt"
SYS="$LOG_DIR/sys.txt"
TOPO="$LOG_DIR/topo.txt"
GPU="$LOG_DIR/gpu_sanity.txt"
NCCL="$LOG_DIR/nccl_tests.txt"

run_cmd() {
  local outfile="$1"
  shift
  echo ">>> $*" | tee -a "$outfile"
  "$@" 2>&1 | tee -a "$outfile"
}

run_cmd_shell() {
  local outfile="$1"
  shift
  echo ">>> $*" | tee -a "$outfile"
  bash -lc "$*" 2>&1 | tee -a "$outfile"
}

IFNAME="$(ip route | awk '/default/ {print $5; exit}')"
if [[ -z "$IFNAME" ]]; then
  echo "WARN: could not determine default IFNAME" | tee -a "$SUMMARY"
fi

MASTER_PORT="${MASTER_PORT:-$((29500 + RANDOM % 1000))}"

{
  echo "log_dir=$LOG_DIR"
  echo "ifname=${IFNAME:-}"
  echo "master_port=$MASTER_PORT"
} | tee -a "$SUMMARY"

run_cmd "$SYS" which python
run_cmd "$SYS" python -V
run_cmd_shell "$SYS" "python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.backends.cudnn.version())'"
run_cmd "$SYS" nvidia-smi
run_cmd "$TOPO" nvidia-smi -L
run_cmd "$TOPO" nvidia-smi topo -m
run_cmd_shell "$TOPO" "nvidia-smi -q -d P2P || true"
run_cmd_shell "$TOPO" "nvidia-smi -q -d PCI || true"
run_cmd_shell "$SYS" "lsmod | egrep 'nvidia|nv_peer_mem|gdrdrv' || true"
run_cmd_shell "$SYS" "dmesg -T | egrep -i 'nvrm|pcie|xid|gpu|nvlink' | tail -n 200 || true"

run_cmd_shell "$GPU" "CUDA_VISIBLE_DEVICES=0 python - <<'PY'
import torch
torch.cuda.init()
x=torch.randn(8192,8192,device='cuda',dtype=torch.float16)
y=x@x
torch.cuda.synchronize()
print('gpu0 ok', y.mean().item())
PY"

run_cmd_shell "$GPU" "CUDA_VISIBLE_DEVICES=1 python - <<'PY'
import torch
torch.cuda.init()
x=torch.randn(8192,8192,device='cuda',dtype=torch.float16)
y=x@x
torch.cuda.synchronize()
print('gpu1 ok', y.mean().item())
PY"

ENV_IF="MASTER_PORT=$MASTER_PORT GLOO_SOCKET_IFNAME=$IFNAME NCCL_SOCKET_IFNAME=$IFNAME NCCL_SOCKET_FAMILY=AF_INET"

run_cmd_shell "$NCCL" "$ENV_IF CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 -m agar.tools.ddp_sanity backend=gloo"

run_cmd_shell "$NCCL" "$ENV_IF NCCL_DEBUG=INFO TORCH_DISTRIBUTED_DEBUG=DETAIL CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 -m agar.tools.ddp_sanity backend=nccl || true"

run_cmd_shell "$NCCL" "$ENV_IF NCCL_CUMEM_HOST_ENABLE=0 NCCL_DEBUG=INFO TORCH_DISTRIBUTED_DEBUG=DETAIL CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 -m agar.tools.ddp_sanity backend=nccl || true"

run_cmd_shell "$NCCL" "$ENV_IF NCCL_NET=Socket NCCL_IB_DISABLE=1 NCCL_CUMEM_HOST_ENABLE=0 NCCL_DEBUG=INFO TORCH_DISTRIBUTED_DEBUG=DETAIL CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 -m agar.tools.ddp_sanity backend=nccl || true"

run_cmd_shell "$NCCL" "$ENV_IF NCCL_NET=Socket NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1 NCCL_IB_DISABLE=1 NCCL_CUMEM_HOST_ENABLE=0 NCCL_DEBUG=INFO TORCH_DISTRIBUTED_DEBUG=DETAIL CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 -m agar.tools.ddp_sanity backend=nccl || true"

{
  echo "Conclusion hints:"
  echo "- If GPU1 single-card sanity fails: suspect GPU1/driver/hardware"
  echo "- If GPU1 sanity succeeds but NCCL SIGSEGV persists: suspect NCCL/driver/PCIe/topology; upgrade NVIDIA driver and retest"
} | tee -a "$SUMMARY"
