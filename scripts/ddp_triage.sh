#!/usr/bin/env bash
set -euo pipefail

TS="$(date +%Y%m%d_%H%M%S)"
OUTDIR="logs/ddp_triage_${TS}"
mkdir -p "${OUTDIR}"

# tee helper
run() {
  local name="$1"; shift
  echo "==================================================" | tee -a "${OUTDIR}/${name}.log"
  echo "CMD: $*" | tee -a "${OUTDIR}/${name}.log"
  echo "TIME: $(date)" | tee -a "${OUTDIR}/${name}.log"
  echo "--------------------------------------------------" | tee -a "${OUTDIR}/${name}.log"
  ( "$@" 2>&1 | tee -a "${OUTDIR}/${name}.log" )
  echo "" | tee -a "${OUTDIR}/${name}.log"
}

echo "Writing logs to: ${OUTDIR}"

# 0) Basic env
run "env" bash -lc 'echo "HOST=$(hostname)"; echo "USER=$(whoami)"; echo "PWD=$(pwd)"; echo "PY=$(which python)"; python -V; echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES";'

# 1) nvidia-smi -L
run "nvidia_smi_L" bash -lc 'nvidia-smi -L'

# 2) nvidia-smi topo -m
run "nvidia_smi_topo" bash -lc 'nvidia-smi topo -m'

# 3) detect default NIC (physical route) and print
run "net_default" bash -lc '
set -e
IFNAME=$(ip route show default 2>/dev/null | awk '"'"'{print $5; exit}'"'"')
echo "default_ifname=${IFNAME}"
ip -o addr show dev "${IFNAME}" || true
'

# 4) Single-GPU sanity on GPU1: fp16 matmul
run "gpu1_matmul_sanity" bash -lc '
set -e
CUDA_VISIBLE_DEVICES=1 python - <<'"'"'PY'"'"'
import os, torch, time
print("CUDA_VISIBLE_DEVICES=", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("torch=", torch.__version__, "cuda=", torch.version.cuda, "cudnn=", torch.backends.cudnn.version())
assert torch.cuda.is_available()
print("device_count:", torch.cuda.device_count())
print("device0:", torch.cuda.get_device_name(0))
try:
    torch.backends.cuda.matmul.fp32_precision = "tf32"
    torch.backends.cudnn.conv.fp32_precision = "tf32"
except Exception:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
torch.cuda.init()
torch.cuda.synchronize()

# a slightly smaller size to reduce OOM risk; raise if you want
n = 6144
x = torch.randn((n, n), device="cuda", dtype=torch.float16)
torch.cuda.synchronize()
t0 = time.time()
y = x @ x
torch.cuda.synchronize()
t1 = time.time()
print("matmul_ok mean=", float(y.mean().item()), "elapsed_s=", round(t1-t0, 4))
PY
'

# 5) NCCL ddp_sanity with fixed NIC + CUMEM host off
run "nccl_sanity_fixed_if" bash -lc '
set -e
IFNAME=$(ip route show default 2>/dev/null | awk '"'"'{print $5; exit}'"'"')
echo "Using IFNAME=$IFNAME"
export CUDA_VISIBLE_DEVICES=0,1
export GLOO_SOCKET_IFNAME=$IFNAME
export NCCL_SOCKET_IFNAME=$IFNAME
export NCCL_SOCKET_FAMILY=AF_INET
export NCCL_CUMEM_HOST_ENABLE=0
export NCCL_NET=Socket
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=DETAIL

torchrun --nproc_per_node=2 -m agar.tools.ddp_sanity backend=nccl
'

echo "Done. Collected logs under: ${OUTDIR}"
