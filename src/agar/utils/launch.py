from __future__ import annotations

import os
import torch

from .distributed import in_torchrun_env
from .rich_log import log


def is_torchrun_env() -> bool:
    return in_torchrun_env()


def resolve_accelerator(accelerator: str) -> str:
    if accelerator == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return accelerator


def log_prelaunch_info(local_rank_to_bind: int | None) -> None:
    rank = os.environ.get("RANK", "n/a")
    local_rank = os.environ.get("LOCAL_RANK", "n/a")
    world_size = os.environ.get("WORLD_SIZE", "n/a")
    local_world_size = os.environ.get("LOCAL_WORLD_SIZE", "n/a")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    device_count = torch.cuda.device_count()
    device_names = []
    for i in range(device_count):
        try:
            name = torch.cuda.get_device_name(i)
        except Exception:
            name = "n/a"
        device_names.append(f"{i}:{name}")

    print(
        "prelaunch rank={} local_rank={} world_size={} local_world_size={}".format(
            rank, local_rank, world_size, local_world_size
        ),
        flush=True,
    )
    print(f"prelaunch CUDA_VISIBLE_DEVICES={visible}", flush=True)
    print(f"prelaunch cuda.device_count={device_count}", flush=True)
    if device_names:
        print(f"prelaunch cuda.devices={', '.join(device_names)}", flush=True)
    print(f"prelaunch will_bind_local_rank={local_rank_to_bind}", flush=True)


def log_launch_info(fabric) -> None:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if torch.cuda.is_available() and fabric.device.type == "cuda":
        idx = torch.cuda.current_device()
        name = torch.cuda.get_device_name(idx)
    else:
        idx = -1
        name = "n/a"

    # Per-rank line
    print(
        "rank={}/{} local_rank={} cuda_device={} name={} CUDA_VISIBLE_DEVICES={}".format(
            fabric.global_rank,
            fabric.world_size,
            fabric.local_rank,
            idx,
            name,
            visible,
        ),
        flush=True,
    )

    if fabric.is_global_zero:
        log(f"torch.float32_matmul_precision={torch.get_float32_matmul_precision()}")
        log(f"cuda.device_count={torch.cuda.device_count()}")
