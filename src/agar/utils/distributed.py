from __future__ import annotations

import os
from typing import Any, Dict


def in_torchrun_env() -> bool:
    return ("LOCAL_RANK" in os.environ) and ("WORLD_SIZE" in os.environ)


def torchrun_env() -> Dict[str, Any]:
    if not in_torchrun_env():
        return {"in_torchrun": False}

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", str(local_rank)))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", str(world_size)))
    return {
        "in_torchrun": True,
        "local_rank": local_rank,
        "rank": rank,
        "world_size": world_size,
        "local_world_size": local_world_size,
    }


def align_cfg_for_torchrun(cfg):
    env = torchrun_env()
    if not env.get("in_torchrun", False):
        return cfg

    from .rich_log import log

    desired_devices = int(env["local_world_size"])
    cfg_devices = int(getattr(cfg.train, "devices", desired_devices))
    if cfg_devices != desired_devices:
        log(
            f"align train.devices: cfg={cfg_devices} -> env LOCAL_WORLD_SIZE={desired_devices} (torchrun)"
        )
        cfg.train.devices = desired_devices

    cfg_strategy = str(getattr(cfg.train, "strategy", "auto"))
    if cfg_strategy in ("auto", "ddp") and cfg_strategy != "ddp":
        log(f"align train.strategy: cfg={cfg_strategy} -> ddp (torchrun)")
        cfg.train.strategy = "ddp"

    cfg_acc = str(getattr(cfg.train, "accelerator", "cuda"))
    if cfg_acc in ("auto", "cuda") and cfg_acc != "cuda":
        log(f"align train.accelerator: cfg={cfg_acc} -> cuda (torchrun)")
        cfg.train.accelerator = "cuda"

    log(
        "torchrun env: rank={rank} local_rank={local_rank} world_size={world_size} local_world_size={local_world_size}".format(
            **env
        )
    )
    return cfg

