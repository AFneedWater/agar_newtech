from __future__ import annotations

import os
import hydra
from lightning_fabric import Fabric
import torch

from .utils.tf32 import configure_tf32

configure_tf32(enable=True)

from .utils.distributed import align_cfg_for_torchrun, torchrun_env
from .utils.launch import is_torchrun_env, log_launch_info, log_prelaunch_info, resolve_accelerator


@hydra.main(config_path="../../conf", config_name="config", version_base="1.3")
def main(cfg) -> None:
    cfg = align_cfg_for_torchrun(cfg)
    devices_cfg = int(getattr(cfg.train, "devices", 1))
    accelerator_cfg = str(getattr(cfg.train, "accelerator", "cuda"))
    strategy_cfg = str(getattr(cfg.train, "strategy", "auto"))
    strict_launch = bool(getattr(cfg.train, "strict_launch", False))

    num_nodes = 1
    local_rank = None
    if is_torchrun_env():
        if not torch.cuda.is_available():
            raise RuntimeError("torchrun detected but CUDA is not available")
        env = torchrun_env()
        local_rank = int(env["local_rank"])
        local_world_size = int(env["local_world_size"])
        world_size = int(env["world_size"])
        num_nodes = max(1, world_size // local_world_size)
        log_prelaunch_info(local_rank)
        torch.cuda.set_device(local_rank)
        devices = local_world_size
        accelerator = "cuda"
        strategy = str(getattr(cfg.train, "strategy", "ddp"))
    else:
        devices = devices_cfg
        if devices > 1:
            raise RuntimeError("train.devices>1 requires torchrun; do not rely on Fabric spawn")
        accelerator = resolve_accelerator(accelerator_cfg)
        strategy = strategy_cfg
        log_prelaunch_info(None)

    if strict_launch and accelerator == "cuda":
        visible = torch.cuda.device_count()
        if devices > visible:
            raise ValueError(
                f"train.devices={devices} exceeds visible CUDA devices={visible}. "
                "Adjust train.devices or set CUDA_VISIBLE_DEVICES."
            )

    fabric = Fabric(
        accelerator=accelerator,
        devices=devices,
        strategy=strategy,
        num_nodes=num_nodes,
        precision=str(cfg.train.precision),
    )
    fabric.launch()

    if strict_launch and (not is_torchrun_env()) and devices == 1 and fabric.world_size != 1:
        raise RuntimeError(
            f"strict_launch: expected world_size=1 but got {fabric.world_size}. "
            "DDP likely mis-launched."
        )

    log_launch_info(fabric)


if __name__ == "__main__":
    main()
