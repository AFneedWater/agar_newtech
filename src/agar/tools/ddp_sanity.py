from __future__ import annotations

import os
from dataclasses import dataclass

import hydra
import torch
import torch.distributed as dist
from hydra.core.config_store import ConfigStore


@dataclass
class DDPConfig:
    backend: str = "nccl"


ConfigStore.instance().store(name="config", node=DDPConfig)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


@hydra.main(config_path=None, config_name="config", version_base="1.3")
def main(cfg: DDPConfig) -> None:
    local_rank = _env_int("LOCAL_RANK", 0)
    rank = _env_int("RANK", 0)
    world_size = _env_int("WORLD_SIZE", 1)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for ddp_sanity")

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    torch.cuda.init()
    torch.cuda.synchronize()
    name = torch.cuda.get_device_name(local_rank)
    print(
        f"pre_init rank={rank} local_rank={local_rank} world_size={world_size} "
        f"device={local_rank} name={name} backend={cfg.backend}",
        flush=True,
    )

    initialized = False
    try:
        init_kwargs = {"backend": cfg.backend, "rank": rank, "world_size": world_size}
        if cfg.backend == "nccl":
            init_kwargs["device_id"] = device
        try:
            dist.init_process_group(**init_kwargs)
        except TypeError:
            init_kwargs.pop("device_id", None)
            dist.init_process_group(**init_kwargs)
        initialized = True
        print(
            f"post_init rank={rank} local_rank={local_rank} world_size={world_size} "
            f"device={local_rank} name={name} backend={cfg.backend}",
            flush=True,
        )

        tensor = torch.ones(1, device=device)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        try:
            if cfg.backend == "nccl":
                dist.barrier(device_ids=[local_rank])
            else:
                dist.barrier()
        except TypeError:
            dist.barrier()

        print(
            f"rank={rank} local_rank={local_rank} world_size={world_size} "
            f"device={local_rank} name={name} CUDA_VISIBLE_DEVICES={visible} all_reduce={tensor.item():.1f}",
            flush=True,
        )
    finally:
        if initialized and dist.is_initialized():
            print(
                f"destroy_process_group: start rank={rank} local_rank={local_rank} world_size={world_size} "
                f"device={local_rank} name={name} backend={cfg.backend}",
                flush=True,
            )
            dist.destroy_process_group()
            print(
                f"destroy_process_group: done rank={rank} local_rank={local_rank} world_size={world_size} "
                f"device={local_rank} name={name} backend={cfg.backend}",
                flush=True,
            )


if __name__ == "__main__":
    main()
