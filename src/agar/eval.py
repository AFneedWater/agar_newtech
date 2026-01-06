from __future__ import annotations
import json
import os
from typing import Dict, List

import hydra
from lightning_fabric import Fabric
import torch

from .utils.tf32 import configure_tf32

configure_tf32(enable=True)

from .data.datamodule import build_dataloaders
from .models.factory import build_model
from .engine.fabric_loop import evaluate
from .utils.io import load_checkpoint
from .utils.distributed import align_cfg_for_torchrun, torchrun_env
from .utils.launch import is_torchrun_env, log_launch_info, resolve_accelerator
from .utils.rich_log import log


def _list_classes(cfg) -> List[str] | None:
    classes = getattr(cfg.data, "classes", None)
    if classes is None:
        return None
    return [str(c) for c in list(classes)]


def _load_category_mapping(cfg) -> Dict[int, str]:
    classes = _list_classes(cfg)
    val_json = str(getattr(cfg.data, "val_json", ""))
    if val_json and os.path.exists(val_json):
        with open(val_json, "r", encoding="utf-8") as f:
            coco = json.load(f)
        categories = coco.get("categories", [])
        if categories and any("name" in c for c in categories):
            return {
                int(c["id"]): str(c.get("name", c.get("id", "")))
                for c in categories
                if "id" in c
            }
        if classes:
            return {i + 1: name for i, name in enumerate(classes)}
        if categories:
            return {int(c["id"]): str(c.get("id", "")) for c in categories if "id" in c}

    if classes:
        return {i + 1: name for i, name in enumerate(classes)}

    return {}


@hydra.main(config_path="../../conf", config_name="config", version_base="1.3")
def main(cfg):
    cfg = align_cfg_for_torchrun(cfg)
    devices_cfg = int(getattr(cfg.train, "devices", 1))
    accelerator_cfg = str(getattr(cfg.train, "accelerator", "cuda"))
    strategy_cfg = str(getattr(cfg.train, "strategy", "auto"))
    strict_launch = bool(getattr(cfg.train, "strict_launch", False))

    num_nodes = 1
    if is_torchrun_env():
        if not torch.cuda.is_available():
            raise RuntimeError("torchrun detected but CUDA is not available")
        env = torchrun_env()
        local_rank = int(env["local_rank"])
        local_world_size = int(env["local_world_size"])
        world_size = int(env["world_size"])
        num_nodes = max(1, world_size // local_world_size)
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

    _, val_loader = build_dataloaders(cfg)
    model = build_model(cfg)

    ckpt_path = getattr(cfg, "ckpt", "")
    if ckpt_path:
        state = load_checkpoint(ckpt_path, map_location="cpu")
        model.load_state_dict(state["model"], strict=True)

    model = fabric.setup(model)
    val_loader = fabric.setup_dataloaders(val_loader)

    metrics = evaluate(fabric, model, val_loader, return_per_class=True)
    if fabric.is_global_zero:
        cat_mapping = _load_category_mapping(cfg)
        if cat_mapping:
            log("category_id -> name:")
            for cid in sorted(cat_mapping):
                log(f"{cid}: {cat_mapping[cid]}")

        log("Eval metrics:")
        for k, v in metrics.items():
            if k.endswith("_per_class") or k == "classes":
                continue
            if isinstance(v, (int, float)):
                log(f"{k}: {v:.6f}")

        class_ids = metrics.get("classes", [])
        if isinstance(class_ids, list) and class_ids:
            for key, values in metrics.items():
                if not key.endswith("_per_class") or not isinstance(values, list):
                    continue
                log(f"{key}:")
                for cid, val in zip(class_ids, values):
                    name = cat_mapping.get(int(cid), str(cid))
                    log(f"  {cid} ({name}): {float(val):.6f}")


if __name__ == "__main__":
    main()
