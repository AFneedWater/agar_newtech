from __future__ import annotations
import json
import os
from typing import Any, Dict, List

import hydra
from omegaconf import OmegaConf
from lightning_fabric import Fabric
import numpy as np
import torch

from .utils.tf32 import configure_tf32

configure_tf32(enable=True)

from .utils.seed import set_seed
from .data.datamodule import build_dataloaders
from .data.coco_paths import resolve_coco_train_val
from .models.factory import build_model
from .engine.fabric_loop import train as train_loop
from .engine.callbacks import CheckpointCallback
from .tracking.tb_logger import TBLogger
from .tracking.mlflow_logger import MLflowLogger
from .utils.distributed import align_cfg_for_torchrun, torchrun_env
from .utils.launch import is_torchrun_env, log_launch_info, log_prelaunch_info, resolve_accelerator
from .utils.rich_log import log


def _flatten_cfg(cfg) -> dict:
    return OmegaConf.to_container(cfg, resolve=True)


def _list_classes(cfg) -> List[str] | None:
    classes = getattr(cfg.data, "classes", None)
    if classes is None:
        return None
    return [str(c) for c in list(classes)]


def _load_coco_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _build_category_mapping(
    categories: List[Dict[str, Any]],
    annotations: List[Dict[str, Any]],
    classes: List[str] | None,
) -> Dict[int, str]:
    if categories and any("name" in c for c in categories):
        return {int(c["id"]): str(c.get("name", c.get("id", ""))) for c in categories if "id" in c}
    if classes:
        return {i + 1: name for i, name in enumerate(classes)}
    if categories:
        return {int(c["id"]): str(c.get("id", "")) for c in categories if "id" in c}
    ids = sorted({int(ann.get("category_id", -1)) for ann in annotations if "category_id" in ann})
    return {cid: str(cid) for cid in ids}


def _compute_stats(json_path: str, classes: List[str] | None) -> Dict[str, Any]:
    coco = _load_coco_json(json_path)
    images = coco.get("images", [])
    annotations = coco.get("annotations", [])
    categories = coco.get("categories", [])

    mapping = _build_category_mapping(categories, annotations, classes)
    class_counts: Dict[str, int] = {}
    areas: List[float] = []

    for ann in annotations:
        cid = int(ann.get("category_id", -1))
        name = mapping.get(cid, str(cid))
        class_counts[name] = class_counts.get(name, 0) + 1
        area = ann.get("area", None)
        if area is None:
            bbox = ann.get("bbox", [0, 0, 0, 0])
            area = float(bbox[2] * bbox[3])
        areas.append(float(area))

    if areas:
        p10, p50, p90 = np.percentile(np.array(areas), [10, 50, 90]).tolist()
    else:
        p10 = p50 = p90 = 0.0

    return {
        "images": len(images),
        "annotations": len(annotations),
        "categories": len(mapping) if mapping else len(categories),
        "class_counts": class_counts,
        "area_quantiles": {"p10": float(p10), "p50": float(p50), "p90": float(p90)},
        "category_mapping": mapping,
    }


def _log_data_stats(cfg, tb_logger, mlflow_logger) -> None:
    classes = _list_classes(cfg)
    try:
        train_paths, val_paths = resolve_coco_train_val(cfg.data)
    except Exception:
        train_paths = val_paths = None

    if (
        train_paths is None
        or val_paths is None
        or (not train_paths.ann_file.exists())
        or (not val_paths.ann_file.exists())
    ):
        log("Data stats skipped: train/val json not found")
        return

    train_stats = _compute_stats(str(train_paths.ann_file), classes)
    val_stats = _compute_stats(str(val_paths.ann_file), classes)
    stats = {"train": train_stats, "val": val_stats}

    log("Data stats:")
    log(json.dumps(stats, ensure_ascii=True, indent=2))

    if tb_logger:
        for split, split_stats in stats.items():
            metrics = {
                "images": split_stats["images"],
                "annotations": split_stats["annotations"],
                "categories": split_stats["categories"],
                "area_p10": split_stats["area_quantiles"]["p10"],
                "area_p50": split_stats["area_quantiles"]["p50"],
                "area_p90": split_stats["area_quantiles"]["p90"],
            }
            for name, count in split_stats["class_counts"].items():
                metrics[f"class_count/{name}"] = count
            tb_logger.log_metrics(metrics, step=0, prefix=f"data/{split}/")

    stats_path = os.path.join(os.getcwd(), "data_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=True, indent=2)
    if mlflow_logger:
        mlflow_logger.log_artifact(stats_path)


@hydra.main(config_path="../../conf", config_name="config", version_base="1.3")
def main(cfg):
    set_seed(int(cfg.seed))
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

    train_loader, val_loader = build_dataloaders(cfg)

    model = build_model(cfg)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=float(cfg.train.lr),
        momentum=float(cfg.train.momentum),
        weight_decay=float(cfg.train.weight_decay),
    )

    model, optimizer = fabric.setup(model, optimizer)

    tb_logger = None
    if bool(cfg.train.tensorboard) and fabric.is_global_zero:
        tb_logger = TBLogger(log_dir=os.path.join(os.getcwd(), "tb"))

    mlflow_logger = None
    if bool(cfg.train.mlflow) and fabric.is_global_zero:
        mlflow_logger = MLflowLogger(
            tracking_uri=str(cfg.train.mlflow_tracking_uri),
            experiment=str(cfg.train.mlflow_experiment),
            run_name=str(cfg.train.mlflow_run_name),
        )
        mlflow_logger.log_params({"config": OmegaConf.to_yaml(cfg)})

    ckpt_cb = CheckpointCallback(out_dir=os.path.join(os.getcwd(), "checkpoints"), monitor=str(cfg.eval.metric_key))

    if fabric.is_global_zero:
        log(f"Output dir: {os.getcwd()}")
        log(OmegaConf.to_yaml(cfg))
        _log_data_stats(cfg, tb_logger, mlflow_logger)

    out = train_loop(
        fabric=fabric,
        cfg=cfg,
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        tb_logger=tb_logger,
        mlflow_logger=mlflow_logger,
        ckpt_cb=ckpt_cb,
    )

    if tb_logger:
        tb_logger.close()
    if mlflow_logger:
        mlflow_logger.close()

    if fabric.is_global_zero:
        log(f"Done. best_metric={out['best_metric']} global_step={out['global_step']}")


if __name__ == "__main__":
    main()
