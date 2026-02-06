from __future__ import annotations
import json
import os
import hashlib
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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


def _file_sha256(path: str) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _find_repo_root(start: Path) -> Optional[Path]:
    cur = start
    for _ in range(8):
        if (cur / ".git").exists():
            return cur
        if cur.parent == cur:
            break
        cur = cur.parent
    return None


def _git_sha_short() -> str:
    repo_root = _find_repo_root(Path(__file__).resolve())
    if repo_root is None:
        return ""
    try:
        out = subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        )
        return out.decode("utf-8").strip()
    except Exception:
        return ""


def _fmt_float(value: Any) -> str:
    try:
        return f"{float(value):.6g}"
    except Exception:
        return str(value)


def _make_run_name(cfg, git_sha: str) -> str:
    model = str(getattr(cfg.model, "name", "model"))
    data = str(getattr(cfg.data, "name", "data"))
    img = str(getattr(cfg.data, "resize_max_size", "na"))
    bs = str(getattr(cfg.data, "batch_size", "na"))
    lr = _fmt_float(getattr(cfg.train, "lr", "na"))
    seed = str(getattr(cfg, "seed", "na"))
    ts = time.strftime("%Y%m%d-%H%M%S")

    parts = [model, data, f"img{img}", f"bs{bs}", f"lr{lr}", f"seed{seed}"]
    if git_sha:
        parts.append(git_sha)
    parts.append(ts)
    name = "-".join(parts)
    name = name.replace("/", "-").replace(" ", "")
    return name[:200]


def _collect_mlflow_params(cfg) -> Dict[str, Any]:
    return {
        "model_name": getattr(cfg.model, "name", ""),
        "model_num_classes": getattr(cfg.model, "num_classes", ""),
        "data_name": getattr(cfg.data, "name", ""),
        "data_source": getattr(cfg.data, "source", ""),
        "data_resize_max_size": getattr(cfg.data, "resize_max_size", ""),
        "data_batch_size": getattr(cfg.data, "batch_size", ""),
        "data_num_workers": getattr(cfg.data, "num_workers", ""),
        "data_train_ann": getattr(cfg.data, "train_ann", ""),
        "data_val_ann": getattr(cfg.data, "val_ann", ""),
        "train_lr": getattr(cfg.train, "lr", ""),
        "train_weight_decay": getattr(cfg.train, "weight_decay", ""),
        "train_momentum": getattr(cfg.train, "momentum", ""),
        "train_epochs": getattr(cfg.train, "epochs", ""),
        "train_precision": getattr(cfg.train, "precision", ""),
        "train_devices": getattr(cfg.train, "devices", ""),
        "train_strategy": getattr(cfg.train, "strategy", ""),
        "train_grad_accum_steps": getattr(cfg.train, "grad_accum_steps", ""),
        "train_max_steps": getattr(cfg.train, "max_steps", ""),
        "seed": getattr(cfg, "seed", ""),
        "eval_metric_key": getattr(cfg.eval, "metric_key", ""),
    }


def _collect_mlflow_tags(cfg, git_sha: str, data_fingerprints: Optional[Dict[str, str]]) -> Dict[str, Any]:
    tags: Dict[str, Any] = {
        "stage": "train",
        "git_sha": git_sha,
    }
    if data_fingerprints and bool(getattr(cfg.train, "mlflow_log_data_fingerprint", True)):
        tags.update({f"data_fingerprint_{k}": v for k, v in data_fingerprints.items() if v})
    extra = getattr(cfg.train, "mlflow_tags", None) or {}
    if isinstance(extra, dict):
        tags.update(extra)
    return tags


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


def _log_data_stats(cfg, tb_logger, mlflow_logger) -> Tuple[Optional[str], Optional[Dict[str, str]]]:
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
        return None, None

    train_stats = _compute_stats(str(train_paths.ann_file), classes)
    val_stats = _compute_stats(str(val_paths.ann_file), classes)
    fingerprints = {
        "train_ann_sha256": _file_sha256(str(train_paths.ann_file)),
        "val_ann_sha256": _file_sha256(str(val_paths.ann_file)),
    }
    stats = {"train": train_stats, "val": val_stats, "fingerprints": fingerprints}

    log("Data stats:")
    log(json.dumps(stats, ensure_ascii=True, indent=2))

    if tb_logger:
        for split in ("train", "val"):
            split_stats = stats.get(split)
            if not isinstance(split_stats, dict):
                continue
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
    return stats_path, fingerprints


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
    git_sha = _git_sha_short()
    if bool(cfg.train.mlflow) and fabric.is_global_zero:
        run_name = str(cfg.train.mlflow_run_name).strip()
        if not run_name:
            run_name = _make_run_name(cfg, git_sha)
        mlflow_logger = MLflowLogger(
            tracking_uri=str(cfg.train.mlflow_tracking_uri),
            experiment=str(cfg.train.mlflow_experiment),
            run_name=run_name,
        )
        mlflow_logger.log_params(_collect_mlflow_params(cfg))
        if getattr(cfg.train, "mlflow_description", "").strip():
            mlflow_logger.set_tag("mlflow.note.content", str(cfg.train.mlflow_description))

    ckpt_cb = CheckpointCallback(out_dir=os.path.join(os.getcwd(), "checkpoints"), monitor=str(cfg.eval.metric_key))

    data_stats_path = None
    data_fingerprints = None
    if fabric.is_global_zero:
        log(f"Output dir: {os.getcwd()}")
        log(OmegaConf.to_yaml(cfg))
        data_stats_path, data_fingerprints = _log_data_stats(cfg, tb_logger, mlflow_logger)
        if mlflow_logger:
            tags = _collect_mlflow_tags(cfg, git_sha, data_fingerprints)
            mlflow_logger.set_tags(tags)
            if bool(getattr(cfg.train, "mlflow_log_config", True)):
                hydra_dir = os.path.join(os.getcwd(), ".hydra")
                for name in ("config.yaml", "overrides.yaml", "hydra.yaml"):
                    path = os.path.join(hydra_dir, name)
                    if os.path.exists(path):
                        mlflow_logger.log_artifact(path)

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
        if bool(getattr(cfg.train, "mlflow_log_checkpoints", True)):
            ckpt_dir = Path(os.getcwd()) / "checkpoints"
            best_path = ckpt_dir / "checkpoint_best.pt"
            last_path = ckpt_dir / "checkpoint_last.pt"
            log_best_only = bool(getattr(cfg.train, "mlflow_log_checkpoints_best_only", False))
            if best_path.exists():
                mlflow_logger.log_artifact(str(best_path))
            if (not log_best_only) and last_path.exists():
                mlflow_logger.log_artifact(str(last_path))
        mlflow_logger.close()

    if fabric.is_global_zero:
        log(f"Done. best_metric={out['best_metric']} global_step={out['global_step']}")


if __name__ == "__main__":
    main()
