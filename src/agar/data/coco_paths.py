from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Tuple


@dataclass(frozen=True)
class CocoSplitPaths:
    ann_file: Path
    image_root: Path


def _to_path(p: Any) -> Path:
    if p is None:
        return Path()
    s = str(p).strip()
    if not s:
        return Path()
    return Path(s).expanduser()


def _is_set(path: Path) -> bool:
    # Path("") becomes "."; treat that as "unset" for config parsing.
    return str(path) not in {"", "."}


def _resolve(base: Path | None, p: Any) -> Path:
    path = _to_path(p)
    if not _is_set(path):
        return Path()
    if path.is_absolute():
        return path
    if base is None or (not _is_set(base)):
        return path
    return base / path


def resolve_coco_split_paths(cfg_data: Any, split: str) -> CocoSplitPaths:
    """
    Resolve standard COCO input paths from Hydra cfg.

    New (preferred) keys:
      - coco_root: base dir for relative paths (optional)
      - train_ann / val_ann: annotation json (relative to coco_root or absolute)
      - train_images / val_images: image root dir (relative to coco_root or absolute)
      - images_root: shared image root for both splits (optional)

    Back-compat (deprecated):
      - train_json / val_json
      - images_dir
    """

    split = str(split).lower()
    if split not in {"train", "val"}:
        raise ValueError(f"split must be train/val, got: {split}")

    base_candidate = _resolve(None, getattr(cfg_data, "coco_root", ""))
    base = base_candidate if _is_set(base_candidate) else None

    ann_key = f"{split}_ann"
    images_key = f"{split}_images"

    ann = _resolve(base, getattr(cfg_data, ann_key, ""))
    image_root = _resolve(base, getattr(cfg_data, images_key, ""))

    if not _is_set(ann):
        # deprecated
        legacy = "train_json" if split == "train" else "val_json"
        ann = _resolve(base, getattr(cfg_data, legacy, ""))

    if not _is_set(image_root):
        image_root = _resolve(base, getattr(cfg_data, "images_root", ""))

    if not _is_set(image_root):
        # deprecated
        image_root = _resolve(base, getattr(cfg_data, "images_dir", ""))

    if split == "val":
        # default val to train if not provided
        if not _is_set(ann):
            ann = resolve_coco_split_paths(cfg_data, "train").ann_file
        if not _is_set(image_root):
            image_root = resolve_coco_split_paths(cfg_data, "train").image_root

    if not _is_set(ann):
        raise ValueError("Missing COCO annotation path. Set cfg.data.train_ann (and optionally cfg.data.val_ann).")
    if not _is_set(image_root):
        raise ValueError(
            "Missing COCO image root. Set cfg.data.images_root (or cfg.data.train_images/cfg.data.val_images)."
        )

    return CocoSplitPaths(ann_file=ann, image_root=image_root)


def resolve_coco_train_val(cfg_data: Any) -> Tuple[CocoSplitPaths, CocoSplitPaths]:
    return resolve_coco_split_paths(cfg_data, "train"), resolve_coco_split_paths(cfg_data, "val")
