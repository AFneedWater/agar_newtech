from __future__ import annotations
from typing import Tuple
from torch.utils.data import DataLoader

from .agar_dataset import COCODetectionDataset
from .coco_paths import resolve_coco_train_val
from .transforms import Compose, RandomHorizontalFlip


def detection_collate(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)


def build_dataloaders(cfg) -> Tuple[DataLoader, DataLoader]:
    train_tf = Compose([RandomHorizontalFlip(p=0.5)])
    val_tf = None
    source = getattr(cfg.data, "source", "coco")

    if source == "coco":
        train_paths, val_paths = resolve_coco_train_val(cfg.data)
        train_ds = COCODetectionDataset(
            image_root=str(train_paths.image_root),
            ann_file=str(train_paths.ann_file),
            transforms=train_tf,
        )
        val_ds = COCODetectionDataset(
            image_root=str(val_paths.image_root),
            ann_file=str(val_paths.ann_file),
            transforms=val_tf,
        )
    elif source == "fiftyone":
        raise NotImplementedError("FiftyOne-backed dataloader is not implemented in this stage.")
    else:
        raise ValueError(f"Unknown data.source: {source}")

    num_workers = int(cfg.data.num_workers)
    persistent = bool(cfg.data.persistent_workers) and num_workers > 0

    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg.data.batch_size),
        shuffle=True,
        num_workers=num_workers,
        pin_memory=bool(cfg.data.pin_memory),
        persistent_workers=persistent,
        collate_fn=detection_collate,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg.data.batch_size),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=bool(cfg.data.pin_memory),
        persistent_workers=persistent,
        collate_fn=detection_collate,
    )
    return train_loader, val_loader
