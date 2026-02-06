from __future__ import annotations
from typing import Tuple
from torch.utils.data import ConcatDataset, DataLoader

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

    resize_max_size = getattr(cfg.data, "resize_max_size", 1024)

    if source == "coco":
        coco_datasets = getattr(cfg.data, "coco_datasets", None)
        if coco_datasets:
            train_datasets = []
            val_datasets = []
            for ds_cfg in coco_datasets:
                train_paths, val_paths = resolve_coco_train_val(ds_cfg)
                train_datasets.append(
                    COCODetectionDataset(
                        image_root=str(train_paths.image_root),
                        ann_file=str(train_paths.ann_file),
                        transforms=train_tf,
                        resize_max_size=resize_max_size,
                    )
                )
                val_datasets.append(
                    COCODetectionDataset(
                        image_root=str(val_paths.image_root),
                        ann_file=str(val_paths.ann_file),
                        transforms=val_tf,
                        resize_max_size=resize_max_size,
                    )
                )

            ref_ids = sorted(train_datasets[0].cat_id_to_label.keys())
            for ds in train_datasets[1:]:
                if sorted(ds.cat_id_to_label.keys()) != ref_ids:
                    raise ValueError(
                        "Mismatched category IDs across coco_datasets. "
                        "Ensure all datasets share identical COCO categories."
                    )

            train_ds = ConcatDataset(train_datasets)

            concat_val = bool(getattr(cfg.data, "concat_val", True))
            if concat_val:
                val_ds = ConcatDataset(val_datasets)
            else:
                val_ds = val_datasets[0]
        else:
            train_paths, val_paths = resolve_coco_train_val(cfg.data)
            train_ds = COCODetectionDataset(
                image_root=str(train_paths.image_root),
                ann_file=str(train_paths.ann_file),
                transforms=train_tf,
                resize_max_size=resize_max_size,
            )
            val_ds = COCODetectionDataset(
                image_root=str(val_paths.image_root),
                ann_file=str(val_paths.ann_file),
                transforms=val_tf,
                resize_max_size=resize_max_size,
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
