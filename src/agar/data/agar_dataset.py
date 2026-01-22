from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from PIL import Image


class COCODetectionDataset(torch.utils.data.Dataset):
    """
    Minimal COCO detection dataset for torchvision detection models.
    Expects:
      - image_root: root directory containing image files referenced by COCO's `file_name`
      - ann_file: COCO annotations json (bbox in xywh)
    Produces:
      image: FloatTensor[C,H,W] in [0,1]
      target: dict with keys boxes (xyxy), labels, image_id, area, iscrowd
    """

    def __init__(self, image_root: str, ann_file: str, transforms=None):
        self.image_root = Path(image_root)
        self.ann_file = Path(ann_file)
        self.transforms = transforms

        with self.ann_file.open("r", encoding="utf-8") as f:
            coco = json.load(f)

        self.images = coco["images"]
        self.annotations = coco["annotations"]
        self.categories = coco.get("categories", [])

        # Map category_id -> contiguous label id in [1..K]
        cat_ids = [c["id"] for c in self.categories] if self.categories else sorted(
            {a["category_id"] for a in self.annotations}
        )
        self.cat_id_to_label = {cid: i + 1 for i, cid in enumerate(sorted(cat_ids))}

        # group annotations by image_id
        self.ann_by_image: Dict[int, List[dict]] = {}
        for ann in self.annotations:
            self.ann_by_image.setdefault(ann["image_id"], []).append(ann)

    def __len__(self) -> int:
        return len(self.images)

    def _load_image(self, file_name: str) -> Image.Image:
        p = Path(file_name)
        path = p if p.is_absolute() else (self.image_root / p)
        img = Image.open(path).convert("RGB")
        return img

    @staticmethod
    def _xywh_to_xyxy(box: List[float]) -> List[float]:
        x, y, w, h = box
        return [x, y, x + w, y + h]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, Any]]:
        info = self.images[idx]
        image_id = int(info["id"])
        file_name = info["file_name"]

        img = self._load_image(file_name)
        img_w, img_h = img.size

        anns = self.ann_by_image.get(image_id, [])
        boxes = []
        labels = []
        areas = []
        iscrowd = []

        for ann in anns:
            if ann.get("iscrowd", 0) == 1:
                # keep it; COCO eval can handle crowd via iscrowd
                pass
            bbox = ann["bbox"]
            xyxy = self._xywh_to_xyxy(bbox)
            x1 = max(0.0, float(xyxy[0]))
            y1 = max(0.0, float(xyxy[1]))
            x2 = min(float(img_w), float(xyxy[2]))
            y2 = min(float(img_h), float(xyxy[3]))
            # filter invalid after clip
            if x2 <= x1 or y2 <= y1:
                continue
            boxes.append([x1, y1, x2, y2])
            labels.append(self.cat_id_to_label.get(int(ann["category_id"]), 1))
            areas.append(float((x2 - x1) * (y2 - y1)))
            iscrowd.append(int(ann.get("iscrowd", 0)))

        if len(boxes) == 0:
            boxes_t = torch.zeros((0, 4), dtype=torch.float32)
            labels_t = torch.zeros((0,), dtype=torch.int64)
            areas_t = torch.zeros((0,), dtype=torch.float32)
            iscrowd_t = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes_t = torch.tensor(boxes, dtype=torch.float32)
            labels_t = torch.tensor(labels, dtype=torch.int64)
            areas_t = torch.tensor(areas, dtype=torch.float32)
            iscrowd_t = torch.tensor(iscrowd, dtype=torch.int64)

        # basic to tensor
        img_t = torch.from_numpy(__import__("numpy").array(img)).permute(2, 0, 1).float() / 255.0

        target: Dict[str, Any] = {
            "boxes": boxes_t,
            "labels": labels_t,
            "image_id": torch.tensor([image_id], dtype=torch.int64),
            "area": areas_t,
            "iscrowd": iscrowd_t,
        }

        if self.transforms is not None:
            img_t, target = self.transforms(img_t, target)

        return img_t, target
