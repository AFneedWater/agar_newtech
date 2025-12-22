from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Tuple
import torch
import random


@dataclass
class RandomHorizontalFlip:
    p: float = 0.5

    def __call__(
        self, image: torch.Tensor, target: Dict[str, Any]
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        if random.random() > self.p:
            return image, target
        _, h, w = image.shape
        image = torch.flip(image, dims=[2])
        boxes = target["boxes"]
        if boxes.numel() > 0:
            x1 = boxes[:, 0].clone()
            x2 = boxes[:, 2].clone()
            boxes[:, 0] = (w - x2)
            boxes[:, 2] = (w - x1)
            target["boxes"] = boxes
        return image, target


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target
