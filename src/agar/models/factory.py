from __future__ import annotations
import torchvision

from .detector import DetectorWrapper


def build_model(cfg):
    name = cfg.model.name
    num_fg = int(cfg.model.num_classes)
    num_classes = num_fg + 1  # + background

    if name == "fasterrcnn_resnet50_fpn_v2":
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(
            weights="DEFAULT" if bool(cfg.model.pretrained) else None,
            num_classes=num_classes,
        )
    else:
        raise ValueError(f"Unknown model name: {name}")

    return DetectorWrapper(model)
