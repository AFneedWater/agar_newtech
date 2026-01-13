from __future__ import annotations
import torchvision

from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

from .detector import DetectorWrapper


def build_model(cfg):
    name = cfg.model.name
    num_fg = int(cfg.model.num_classes)
    num_classes = num_fg + 1  # + background
    pretrained = bool(getattr(cfg.model, "pretrained", False))
    weights_cfg = getattr(cfg.model, "weights", None)

    if name == "fasterrcnn_resnet50_fpn_v2":
        weights = None
        weights_backbone = None
        if pretrained:
            weights = weights_cfg or "DEFAULT"
            if num_classes != 91:
                weights = None
                weights_backbone = "DEFAULT"
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(
            weights=weights,
            weights_backbone=weights_backbone,
            num_classes=num_classes,
        )
    elif name == "fasterrcnn_resnet101_fpn":
        weights_backbone = None
        if pretrained:
            weights_backbone = weights_cfg or "DEFAULT"
        backbone = resnet_fpn_backbone("resnet101", weights=weights_backbone)
        model = FasterRCNN(backbone, num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model name: {name}")

    return DetectorWrapper(model)
