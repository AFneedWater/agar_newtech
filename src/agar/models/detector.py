from __future__ import annotations
from typing import Any, Dict, List, Optional
import torch
import torch.nn as nn


class DetectorWrapper(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, images: List[torch.Tensor], targets: Optional[List[Dict[str, Any]]] = None):
        return self.model(images, targets)
