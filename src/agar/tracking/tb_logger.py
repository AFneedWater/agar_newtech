from __future__ import annotations
from typing import Dict
from torch.utils.tensorboard import SummaryWriter


class TBLogger:
    def __init__(self, log_dir: str):
        self.writer = SummaryWriter(log_dir=log_dir)

    def log_metrics(self, metrics: Dict[str, float], step: int, prefix: str = "") -> None:
        for k, v in metrics.items():
            self.writer.add_scalar(f"{prefix}{k}", float(v), step)

    def close(self) -> None:
        self.writer.close()
