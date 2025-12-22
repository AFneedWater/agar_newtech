from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

from ..utils.io import save_checkpoint


@dataclass
class TrainState:
    epoch: int = 0
    global_step: int = 0
    best_metric: float = float("-inf")
    best_path: Optional[str] = None
    last_path: Optional[str] = None


class CheckpointCallback:
    def __init__(self, out_dir: str, monitor: str = "map"):
        self.out_dir = Path(out_dir)
        self.monitor = monitor
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def save_last(self, fabric, state: TrainState, model, optimizer) -> None:
        if not fabric.is_global_zero:
            return
        path = self.out_dir / "checkpoint_last.pt"
        save_checkpoint(
            path,
            {
                "epoch": state.epoch,
                "global_step": state.global_step,
                "best_metric": state.best_metric,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            },
        )
        state.last_path = str(path)

    def maybe_save_best(
        self, fabric, state: TrainState, model, optimizer, metrics: Dict[str, float]
    ) -> None:
        if not fabric.is_global_zero:
            return
        value = float(metrics.get(self.monitor, float("-inf")))
        if value > state.best_metric:
            state.best_metric = value
            path = self.out_dir / "checkpoint_best.pt"
            save_checkpoint(
                path,
                {
                    "epoch": state.epoch,
                    "global_step": state.global_step,
                    "best_metric": state.best_metric,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                },
            )
            state.best_path = str(path)
