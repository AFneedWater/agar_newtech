from __future__ import annotations
from pathlib import Path
from typing import Any, Dict
import torch


def ensure_dir(p: str | Path) -> Path:
    path = Path(p)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_checkpoint(path: str | Path, state: Dict[str, Any]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    torch.save(state, path)


def load_checkpoint(path: str | Path, map_location: str = "cpu") -> Dict[str, Any]:
    return torch.load(Path(path), map_location=map_location)
