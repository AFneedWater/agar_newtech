from __future__ import annotations
from pathlib import Path


def resolve_path(p: str) -> str:
    return str(Path(p).expanduser().resolve())
