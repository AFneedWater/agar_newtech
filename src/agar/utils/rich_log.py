from __future__ import annotations
from rich.console import Console

console = Console()


def log(msg: str) -> None:
    console.print(msg)
