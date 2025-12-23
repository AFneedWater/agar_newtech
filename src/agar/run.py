from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple


def _in_torchrun_env() -> bool:
    return ("LOCAL_RANK" in os.environ) and ("WORLD_SIZE" in os.environ)


def _parse_kv_override(arg: str) -> Tuple[str, str] | None:
    if "=" not in arg:
        return None
    key, value = arg.split("=", 1)
    key = key.lstrip("+").strip()
    return key, value.strip()


def parse_train_devices(overrides: List[str], default: int = 1) -> int:
    for arg in overrides:
        kv = _parse_kv_override(arg)
        if not kv:
            continue
        key, value = kv
        if key == "train.devices":
            try:
                return int(value)
            except ValueError:
                return default
    return default


def _find_override(overrides: List[str], key: str) -> Optional[str]:
    for arg in overrides:
        kv = _parse_kv_override(arg)
        if not kv:
            continue
        k, v = kv
        if k == key:
            return v
    return None


def _ensure_override(overrides: List[str], key: str, value: str) -> List[str]:
    if _find_override(overrides, key) is None:
        return overrides + [f"{key}={value}"]
    return overrides


def _maybe_force_ddp_strategy(overrides: List[str]) -> List[str]:
    v = _find_override(overrides, "train.strategy")
    if v is None:
        return overrides + ["train.strategy=ddp"]
    if v in ("auto", "ddp"):
        return overrides + ["train.strategy=ddp"]
    return overrides


def _exec(cmd: List[str]) -> None:
    os.execvpe(cmd[0], cmd, os.environ.copy())


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _module_for_command(command: str) -> str:
    mapping = {
        "train": "agar.train",
        "eval": "agar.eval",
        "smoke": "agar.smoke",
        "launch_check": "agar.launch_check",
        "ddp_sanity": "agar.tools.ddp_sanity",
    }
    if command not in mapping:
        raise SystemExit(f"Unknown command: {command}")
    return mapping[command]


def main(argv: List[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)

    parser = argparse.ArgumentParser(prog="python -m agar.run")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("train", "eval", "smoke", "launch_check", "ddp_sanity"):
        p = sub.add_parser(name)
        p.add_argument("overrides", nargs=argparse.REMAINDER)

    ns = parser.parse_args(argv)
    command: str = ns.command
    overrides: List[str] = list(ns.overrides or [])

    module = _module_for_command(command)

    if _in_torchrun_env():
        _exec([sys.executable, "-m", module, *overrides])

    devices = parse_train_devices(overrides, default=1)
    if devices <= 1:
        _exec([sys.executable, "-m", module, *overrides])

    root = _repo_root()
    run_ddp = root / "scripts" / "run_ddp.sh"

    aligned_overrides = overrides
    if command in ("train", "eval", "smoke", "launch_check"):
        aligned_overrides = _ensure_override(aligned_overrides, "train.devices", str(devices))
        aligned_overrides = _maybe_force_ddp_strategy(aligned_overrides)

    cmd = [
        "bash",
        str(run_ddp),
        "torchrun",
        f"--nproc_per_node={devices}",
        "-m",
        module,
        *aligned_overrides,
    ]
    _exec(cmd)


if __name__ == "__main__":
    main()

