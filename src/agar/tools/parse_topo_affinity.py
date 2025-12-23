from __future__ import annotations

import argparse
import re
import sys
from typing import Dict, List


ANSI_RE = re.compile(r"\x1B\[[0-9;]*[mK]")


def _strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text)


def _split_tokens(line: str) -> List[str]:
    return [t for t in line.strip().split() if t]


def parse_topo_affinity(text: str) -> Dict[str, str]:
    cleaned = _strip_ansi(text)
    lines = [line for line in cleaned.splitlines() if line.strip()]
    header = ""
    for line in lines:
        if "CPU Affinity" in line:
            header = line
            break
    if not header:
        return {}

    tokens = _split_tokens(header)
    col_idx = None
    for i in range(len(tokens) - 1):
        if tokens[i] == "CPU" and tokens[i + 1] == "Affinity":
            col_idx = i + 1
            break
    if col_idx is None:
        return {}

    out: Dict[str, str] = {}
    for line in lines:
        if "CPU Affinity" in line:
            continue
        parts = _split_tokens(line)
        if not parts:
            continue
        first = parts[0]
        if not (first.startswith("GPU") and first[3:].isdigit()):
            continue
        if col_idx < len(parts):
            value = parts[col_idx]
            if value not in ("N/A", "n/a"):
                out[first] = value
    return out


def _main() -> None:
    parser = argparse.ArgumentParser(description="Parse CPU affinity from nvidia-smi topo -m output.")
    parser.add_argument("--gpu", type=int, help="GPU index to print (0-based).")
    parser.add_argument("--file", type=str, default="", help="Read topo output from file.")
    args = parser.parse_args()

    if args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            text = f.read()
    else:
        text = sys.stdin.read()

    out = parse_topo_affinity(text)
    if args.gpu is None:
        for key in sorted(out.keys(), key=lambda k: int(k[3:]) if k[3:].isdigit() else 0):
            print(f"{key}={out[key]}")
    else:
        value = out.get(f"GPU{args.gpu}", "")
        if value:
            print(value)


if __name__ == "__main__":
    _main()
