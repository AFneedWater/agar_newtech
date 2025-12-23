from __future__ import annotations

import os
import sys
from typing import Iterable, List, Set


def _parse_cpu_list(spec: str) -> List[int]:
    cpus: Set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            try:
                start = int(start_s)
                end = int(end_s)
            except ValueError:
                continue
            if end < start:
                start, end = end, start
            cpus.update(range(start, end + 1))
        else:
            try:
                cpus.add(int(part))
            except ValueError:
                continue
    return sorted(cpus)


def _set_affinity(cpus: Iterable[int]) -> None:
    try:
        os.sched_setaffinity(0, set(cpus))
        print(f"[pin_and_exec] pinned to CPUs: {','.join(map(str, sorted(cpus)))}", flush=True)
    except Exception as exc:  # pragma: no cover - depends on OS/kernel
        print(f"[pin_and_exec] WARNING: failed to set CPU affinity: {exc}", flush=True)


def main() -> None:
    args = sys.argv[1:]
    if "--" in args:
        idx = args.index("--")
        cmd = args[idx + 1 :]
    else:
        cmd = args

    if not cmd:
        raise SystemExit("Usage: python -m agar.tools.pin_and_exec -- <command>")

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", str(local_rank)))
    if os.environ.get("DDP_PIN_CPU", "0") == "1":
        affinity = ""
        affinity_list = os.environ.get("DDP_CPU_AFFINITIES", "")
        if not affinity_list:
            affinity_list = os.environ.get("DDP_CPU_AFFINITY_LIST", "")
        if affinity_list:
            entries = affinity_list.split(";")
            if local_rank < len(entries):
                affinity = entries[local_rank].strip()
        if not affinity:
            affinity = os.environ.get(f"DDP_CPU_AFFINITY_GPU{local_rank}", "")

        if affinity and affinity.lower() != "n/a":
            cpus = _parse_cpu_list(affinity)
            if cpus:
                _set_affinity(cpus)
                print(
                    f"[pin_and_exec] rank={rank} local_rank={local_rank} applied affinity={affinity}",
                    flush=True,
                )
            else:
                print(
                    f"[pin_and_exec] WARNING: invalid CPU affinity list: {affinity}",
                    flush=True,
                )
        else:
            print(
                f"[pin_and_exec] WARNING: no CPU affinity for rank={rank} local_rank={local_rank}; skip pinning",
                flush=True,
            )

    if cmd[0] == "-m":
        cmd = [sys.executable] + cmd

    os.execvp(cmd[0], cmd)


if __name__ == "__main__":
    main()
