from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from hydra import compose, initialize


def _load_coco(ann_json: Path) -> Tuple[Dict[int, Tuple[float, float]], List[List[float]]]:
    with ann_json.open("r", encoding="utf-8") as f:
        coco = json.load(f)

    images = coco.get("images", [])
    annotations = coco.get("annotations", [])

    id_to_size: Dict[int, Tuple[float, float]] = {}
    for img in images:
        img_id = int(img["id"])
        w = float(img["width"])
        h = float(img["height"])
        id_to_size[img_id] = (w, h)

    bboxes: List[List[float]] = []
    for ann in annotations:
        if ann.get("iscrowd", 0) == 1:
            continue
        bbox = ann.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        img_id = int(ann["image_id"])
        if img_id not in id_to_size:
            continue
        bboxes.append([img_id, float(bbox[2]), float(bbox[3])])  # store w, h

    return id_to_size, bboxes


def _compute_ratios(id_to_size: Dict[int, Tuple[float, float]], bboxes: List[List[float]]) -> np.ndarray:
    ratios = []
    for img_id, bw, bh in bboxes:
        img_w, img_h = id_to_size[img_id]
        if img_w <= 0 or img_h <= 0:
            continue
        rw = max(0.0, min(1.0, bw / img_w))
        rh = max(0.0, min(1.0, bh / img_h))
        if rw == 0.0 or rh == 0.0:
            continue
        ratios.append([rw, rh])
    if not ratios:
        return np.zeros((0, 2), dtype=np.float32)
    return np.asarray(ratios, dtype=np.float32)


def _summarize(ratios: np.ndarray) -> str:
    if ratios.size == 0:
        return "No valid boxes found."
    area = ratios[:, 0] * ratios[:, 1]
    stats = {
        "count": int(ratios.shape[0]),
        "w_mean": float(ratios[:, 0].mean()),
        "h_mean": float(ratios[:, 1].mean()),
        "area_mean": float(area.mean()),
        "w_p50": float(np.percentile(ratios[:, 0], 50)),
        "h_p50": float(np.percentile(ratios[:, 1], 50)),
        "area_p50": float(np.percentile(area, 50)),
        "w_p90": float(np.percentile(ratios[:, 0], 90)),
        "h_p90": float(np.percentile(ratios[:, 1], 90)),
        "area_p90": float(np.percentile(area, 90)),
    }
    lines = [f"{k}: {v:.6f}" if isinstance(v, float) else f"{k}: {v}" for k, v in stats.items()]
    return "\n".join(lines)


def _plot_scatter(ratios: np.ndarray, out_path: Path, title: str, max_points: int, seed: int) -> None:
    import matplotlib.pyplot as plt

    if ratios.size == 0:
        raise ValueError("No valid boxes to plot.")

    if max_points > 0 and ratios.shape[0] > max_points:
        rng = np.random.default_rng(seed)
        idx = rng.choice(ratios.shape[0], size=max_points, replace=False)
        ratios = ratios[idx]

    area = ratios[:, 0] * ratios[:, 1]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(7.5, 7.0), dpi=150)
    fig.patch.set_facecolor("#f7f4ef")
    ax.set_facecolor("#fbfaf6")

    sc = ax.scatter(
        ratios[:, 0],
        ratios[:, 1],
        s=10,
        c=area,
        cmap="viridis",
        alpha=0.35,
        edgecolors="none",
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("bbox width / image width")
    ax.set_ylabel("bbox height / image height")
    ax.set_title(title)

    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("relative area")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.6)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)


def _compose_ann_json(experiment: str, split: str) -> Path:
    project_root = Path(__file__).resolve().parents[4]
    conf_dir = project_root / "conf"
    if not conf_dir.exists():
        raise FileNotFoundError(f"Missing conf dir: {conf_dir}")

    with initialize(version_base=None, config_path=str(conf_dir)):
        cfg = compose(config_name="config", overrides=[f"experiment={experiment}"])

    if split not in {"train", "val"}:
        raise ValueError(f"Unknown split: {split}")
    ann_path = Path(getattr(cfg.data, f"{split}_json"))
    if not ann_path.exists():
        raise FileNotFoundError(f"Missing annotations json: {ann_path}")
    return ann_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot relative bbox sizes from COCO annotations.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--ann-json", help="Path to COCO annotations json.")
    group.add_argument("--experiment", help="Experiment config name, e.g., exp_agarv2_resnet50.")
    parser.add_argument("--out", required=True, help="Output image path, e.g., outputs/anchor_scatter.png")
    parser.add_argument("--title", default="Relative BBox Size Scatter", help="Plot title.")
    parser.add_argument("--split", default="train", choices=["train", "val"], help="Dataset split.")
    parser.add_argument("--max-points", type=int, default=20000, help="Downsample points for plotting.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for downsampling.")
    args = parser.parse_args()

    if args.ann_json:
        ann_json = Path(args.ann_json)
    else:
        ann_json = _compose_ann_json(args.experiment, args.split)
    out_path = Path(args.out)

    id_to_size, bboxes = _load_coco(ann_json)
    ratios = _compute_ratios(id_to_size, bboxes)

    print(_summarize(ratios))
    _plot_scatter(ratios, out_path, args.title, args.max_points, args.seed)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
