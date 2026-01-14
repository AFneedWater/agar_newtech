from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from hydra import compose, initialize_config_dir


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


def _transform_scale(values: np.ndarray, mode: str) -> Tuple[np.ndarray, float]:
    if mode == "sqrt":
        return np.sqrt(values), 0.0
    if mode == "log":
        eps = 1e-6
        return np.log10(values + eps), eps
    return values, 0.0


def _inverse_scale(values: np.ndarray, mode: str, eps: float) -> np.ndarray:
    if mode == "sqrt":
        return np.square(values)
    if mode == "log":
        return np.power(10.0, values) - eps
    return values


def _plot_scatter(
    ratios: np.ndarray,
    out_path: Path,
    title: str,
    max_points: int,
    seed: int,
    scale: str,
    clip_pct: float,
) -> None:
    import matplotlib.pyplot as plt

    if ratios.size == 0:
        raise ValueError("No valid boxes to plot.")

    # Downsample points for speed / readability
    if max_points > 0 and ratios.shape[0] > max_points:
        rng = np.random.default_rng(seed)
        idx = rng.choice(ratios.shape[0], size=max_points, replace=False)
        ratios = ratios[idx]

    x = ratios[:, 0]
    y = ratios[:, 1]
    area = x * y

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(7.5, 7.0), dpi=150)
    fig.patch.set_facecolor("#f7f4ef")
    ax.set_facecolor("#fbfaf6")

    # Transform axes to spread dense small ratios
    x_t, eps = _transform_scale(x, scale)
    y_t, _ = _transform_scale(y, scale)

    sc = ax.scatter(
        x_t,
        y_t,
        s=8,            # smaller points help dense regions
        c=area,
        cmap="viridis",
        alpha=0.20,     # lower alpha shows density layering
        edgecolors="none",
    )

    # ====== Key: clip axis upper limits by percentile to avoid a few large boxes dominating the range ======
    if clip_pct and 0.0 < clip_pct < 100.0:
        x_hi = float(np.percentile(x, clip_pct))
        y_hi = float(np.percentile(y, clip_pct))

        # Choose a safe lower bound for log scale; linear/sqrt can start at 0
        x_lo = 1e-6 if scale == "log" else 0.0
        y_lo = 1e-6 if scale == "log" else 0.0

        xlim_t, _ = _transform_scale(np.asarray([x_lo, x_hi], dtype=np.float32), scale)
        ylim_t, _ = _transform_scale(np.asarray([y_lo, y_hi], dtype=np.float32), scale)
        ax.set_xlim(float(xlim_t[0]), float(xlim_t[1]))
        ax.set_ylim(float(ylim_t[0]), float(ylim_t[1]))
    else:
        ax.set_xlim(x_t.min() * 0.98, x_t.max() * 1.02)
        ax.set_ylim(y_t.min() * 0.98, y_t.max() * 1.02)

    ax.set_xlabel("bbox width / image width")
    ax.set_ylabel("bbox height / image height")
    ax.set_title(title)

    # Keep tick labels in original (0..1) space for interpretability
    if scale in {"sqrt", "log"}:
        ticks = np.linspace(0, 1, 6)
        xt = _transform_scale(ticks, scale)[0]
        yt = _transform_scale(ticks, scale)[0]
        ax.set_xticks(xt)
        ax.set_yticks(yt)
        ax.set_xticklabels([f"{t:.2f}" for t in ticks])
        ax.set_yticklabels([f"{t:.2f}" for t in ticks])

    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("relative area")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.6)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)


def _plot_hist2d(
    ratios: np.ndarray,
    out_path: Path,
    title: str,
    bins: int,
    density_log: bool,
    clip_pct: float,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    if ratios.size == 0:
        raise ValueError("No valid boxes to plot.")

    x = ratios[:, 0]
    y = ratios[:, 1]
    x_hi = float(np.percentile(x, clip_pct))
    y_hi = float(np.percentile(y, clip_pct))
    x_hi = max(x_hi, 1e-6)
    y_hi = max(y_hi, 1e-6)

    fig, ax = plt.subplots(figsize=(7.2, 6.8), dpi=150)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    norm = LogNorm() if density_log else None
    ax.hist2d(
        x,
        y,
        bins=bins,
        range=[[0.0, x_hi], [0.0, y_hi]],
        cmap="Blues",
        norm=norm,
        cmin=1,
    )
    ax.set_xlim(0.0, x_hi)
    ax.set_ylim(0.0, y_hi)
    ax.set_xlabel("width")
    ax.set_ylabel("height")
    ax.set_title(title)
    ax.grid(False)

    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
        spine.set_color("black")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)


def _compose_ann_json(experiment: str, split: str) -> Path:
    project_root = Path(__file__).resolve().parents[3]
    conf_dir = project_root / "conf"
    if not conf_dir.exists():
        raise FileNotFoundError(f"Missing conf dir: {conf_dir}")

    with initialize_config_dir(version_base=None, config_dir=str(conf_dir)):
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
    parser.add_argument(
        "--plot",
        default="hist2d",
        choices=["scatter", "hist2d"],
        help="Plot style.",
    )
    parser.add_argument("--bins", type=int, default=70, help="Number of bins for hist2d.")
    parser.add_argument(
        "--density-log",
        action="store_true",
        default=True,
        help="Use log normalization for hist2d density.",
    )
    parser.add_argument(
        "--scale",
        default="sqrt",
        choices=["linear", "sqrt", "log"],
        help="Axis scale to spread small boxes; sqrt is good for dense small ratios.",
    )
    parser.add_argument("--max-points", type=int, default=20000, help="Downsample points for plotting.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for downsampling.")
    parser.add_argument(
        "--clip-pct",
        type=float,
        default=99.0,
        help="Clip axis upper limit to this percentile (0 disables). Recommended: 99.0~99.7.",
    )
    args = parser.parse_args()

    if args.ann_json:
        ann_json = Path(args.ann_json)
    else:
        ann_json = _compose_ann_json(args.experiment, args.split)
    out_path = Path(args.out)

    id_to_size, bboxes = _load_coco(ann_json)
    ratios = _compute_ratios(id_to_size, bboxes)

    print(_summarize(ratios))
    if args.plot == "hist2d":
        _plot_hist2d(
            ratios=ratios,
            out_path=out_path,
            title=args.title,
            bins=args.bins,
            density_log=args.density_log,
            clip_pct=args.clip_pct,
        )
    else:
        _plot_scatter(
            ratios=ratios,
            out_path=out_path,
            title=args.title,
            max_points=args.max_points,
            seed=args.seed,
            scale=args.scale,
            clip_pct=args.clip_pct,
        )
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
