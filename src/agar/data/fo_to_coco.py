from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from tqdm import tqdm


def _require_fiftyone():
    try:
        import fiftyone as fo  # type: ignore
    except Exception as exc:  # pragma: no cover - exercised when missing
        raise RuntimeError(
            "fiftyone is not installed. Install it to export COCO from FiftyOne datasets."
        ) from exc
    return fo


def _parse_view_spec(dataset, view_spec: str):
    if not view_spec:
        return dataset
    if view_spec.startswith("tag:"):
        tag = view_spec.split(":", 1)[1]
        if not tag:
            raise ValueError("view_spec 'tag:' must include a tag name")
        return dataset.match_tags(tag)
    raise ValueError(f"Unsupported view_spec: {view_spec}")


def _stable_image_id(sample_id: str) -> int:
    return int(hashlib.md5(sample_id.encode("utf-8")).hexdigest()[:12], 16)


def _as_posix_relpath(path: str, root: str) -> str:
    rel = os.path.relpath(path, root)
    return Path(rel).as_posix()


def _ensure_image_size(sample) -> tuple[int, int]:
    metadata = getattr(sample, "metadata", None)
    width = getattr(metadata, "width", None)
    height = getattr(metadata, "height", None)
    if width is not None and height is not None:
        return int(width), int(height)
    sample.compute_metadata()
    metadata = getattr(sample, "metadata", None)
    width = getattr(metadata, "width", None)
    height = getattr(metadata, "height", None)
    if width is not None and height is not None:
        return int(width), int(height)
    try:
        from PIL import Image
    except Exception as exc:
        raise RuntimeError(
            f"Failed to infer image size for {sample.filepath}; install pillow or compute metadata."
        ) from exc
    with Image.open(sample.filepath) as img:
        width, height = img.size
    return int(width), int(height)


def _normalize_bbox(rel_bbox: list[float], width: int, height: int) -> Optional[list[float]]:
    x, y, w, h = rel_bbox
    x_abs = float(x) * width
    y_abs = float(y) * height
    w_abs = float(w) * width
    h_abs = float(h) * height

    x1 = max(0.0, x_abs)
    y1 = max(0.0, y_abs)
    x2 = min(float(width), x_abs + w_abs)
    y2 = min(float(height), y_abs + h_abs)

    new_w = x2 - x1
    new_h = y2 - y1
    if new_w <= 0.0 or new_h <= 0.0:
        return None
    return [x1, y1, new_w, new_h]


def _collect_labels(view, label_field: str) -> List[str]:
    labels = set()
    for sample in view:
        dets = _get_label_field(sample, label_field)
        if dets is None:
            continue
        for det in dets.detections or []:
            if det.label is not None:
                labels.add(str(det.label))
    return sorted(labels)


def _copy_or_link(src: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _build_coco_from_view(
    view,
    label_field: str,
    categories: Dict[str, int],
    copy_images: bool,
    out_images_dir: Path,
    image_root: Optional[str],
    desc: str,
) -> Dict[str, Any]:
    images = []
    annotations = []
    ann_id = 1
    class_counts: Dict[str, int] = defaultdict(int)
    seen_image_ids = set()

    total = len(view)
    for sample in tqdm(view, total=total, desc=desc):
        if not os.path.exists(sample.filepath):
            raise FileNotFoundError(f"Image not found: {sample.filepath}")

        image_id = _stable_image_id(sample.id)
        if image_id in seen_image_ids:
            continue
        seen_image_ids.add(image_id)

        width, height = _ensure_image_size(sample)
        if copy_images:
            if image_root:
                rel = _as_posix_relpath(sample.filepath, image_root)
            else:
                rel = Path(sample.filepath).name
            dst = out_images_dir / rel
            _copy_or_link(sample.filepath, dst)
            file_name = Path(rel).as_posix()
        else:
            if not image_root:
                raise ValueError("image_root must be provided when copy_images is false")
            file_name = _as_posix_relpath(sample.filepath, image_root)

        images.append(
            {
                "id": image_id,
                "file_name": file_name,
                "width": width,
                "height": height,
            }
        )

        dets = _get_label_field(sample, label_field)
        if dets is None:
            continue
        for det in dets.detections or []:
            label = str(det.label)
            if label not in categories:
                continue
            rel_bbox = det.bounding_box
            if not rel_bbox:
                continue
            bbox = _normalize_bbox(rel_bbox, width, height)
            if bbox is None:
                continue
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": categories[label],
                    "bbox": bbox,
                    "area": float(bbox[2] * bbox[3]),
                    "iscrowd": 0,
                }
            )
            ann_id += 1
            class_counts[label] += 1

    return {
        "images": images,
        "annotations": annotations,
        "class_counts": dict(class_counts),
    }


def _get_label_field(sample, label_field: str):
    try:
        return sample.get_field(label_field)
    except Exception:
        return getattr(sample, label_field, None)


def _random_split(dataset, train_ratio: float, seed: int):
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be in (0, 1)")
    total = len(dataset)
    n_train = int(total * train_ratio)
    n_train = max(0, min(total, n_train))

    shuffled = dataset.shuffle(seed=seed)
    if hasattr(shuffled, "take") and hasattr(shuffled, "skip"):
        train_view = shuffled.take(n_train)
        val_view = shuffled.skip(n_train)
        return train_view, val_view

    raise RuntimeError("FiftyOne dataset does not support shuffle/take/skip for random split")


def export_fiftyone_to_coco(
    fo_dataset_name: str,
    label_field: str,
    classes: Optional[List[str]],
    train_view_spec: str,
    val_view_spec: str,
    out_dir: str,
    copy_images: bool,
    image_root: Optional[str],
    split_method: str = "views",
    train_ratio: float = 0.8,
    seed: int = 42,
) -> Dict[str, Any]:
    fo = _require_fiftyone()
    dataset = fo.load_dataset(fo_dataset_name)

    if split_method == "random":
        train_view, val_view = _random_split(dataset, train_ratio=train_ratio, seed=seed)
    elif split_method == "views":
        train_view = _parse_view_spec(dataset, train_view_spec)
        val_view = _parse_view_spec(dataset, val_view_spec)
    else:
        raise ValueError(f"Unsupported split_method: {split_method}")

    if classes is None:
        classes = sorted(set(_collect_labels(train_view, label_field)) | set(_collect_labels(val_view, label_field)))

    categories = {name: idx + 1 for idx, name in enumerate(classes)}
    categories_list = [{"id": idx + 1, "name": name} for idx, name in enumerate(classes)]

    out_path = Path(out_dir)
    coco_dir = out_path / "coco"
    images_dir = out_path / "images"
    coco_dir.mkdir(parents=True, exist_ok=True)
    if copy_images:
        images_dir.mkdir(parents=True, exist_ok=True)

    train_coco = _build_coco_from_view(
        view=train_view,
        label_field=label_field,
        categories=categories,
        copy_images=copy_images,
        out_images_dir=images_dir,
        image_root=image_root,
        desc="train",
    )
    val_coco = _build_coco_from_view(
        view=val_view,
        label_field=label_field,
        categories=categories,
        copy_images=copy_images,
        out_images_dir=images_dir,
        image_root=image_root,
        desc="val",
    )

    train_json = {
        "images": train_coco["images"],
        "annotations": train_coco["annotations"],
        "categories": categories_list,
    }
    val_json = {
        "images": val_coco["images"],
        "annotations": val_coco["annotations"],
        "categories": categories_list,
    }

    train_path = coco_dir / "train.json"
    val_path = coco_dir / "val.json"
    categories_path = coco_dir / "categories.json"

    train_path.write_text(json.dumps(train_json, ensure_ascii=True), encoding="utf-8")
    val_path.write_text(json.dumps(val_json, ensure_ascii=True), encoding="utf-8")
    categories_path.write_text(json.dumps(categories_list, ensure_ascii=True), encoding="utf-8")

    stats = {
        "train_images": len(train_json["images"]),
        "train_annotations": len(train_json["annotations"]),
        "val_images": len(val_json["images"]),
        "val_annotations": len(val_json["annotations"]),
        "train_class_counts": train_coco["class_counts"],
        "val_class_counts": val_coco["class_counts"],
    }

    print("Export complete")
    print(json.dumps(stats, indent=2, ensure_ascii=True))

    return {
        "train": str(train_path),
        "val": str(val_path),
        "categories": str(categories_path),
        "stats": stats,
    }


def _str2bool(value: str) -> bool:
    value = value.strip().lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export FiftyOne dataset to COCO JSON")
    parser.add_argument("--fo-dataset", required=True, help="FiftyOne dataset name")
    parser.add_argument("--label-field", default="detections", help="Detections field name")
    parser.add_argument("--train-view", default="", help="View spec for train")
    parser.add_argument("--val-view", default="", help="View spec for val")
    parser.add_argument("--out-dir", required=True, help="Output directory")
    parser.add_argument("--copy-images", type=_str2bool, default=False, help="Copy or link images")
    parser.add_argument("--image-root", default="", help="Image root for relative paths")
    parser.add_argument("--classes", default="", help="Comma-separated class names")
    parser.add_argument(
        "--split-method",
        choices=["views", "random"],
        default="views",
        help="Split method for train/val",
    )
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train split ratio for random")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    classes = None
    if args.classes:
        classes = [c.strip() for c in args.classes.split(",") if c.strip()]

    image_root = args.image_root or None

    export_fiftyone_to_coco(
        fo_dataset_name=args.fo_dataset,
        label_field=args.label_field,
        classes=classes,
        train_view_spec=args.train_view,
        val_view_spec=args.val_view,
        out_dir=args.out_dir,
        copy_images=bool(args.copy_images),
        image_root=image_root,
        split_method=args.split_method,
        train_ratio=float(args.train_ratio),
        seed=int(args.seed),
    )


if __name__ == "__main__":
    main()
