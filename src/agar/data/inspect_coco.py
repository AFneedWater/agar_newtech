from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _get_image_size(image_info: Dict[str, Any], images_dir: str) -> Tuple[int, int]:
    width = image_info.get("width")
    height = image_info.get("height")
    if width is not None and height is not None:
        return int(width), int(height)

    try:
        from PIL import Image
    except Exception as exc:
        raise RuntimeError("PIL is required to read image sizes when width/height missing") from exc

    file_name = image_info.get("file_name", "")
    path = Path(images_dir) / file_name
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    with Image.open(path) as img:
        w, h = img.size
    return int(w), int(h)


def _bbox_within(bbox: list[float], width: int, height: int) -> bool:
    x, y, bw, bh = [float(v) for v in bbox]
    return x >= 0.0 and y >= 0.0 and (x + bw) <= width and (y + bh) <= height

def _build_category_mapping(
    categories: List[Dict[str, Any]],
    annotations: List[Dict[str, Any]],
    classes: List[str] | None,
) -> Dict[int, str]:
    if categories:
        has_names = any("name" in c for c in categories)
        if has_names:
            return {
                int(c["id"]): str(c.get("name", c.get("id", "")))
                for c in categories
                if "id" in c
            }
        if classes:
            return {i + 1: name for i, name in enumerate(classes)}
        return {int(c["id"]): str(c.get("id", "")) for c in categories if "id" in c}

    if classes:
        return {i + 1: name for i, name in enumerate(classes)}

    ids = sorted({int(ann.get("category_id", -1)) for ann in annotations if "category_id" in ann})
    return {cid: str(cid) for cid in ids}


def inspect_coco(
    json_path: str,
    images_dir: str,
    check_all: bool = False,
    classes: List[str] | None = None,
) -> None:
    coco = _load_json(json_path)
    images = coco.get("images", [])
    annotations = coco.get("annotations", [])
    categories = coco.get("categories", [])
    cat_mapping = _build_category_mapping(categories, annotations, classes)

    print(f"images: {len(images)}")
    print(f"annotations: {len(annotations)}")
    print(f"categories: {len(categories)}")

    if images:
        print("first image:")
        print(json.dumps(images[0], ensure_ascii=True, indent=2))
    if annotations:
        print("first annotation:")
        print(json.dumps(annotations[0], ensure_ascii=True, indent=2))

    image_by_id = {int(img["id"]): img for img in images if "id" in img}

    if annotations:
        sample_n = min(10, len(annotations))
        sampled = random.sample(annotations, sample_n)
        print(f"sampled {sample_n} bboxes:")
        out_of_bounds = 0
        for ann in sampled:
            image_id = int(ann.get("image_id", -1))
            img_info = image_by_id.get(image_id)
            if img_info is None:
                print(f"  ann_id={ann.get('id')} image_id={image_id} missing image info")
                continue
            w, h = _get_image_size(img_info, images_dir)
            bbox = ann.get("bbox", [0, 0, 0, 0])
            within = _bbox_within(bbox, w, h)
            status = "ok" if within else "out_of_bounds"
            if not within:
                out_of_bounds += 1
                status += " WARNING"
            x, y, bw, bh = [float(v) for v in bbox]
            print(
                f"  ann_id={ann.get('id')} image_id={image_id} bbox=({x:.1f},{y:.1f},{bw:.1f},{bh:.1f})"
                f" image=({w},{h}) {status}"
            )
        if sample_n > 0:
            ratio = out_of_bounds / sample_n
            print(f"out_of_bounds_ratio: {out_of_bounds}/{sample_n} ({ratio:.2%})")
            print("note: Dataset clips bboxes during training; out_of_bounds won't crash.")
    else:
        print("no annotations to sample")

    if annotations:
        counts = Counter(int(ann.get("category_id", -1)) for ann in annotations)
        top = counts.most_common(10)
        print("top category_id counts:")
        for cid, count in top:
            name = cat_mapping.get(cid, str(cid))
            print(f"  category_id={cid} name={name}: {count}")

    if check_all and annotations:
        sizes = {}
        missing_sizes = 0
        for img in images:
            img_id = img.get("id")
            w = img.get("width")
            h = img.get("height")
            if img_id is None or w is None or h is None:
                missing_sizes += 1
                continue
            sizes[int(img_id)] = (int(w), int(h))

        total = 0
        out_of_bounds = 0
        skipped = 0
        for ann in annotations:
            image_id = int(ann.get("image_id", -1))
            size = sizes.get(image_id)
            if size is None:
                skipped += 1
                continue
            w, h = size
            bbox = ann.get("bbox", [0, 0, 0, 0])
            if not _bbox_within(bbox, w, h):
                out_of_bounds += 1
            total += 1

        ratio = (out_of_bounds / total) if total else 0.0
        print("check_all (no image reads):")
        print(f"  checked: {total}")
        print(f"  out_of_bounds: {out_of_bounds} ({ratio:.2%})")
        print(f"  skipped_missing_sizes: {skipped}")
        if missing_sizes:
            print(f"  images_missing_sizes: {missing_sizes}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect COCO JSON cache")
    parser.add_argument("--json", required=True, help="Path to COCO JSON")
    parser.add_argument("--images-dir", required=True, help="Images root directory")
    parser.add_argument("--check-all", action="store_true", help="Check all annotations without image reads")
    parser.add_argument("--classes", default="", help="Comma-separated class names")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    if not os.path.exists(args.json):
        raise FileNotFoundError(args.json)
    if not os.path.isdir(args.images_dir):
        raise NotADirectoryError(args.images_dir)

    classes = None
    if args.classes:
        classes = [c.strip() for c in args.classes.split(",") if c.strip()]

    inspect_coco(args.json, args.images_dir, check_all=bool(args.check_all), classes=classes)


if __name__ == "__main__":
    main()
