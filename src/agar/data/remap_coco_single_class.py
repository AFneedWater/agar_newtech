from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Set


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: str, payload: Dict[str, Any]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2)


def _as_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def remap_coco_single_class(
    in_path: str,
    out_path: str,
    drop_empty: bool = False,
) -> None:
    coco = _load_json(in_path)
    annotations: List[Dict[str, Any]] = coco.get("annotations", [])
    images: List[Dict[str, Any]] = coco.get("images", [])

    for ann in annotations:
        ann["category_id"] = 1

    coco["categories"] = [{"id": 1, "name": "single_colony"}]

    if drop_empty:
        annotated_ids: Set[int] = set()
        for ann in annotations:
            image_id = _as_int(ann.get("image_id"))
            if image_id is not None:
                annotated_ids.add(image_id)
        images = [img for img in images if _as_int(img.get("id")) in annotated_ids]
        coco["images"] = images

    _write_json(out_path, coco)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Remap COCO JSON to a single class")
    parser.add_argument("--in", dest="in_path", required=True, help="Path to COCO JSON")
    parser.add_argument("--out", dest="out_path", required=True, help="Output COCO JSON path")
    parser.add_argument(
        "--drop-empty",
        action="store_true",
        help="Drop images without annotations (default keeps them)",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    if not os.path.exists(args.in_path):
        raise FileNotFoundError(args.in_path)

    remap_coco_single_class(args.in_path, args.out_path, drop_empty=bool(args.drop_empty))


if __name__ == "__main__":
    main()
