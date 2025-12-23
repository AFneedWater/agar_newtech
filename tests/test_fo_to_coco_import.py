import importlib

import pytest


def _has_fiftyone() -> bool:
    try:
        import fiftyone  # noqa: F401
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _has_fiftyone(), reason="fiftyone not installed")
def test_module_import_and_argparse():
    mod = importlib.import_module("agar.data.fo_to_coco")
    parser = mod.build_arg_parser()
    args = parser.parse_args(
        [
            "--fo-dataset",
            "agar_lower",
            "--label-field",
            "detections",
            "--train-view",
            "tag:train",
            "--val-view",
            "tag:val",
            "--out-dir",
            "/tmp/agar_coco",
            "--copy-images",
            "false",
            "--image-root",
            "/data/images",
            "--classes",
            "A,B",
        ]
    )
    assert args.fo_dataset == "agar_lower"
    assert args.copy_images is False
