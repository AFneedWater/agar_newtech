import os
import pytest
import torch
from agar.data.agar_dataset import COCODetectionDataset


@pytest.mark.skipif(
    not os.getenv("AGAR_IMAGES_DIR") or not os.getenv("AGAR_TRAIN_JSON"),
    reason="Set AGAR_IMAGES_DIR and AGAR_TRAIN_JSON to run this test.",
)
def test_dataset_contract():
    ds = COCODetectionDataset(os.environ["AGAR_IMAGES_DIR"], os.environ["AGAR_TRAIN_JSON"])
    img, target = ds[0]
    assert isinstance(img, torch.Tensor) and img.ndim == 3
    assert "boxes" in target and "labels" in target
    assert target["boxes"].shape[1] == 4
    assert target["labels"].dtype == torch.int64
