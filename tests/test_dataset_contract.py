import os
import pytest
import torch
from agar.data.agar_dataset import COCODetectionDataset


@pytest.mark.skipif(
    not os.getenv("AGAR_IMAGE_ROOT") or not os.getenv("AGAR_ANN_FILE"),
    reason="Set AGAR_IMAGE_ROOT and AGAR_ANN_FILE to run this test.",
)
def test_dataset_contract():
    ds = COCODetectionDataset(os.environ["AGAR_IMAGE_ROOT"], os.environ["AGAR_ANN_FILE"])
    img, target = ds[0]
    assert isinstance(img, torch.Tensor) and img.ndim == 3
    assert "boxes" in target and "labels" in target
    assert target["boxes"].shape[1] == 4
    assert target["labels"].dtype == torch.int64
