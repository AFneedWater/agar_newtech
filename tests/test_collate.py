from agar.data.datamodule import detection_collate
import torch


def test_detection_collate_shapes():
    img1 = torch.zeros((3, 10, 10))
    img2 = torch.zeros((3, 12, 12))
    t1 = {"boxes": torch.zeros((0, 4)), "labels": torch.zeros((0,), dtype=torch.int64)}
    t2 = {"boxes": torch.zeros((1, 4)), "labels": torch.ones((1,), dtype=torch.int64)}
    images, targets = detection_collate([(img1, t1), (img2, t2)])
    assert len(images) == 2
    assert len(targets) == 2
