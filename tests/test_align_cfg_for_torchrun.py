import os

from omegaconf import OmegaConf

from agar.utils.distributed import align_cfg_for_torchrun


def test_align_cfg_for_torchrun(monkeypatch):
    cfg = OmegaConf.create({"train": {"devices": 1, "strategy": "auto", "accelerator": "auto"}})
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "2")
    out = align_cfg_for_torchrun(cfg)
    assert int(out.train.devices) == 2
    assert str(out.train.strategy) == "ddp"
    assert str(out.train.accelerator) == "cuda"


def test_align_cfg_for_torchrun_noop(monkeypatch):
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    cfg = OmegaConf.create({"train": {"devices": 1, "strategy": "auto", "accelerator": "cpu"}})
    out = align_cfg_for_torchrun(cfg)
    assert int(out.train.devices) == 1
    assert str(out.train.strategy) == "auto"
    assert str(out.train.accelerator) == "cpu"

