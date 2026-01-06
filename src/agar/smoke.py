from __future__ import annotations

import os
from typing import Dict, Any

import hydra
from lightning_fabric import Fabric
import torch
from torch.utils.data import DataLoader, Dataset

from .utils.tf32 import configure_tf32

configure_tf32(enable=True)

from .data.datamodule import build_dataloaders, detection_collate
from .models.factory import build_model
from .utils.distributed import align_cfg_for_torchrun, torchrun_env
from .utils.launch import is_torchrun_env, log_launch_info, log_prelaunch_info, resolve_accelerator
from .utils.rich_log import log


class DummyDetectionDataset(Dataset):
    def __init__(self, num_samples: int, image_size: int = 128):
        self.num_samples = int(num_samples)
        self.image_size = int(image_size)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int):
        image = torch.rand(3, self.image_size, self.image_size)
        target = {
            "boxes": torch.tensor([[0.0, 0.0, 1.0, 1.0]], dtype=torch.float32),
            "labels": torch.tensor([1], dtype=torch.int64),
            "image_id": torch.tensor([idx], dtype=torch.int64),
            "area": torch.tensor([1.0], dtype=torch.float32),
            "iscrowd": torch.tensor([0], dtype=torch.int64),
        }
        return image, target


def _has_real_data(cfg) -> bool:
    train_json = str(getattr(cfg.data, "train_json", ""))
    val_json = str(getattr(cfg.data, "val_json", ""))
    images_dir = str(getattr(cfg.data, "images_dir", ""))
    if not (train_json and val_json and images_dir):
        return False
    return bool(
        os.path.exists(train_json)
        and os.path.exists(val_json)
        and os.path.isdir(images_dir)
    )


def _build_dummy_loaders(cfg) -> Dict[str, DataLoader]:
    batch_size = int(getattr(cfg.data, "batch_size", 1))
    dataset = DummyDetectionDataset(num_samples=batch_size, image_size=128)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=detection_collate,
    )
    return {"train": loader, "val": loader}


@hydra.main(config_path="../../conf", config_name="config", version_base="1.3")
def main(cfg) -> None:
    cfg = align_cfg_for_torchrun(cfg)
    devices_cfg = int(getattr(cfg.train, "devices", 1))
    accelerator_cfg = str(getattr(cfg.train, "accelerator", "cuda"))
    strategy_cfg = str(getattr(cfg.train, "strategy", "auto"))
    strict_launch = bool(getattr(cfg.train, "strict_launch", False))

    num_nodes = 1
    local_rank = None
    if is_torchrun_env():
        if not torch.cuda.is_available():
            raise RuntimeError("torchrun detected but CUDA is not available")
        env = torchrun_env()
        local_rank = int(env["local_rank"])
        local_world_size = int(env["local_world_size"])
        world_size = int(env["world_size"])
        num_nodes = max(1, world_size // local_world_size)
        log_prelaunch_info(local_rank)
        torch.cuda.set_device(local_rank)
        devices = local_world_size
        accelerator = "cuda"
        strategy = str(getattr(cfg.train, "strategy", "ddp"))
    else:
        devices = devices_cfg
        if devices > 1:
            raise RuntimeError("train.devices>1 requires torchrun; do not rely on Fabric spawn")
        accelerator = resolve_accelerator(accelerator_cfg)
        strategy = strategy_cfg
        log_prelaunch_info(None)

    if strict_launch and accelerator == "cuda":
        visible = torch.cuda.device_count()
        if devices > visible:
            raise ValueError(
                f"train.devices={devices} exceeds visible CUDA devices={visible}. "
                "Adjust train.devices or set CUDA_VISIBLE_DEVICES."
            )

    fabric = Fabric(
        accelerator=accelerator,
        devices=devices,
        strategy=strategy,
        num_nodes=num_nodes,
        precision=str(cfg.train.precision),
    )
    fabric.launch()

    if strict_launch and (not is_torchrun_env()) and devices == 1 and fabric.world_size != 1:
        raise RuntimeError(
            f"strict_launch: expected world_size=1 but got {fabric.world_size}. "
            "DDP likely mis-launched."
        )

    log_launch_info(fabric)

    use_real = _has_real_data(cfg)
    if use_real:
        train_loader, val_loader = build_dataloaders(cfg)
        log("Using real COCO data for smoke run")
    else:
        loaders = _build_dummy_loaders(cfg)
        train_loader, val_loader = loaders["train"], loaders["val"]
        log("Using dummy data for smoke run (missing COCO paths)")

    model = build_model(cfg)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=float(cfg.train.lr),
        momentum=float(cfg.train.momentum),
        weight_decay=float(cfg.train.weight_decay),
    )

    model, optimizer = fabric.setup(model, optimizer)
    train_loader, val_loader = fabric.setup_dataloaders(train_loader, val_loader)

    model.train()
    did_step = False

    images, targets = next(iter(train_loader))
    batch_size = len(images)
    cfg_batch_size = int(getattr(cfg.data, "batch_size", batch_size))
    loss_dict = model(images, targets)
    loss = sum(loss_dict.values())

    fabric.backward(loss)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    did_step = True

    if fabric.is_global_zero:
        log(f"cfg.data.batch_size={cfg_batch_size}, actual_batch={batch_size}")
        log(f"loss={float(loss.detach().cpu()):.6f}")
        log(f"optimizer_step={did_step}")


if __name__ == "__main__":
    main()
