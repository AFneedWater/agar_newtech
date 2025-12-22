from __future__ import annotations
import os
from dataclasses import asdict

import hydra
from omegaconf import OmegaConf
from lightning_fabric import Fabric
import torch

from .utils.seed import set_seed
from .data.datamodule import build_dataloaders
from .models.factory import build_model
from .engine.fabric_loop import train as train_loop
from .engine.callbacks import CheckpointCallback
from .tracking.tb_logger import TBLogger
from .tracking.mlflow_logger import MLflowLogger
from .utils.rich_log import log


def _flatten_cfg(cfg) -> dict:
    return OmegaConf.to_container(cfg, resolve=True)


@hydra.main(config_path="../../conf", config_name="config", version_base="1.3")
def main(cfg):
    set_seed(int(cfg.seed))

    precision = str(cfg.train.precision)
    fabric = Fabric(precision=precision)
    fabric.launch()

    train_loader, val_loader = build_dataloaders(cfg)

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

    tb_logger = None
    if bool(cfg.train.tensorboard) and fabric.is_global_zero:
        tb_logger = TBLogger(log_dir=os.path.join(os.getcwd(), "tb"))

    mlflow_logger = None
    if bool(cfg.train.mlflow) and fabric.is_global_zero:
        mlflow_logger = MLflowLogger(
            tracking_uri=str(cfg.train.mlflow_tracking_uri),
            experiment=str(cfg.train.mlflow_experiment),
            run_name=str(cfg.train.mlflow_run_name),
        )
        mlflow_logger.log_params({"config": OmegaConf.to_yaml(cfg)})

    ckpt_cb = CheckpointCallback(out_dir=os.path.join(os.getcwd(), "checkpoints"), monitor=str(cfg.eval.metric_key))

    if fabric.is_global_zero:
        log(f"Output dir: {os.getcwd()}")
        log(OmegaConf.to_yaml(cfg))

    out = train_loop(
        fabric=fabric,
        cfg=cfg,
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        tb_logger=tb_logger,
        mlflow_logger=mlflow_logger,
        ckpt_cb=ckpt_cb,
    )

    if tb_logger:
        tb_logger.close()
    if mlflow_logger:
        mlflow_logger.close()

    if fabric.is_global_zero:
        log(f"Done. best_metric={out['best_metric']} global_step={out['global_step']}")


if __name__ == "__main__":
    main()
