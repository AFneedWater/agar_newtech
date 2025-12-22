from __future__ import annotations
import hydra
from omegaconf import OmegaConf
from lightning_fabric import Fabric

from .data.datamodule import build_dataloaders
from .models.factory import build_model
from .engine.fabric_loop import evaluate
from .utils.io import load_checkpoint
from .utils.rich_log import log


@hydra.main(config_path="../../conf", config_name="config", version_base="1.3")
def main(cfg):
    fabric = Fabric(precision="32-true")
    fabric.launch()

    _, val_loader = build_dataloaders(cfg)
    model = build_model(cfg)

    ckpt_path = getattr(cfg, "ckpt", "")
    if ckpt_path:
        state = load_checkpoint(ckpt_path, map_location="cpu")
        model.load_state_dict(state["model"], strict=True)

    model = fabric.setup(model)
    val_loader = fabric.setup_dataloaders(val_loader)

    metrics = evaluate(fabric, model, val_loader)
    if fabric.is_global_zero:
        log("Eval metrics:")
        for k, v in metrics.items():
            log(f"{k}: {v:.6f}")


if __name__ == "__main__":
    main()
