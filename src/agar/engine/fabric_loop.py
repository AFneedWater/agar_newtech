from __future__ import annotations
from typing import Dict
import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from tqdm import tqdm


def _move_batch(fabric, images, targets):
    images = [fabric.to_device(img) for img in images]
    moved_targets = []
    for t in targets:
        moved = {}
        for k, v in t.items():
            moved[k] = fabric.to_device(v) if torch.is_tensor(v) else v
        moved_targets.append(moved)
    return images, moved_targets


@torch.no_grad()
def evaluate(fabric, model, val_loader) -> Dict[str, float]:
    model.eval()
    metric = MeanAveragePrecision()
    for images, targets in tqdm(val_loader, desc="eval", disable=not fabric.is_global_zero):
        images, targets = _move_batch(fabric, images, targets)
        preds = model(images)  # list of dict
        # move to cpu for torchmetrics
        preds_cpu = [{k: v.detach().cpu() for k, v in p.items()} for p in preds]
        t_cpu = [
            {k: (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in t.items()}
            for t in targets
        ]
        metric.update(preds_cpu, t_cpu)
    out = metric.compute()
    # torchmetrics returns tensors
    return {k: float(v) for k, v in out.items() if torch.is_tensor(v)}


def train(
    fabric,
    cfg,
    model,
    optimizer,
    train_loader,
    val_loader,
    tb_logger=None,
    mlflow_logger=None,
    ckpt_cb=None,
):
    model.train()
    global_step = 0
    best_metric = float("-inf")

    for epoch in range(int(cfg.train.epochs)):
        model.train()
        pbar = tqdm(train_loader, desc=f"train e{epoch}", disable=not fabric.is_global_zero)
        for step, (images, targets) in enumerate(pbar):
            images, targets = _move_batch(fabric, images, targets)
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())

            fabric.backward(loss)

            if (step + 1) % int(cfg.train.grad_accum_steps) == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            global_step += 1
            if fabric.is_global_zero and (global_step % int(cfg.train.log_every) == 0):
                metrics = {"loss": float(loss.detach().cpu())}
                for k, v in loss_dict.items():
                    metrics[f"loss/{k}"] = float(v.detach().cpu())
                if tb_logger:
                    tb_logger.log_metrics(metrics, step=global_step, prefix="train/")
                if mlflow_logger:
                    mlflow_logger.log_metrics(metrics, step=global_step)
                pbar.set_postfix({k: f"{v:.4f}" for k, v in metrics.items() if k == "loss"})

        # eval
        if (epoch + 1) % int(cfg.train.eval_every) == 0:
            eval_metrics = evaluate(fabric, model, val_loader)
            if fabric.is_global_zero:
                if tb_logger:
                    tb_logger.log_metrics(eval_metrics, step=epoch + 1, prefix="val/")
                if mlflow_logger:
                    mlflow_logger.log_metrics(
                        {f"val/{k}": v for k, v in eval_metrics.items()}, step=epoch + 1
                    )

            # checkpoints
            if ckpt_cb is not None:
                from .callbacks import TrainState

                state = TrainState(epoch=epoch + 1, global_step=global_step, best_metric=best_metric)
                ckpt_cb.save_last(fabric, state, model, optimizer)
                ckpt_cb.maybe_save_best(
                    fabric,
                    state,
                    model,
                    optimizer,
                    metrics=eval_metrics,
                )
                best_metric = state.best_metric

    return {"global_step": global_step, "best_metric": best_metric}
