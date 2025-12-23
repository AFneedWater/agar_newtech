from __future__ import annotations
from typing import Any, Dict
import time
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
def evaluate(fabric, model, val_loader, return_per_class: bool = False) -> Dict[str, Any]:
    model.eval()
    try:
        metric = MeanAveragePrecision()
    except ModuleNotFoundError as exc:
        if fabric.is_global_zero:
            print(
                "MeanAveragePrecision unavailable (missing dependency). "
                "Install pycocotools or faster-coco-eval, or set train.eval_every=0 to skip eval."
            )
            print(f"Error: {exc}")
        return {}
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
    if return_per_class:
        metrics = {}
        for k, v in out.items():
            if not torch.is_tensor(v):
                continue
            v = v.detach().cpu()
            if v.numel() == 1:
                metrics[k] = float(v)
            else:
                metrics[k] = v.tolist()
        return metrics

    # torchmetrics returns tensors
    return {k: float(v) for k, v in out.items() if torch.is_tensor(v) and v.numel() == 1}


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
    max_steps = int(getattr(cfg.train, "max_steps", -1))
    profile_first = bool(getattr(cfg.train, "profile_first_batch", False))
    eval_every = int(cfg.train.eval_every)
    log_every = int(cfg.train.log_every)
    grad_accum = int(cfg.train.grad_accum_steps)
    profiled = False
    train_loader, val_loader = fabric.setup_dataloaders(train_loader, val_loader)

    def _log_metrics(loss_dict):
        if not (fabric.is_global_zero and log_every > 0 and (global_step % log_every == 0)):
            return
        metrics = {"loss": float(sum(loss_dict.values()).detach().cpu())}
        for k, v in loss_dict.items():
            metrics[f"loss/{k}"] = float(v.detach().cpu())
        if tb_logger:
            tb_logger.log_metrics(metrics, step=global_step, prefix="train/")
        if mlflow_logger:
            mlflow_logger.log_metrics(metrics, step=global_step)

    def _run_step(step_idx, images, targets):
        nonlocal global_step
        images, targets = _move_batch(fabric, images, targets)
        loss_dict = model(images, targets)
        loss = sum(loss_dict.values())

        fabric.backward(loss)

        if (step_idx + 1) % grad_accum == 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        global_step += 1
        _log_metrics(loss_dict)
        return loss_dict

    stop_training = False
    for epoch in range(int(cfg.train.epochs)):
        model.train()
        if max_steps > 0 and global_step >= max_steps:
            stop_training = True
            break

        it = iter(train_loader)
        total = len(train_loader)
        first_step = 0

        if profile_first and not profiled:
            if fabric.is_global_zero:
                print("starting first batch...")
            t0 = time.perf_counter()
            try:
                images, targets = next(it)
            except StopIteration:
                break
            data_time = time.perf_counter() - t0
            t1 = time.perf_counter()
            images, targets = _move_batch(fabric, images, targets)
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())
            fwd_time = time.perf_counter() - t1

            t2 = time.perf_counter()
            fabric.backward(loss)
            bwd_time = time.perf_counter() - t2

            t3 = time.perf_counter()
            if (first_step + 1) % grad_accum == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            opt_time = time.perf_counter() - t3

            global_step += 1
            _log_metrics(loss_dict)

            if fabric.is_global_zero:
                print(
                    f"first_batch data_time={data_time:.4f}s "
                    f"fwd_time={fwd_time:.4f}s bwd_time={bwd_time:.4f}s opt_time={opt_time:.4f}s"
                )
            profiled = True
            first_step = 1

            if max_steps > 0 and global_step >= max_steps:
                stop_training = True
                break

        pbar = tqdm(
            it,
            desc=f"train e{epoch}",
            disable=not fabric.is_global_zero,
            total=total,
            initial=first_step,
        )
        for step, (images, targets) in enumerate(pbar, start=first_step):
            loss_dict = _run_step(step, images, targets)
            if fabric.is_global_zero and log_every > 0 and (global_step % log_every == 0):
                loss_val = float(sum(loss_dict.values()).detach().cpu())
                pbar.set_postfix({"loss": f"{loss_val:.4f}"})

            if max_steps > 0 and global_step >= max_steps:
                stop_training = True
                break

        if stop_training:
            break

        # eval
        if eval_every > 0 and (epoch + 1) % eval_every == 0:
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
