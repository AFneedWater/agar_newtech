# agar

Minimal COCO-format detection training project (Hydra + Lightning Fabric).

## Install

```bash
pip install -e . -U
```

## Quick checks

```bash
python -m py_compile $(git ls-files '*.py')
```

## Train

Edit `conf/experiment/exp_agarv2_resnet50.yaml` to point to real data paths, then:

```bash
python -m agar.train experiment=exp_agarv2_resnet50
```

## Eval

```bash
python -m agar.eval ckpt=/path/to/checkpoints/checkpoint_best.pt
```
