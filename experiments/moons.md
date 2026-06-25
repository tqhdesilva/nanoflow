# Stage 1 — Moons

This stage is the cheapest end-to-end sanity check for the NanoFlow training stack: CondOT flow matching, class conditioning, CFG sampling, logging, and image/sample export.

## What this reproduces

- Synthetic two-moons data with 2 classes.
- Unconditional MLP baseline.
- Class-conditional MLP with classifier-free guidance.
- Representative 200-epoch CFG run used in the writeup.

## Cost and runtime

| Item | Value |
|---|---:|
| Dataset size | 8,000 generated points |
| Train cost | local only / \$0 cloud cost |
| Representative wall time | ~37s on MPS |
| Representative compute estimate | ~539 GFLOPs |
| Representative training samples | ~1.28M |
| Representative optimizer steps | ~10k |

## Config files

- `configs/experiment/moons.yaml`
- `configs/experiment/moons_cfg.yaml`
- `configs/dataset/moons.yaml`
- `configs/model/mlp.yaml`
- `configs/model/classcond_mlp.yaml`
- `configs/training/default.yaml`
- `configs/flow/condot.yaml`
- `configs/solver/euler.yaml`

## Effective config excerpts

Dataset:

```yaml
# configs/dataset/moons.yaml
_target_: datasets.MoonsDataset
name: moons
n: 8000
noise: 0.05
```

Unconditional model:

```yaml
# configs/model/mlp.yaml
_target_: models.MLP
hidden_dim: 128
num_layers: 4
time_dim: 32
```

Class-conditional model:

```yaml
# configs/model/classcond_mlp.yaml
_target_: models.ClassCondMLP
hidden_dim: 128
num_layers: 4
time_dim: 32
num_classes: 2
```

CFG experiment:

```yaml
# configs/experiment/moons_cfg.yaml
defaults:
  - override /dataset: moons
  - override /model: classcond_mlp

training:
  p_uncond: 0.1

inference:
  n_samples: 1000
  save_path: moons_cfg_samples.png
  class_sampler:
    num_classes: 2
    guidance_scale: 2.0
    probs: null
  sampler:
    num_steps: 100
```

## Reproduce training

Run from the repo root.

Unconditional baseline:

```bash
uv run python train.py \
  experiment=moons \
  device=mps \
  training.epochs=200
```

Class-conditional CFG model:

```bash
uv run python train.py \
  experiment=moons_cfg \
  device=mps \
  training.epochs=200
```

Use `device=cpu` if MPS is unavailable.

## Reproduce sampling

The training run emits samples through the configured `sample_logger`. To sample from a saved checkpoint explicitly:

```bash
uv run python inference.py \
  experiment=moons_cfg \
  inference.sampler.checkpoint=runs/<run_id>/checkpoints/latest.pt \
  inference.save_path=moons_cfg_samples.png \
  device=mps
```

Expected output: a 2D scatter/grid visualization comparable to `moons_cfg_samples.png`.

## Code pointers

- `flow.py` — CondOT path and target velocity.
- `train.py` — training loop.
- `inference.py` — ODE/CFG sampling.
- `models.py` — `MLP` and `ClassCondMLP`.
- `datasets.py` — synthetic moons dataset.
