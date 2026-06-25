# Stage 3 — CIFAR-10

This stage scales the Fashion-MNIST UNet family to RGB images and a larger semantic distribution, while still training locally on a consumer GPU.

## What this reproduces

- CIFAR-10 unconditional UNet baseline.
- CIFAR-10 class-conditional CFG model used in the writeup.
- Optional CIFAR-10 FID metric wiring.

## Cost and runtime

| Item | Value |
|---|---:|
| Dataset | 60,000 total / 50,000 train / 10,000 test |
| Image shape | 32x32 RGB |
| Model size | 22,311,299 parameters for `ClassCondUNet` |
| Representative run | 500 epochs, batch 256 |
| Optimizer steps | ~97.5k |
| Training samples | ~24.96M |
| Approx compute | ~320 PFLOPs |
| Wall time | ~4.1–4.3h on RTX 4080 |
| Peak GPU memory | ~4.9GB |
| Cloud cost | local only / \$0 |

## Config files

- `configs/experiment/cifar10.yaml`
- `configs/experiment/cifar10_cfg.yaml`
- `configs/dataset/cifar10.yaml`
- `configs/model/unet_cifar.yaml`
- `configs/model/classcond_unet_cifar.yaml`
- `configs/metrics/fid_cifar10.yaml`

## Effective config excerpts

Dataset:

```yaml
# configs/dataset/cifar10.yaml
_target_: datasets.CifarDataset
name: cifar10
root: ./data
```

Class-conditional model:

```yaml
# configs/model/classcond_unet_cifar.yaml
_target_: models.ClassCondUNet
in_ch: 3
base_ch: 128
depth: 3
time_dim: 256
use_attn: true
num_classes: 10
```

CFG experiment:

```yaml
# configs/experiment/cifar10_cfg.yaml
defaults:
  - override /dataset: cifar10
  - override /model: classcond_unet_cifar

training:
  num_workers: 4
  batch_size: 256
  epochs: 500
  log_every: 20
  p_uncond: 0.1

inference:
  n_samples: 64
  save_path: cifar10_cfg_samples.png
  class_sampler:
    num_classes: 10
    guidance_scale: 3.0
  sampler:
    num_steps: 200
    latent_shape: [3, 32, 32]
```

FID metric config:

```yaml
# configs/metrics/fid_cifar10.yaml
inference:
  metrics:
    - _target_: metrics.FIDMetric
      dataset_name: cifar10
      dataset_res: 32
      dataset_split: train
      mode: clean
      device: cpu
```

## Reproduce training

Run from the repo root.

Unconditional baseline:

```bash
uv run python train.py \
  experiment=cifar10 \
  device=cuda
```

Class-conditional CFG model:

```bash
uv run python train.py \
  experiment=cifar10_cfg \
  device=cuda \
  training.epochs=500 \
  training.batch_size=256
```

The explicit `training.epochs` and `training.batch_size` overrides match the representative writeup run, even though they are also present in `configs/experiment/cifar10_cfg.yaml`.

## Reproduce sampling

```bash
uv run python inference.py \
  experiment=cifar10_cfg \
  inference.sampler.checkpoint=runs/<cifar10_cfg_run>/checkpoints/latest.pt \
  inference.save_path=cifar10_cfg_samples.png \
  device=cuda
```

Optional metric-enabled inference:

```bash
uv run python inference.py \
  experiment=cifar10_cfg \
  metrics=fid_cifar10 \
  inference.sampler.checkpoint=runs/<cifar10_cfg_run>/checkpoints/latest.pt \
  inference.n_samples=50000 \
  inference.save_path=runs/<cifar10_cfg_run>/fid_samples.png \
  device=cuda
```

## Code pointers

- `models.py` — `UNet` and `ClassCondUNet`.
- `datasets.py` — CIFAR-10 data wrapper.
- `train.py` — training loop and CFG null-label dropout.
- `inference.py` — CFG sampler.
- `metrics.py` — Clean-FID integration.
