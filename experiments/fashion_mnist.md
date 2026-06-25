# Stage 2 — Fashion-MNIST and Flow-GRPO

This stage moves from 2D points to low-resolution images. It covers pixel-space UNet training, classifier-free guidance, and two Flow-GRPO post-training rewards.

## What this reproduces

- Fashion-MNIST unconditional UNet baseline.
- Fashion-MNIST class-conditional CFG seed model.
- Flow-GRPO post-training for a fixed "pantsiness" reward.
- Flow-GRPO post-training for JPEG compressibility.

## Cost and runtime

All runs in this stage were local, so cloud cost is \$0.

| Run | Approx compute | Wall time |
|---|---:|---:|
| 20-epoch Fashion baseline / CFG | ~0.524 PFLOPs | ~8.1 min on MPS |
| 3-epoch CFG seed for GRPO | ~78.6 TFLOPs | ~1.3 min on MPS |
| GRPO JPEG moderate-long | ~0.700 PFLOPs | ~11.1 min on MPS |
| GRPO pantsiness | ~0.915 PFLOPs | ~16.7 min on MPS |

## Config files

Pretraining:

- `configs/experiment/fashion.yaml`
- `configs/experiment/fashion_cfg.yaml`
- `configs/dataset/fashion.yaml`
- `configs/model/unet_fashion.yaml`
- `configs/model/classcond_unet_fashion.yaml`

Flow-GRPO:

- `configs/config_grpo.yaml`
- `configs/experiment/fashion_grpo_trouser.yaml`
- `configs/experiment/fashion_grpo_jpeg.yaml`
- `configs/rl_training/default.yaml`
- `configs/rl_training/jpeg_compressibility.yaml`
- `configs/reward/fashion_trouser.yaml`
- `configs/reward/fashion_jpeg_compressibility.yaml`
- `configs/metrics/jpeg_compressibility.yaml`

## Effective config excerpts

Dataset:

```yaml
# configs/dataset/fashion.yaml
_target_: datasets.FashionMNISTDataset
name: fashion
root: ./data
```

Class-conditional model:

```yaml
# configs/model/classcond_unet_fashion.yaml
_target_: models.ClassCondUNet
in_ch: 1
base_ch: 32
depth: 2
time_dim: 64
use_attn: false
num_classes: 10
```

CFG training config:

```yaml
# configs/experiment/fashion_cfg.yaml
defaults:
  - override /dataset: fashion
  - override /model: classcond_unet_fashion

training:
  epochs: 20
  num_workers: 4
  log_every: 2
  p_uncond: 0.1

inference:
  n_samples: 64
  save_path: fashion_cfg_samples.png
  class_sampler:
    num_classes: 10
    guidance_scale: 3.0
  sampler:
    num_steps: 200
    latent_shape: [1, 28, 28]
```

Default GRPO settings:

```yaml
# configs/rl_training/default.yaml
epochs: 200
batch_size: 8
G: 8
num_inner: 4
lr: 1.0e-5
grad_clip: 1.0
clip_eps: 0.2
kl_beta: 0.04
advantage_scale: 1.0
sampler:
  T_rollout: 10
  sigma_a: 0.7
  t_min: 1.0e-3
  t_max: 0.999
  guidance_scale: 2.0
latent_shape: [1, 28, 28]
num_classes: 10
```

## Reproduce supervised training

Run from the repo root.

20-epoch unconditional baseline:

```bash
uv run python train.py \
  experiment=fashion \
  device=mps
```

20-epoch class-conditional CFG model:

```bash
uv run python train.py \
  experiment=fashion_cfg \
  device=mps
```

Short 3-epoch CFG seed for GRPO:

```bash
uv run python train.py \
  experiment=fashion_cfg \
  device=mps \
  training.epochs=3 \
  training.run_prefix=fashion_cfg_seed
```

Use the resulting checkpoint as `seed_checkpoint` for GRPO runs.

## Reproduce pantsiness GRPO

Reward: maximize the Fashion classifier's logit/probability for the Trouser class, regardless of the prompt.

Reward config:

```yaml
# configs/reward/fashion_trouser.yaml
_target_: rl.reward.FixedClassReward
classifier_checkpoint: runs/reward_models/fashion_classifier.pt
target_class: 1
device: ${device}
```

Representative run:

```bash
uv run python train_grpo.py \
  experiment=fashion_grpo_trouser \
  device=mps \
  seed_checkpoint=runs/<fashion_cfg_seed>/checkpoints/latest.pt \
  rl_training.epochs=50 \
  rl_training.batch_size=4 \
  rl_training.G=32 \
  rl_training.num_inner=12 \
  rl_training.lr=1.0e-4 \
  rl_training.clip_eps=1.0 \
  rl_training.kl_beta=0.001 \
  rl_training.advantage_scale=20.0
```

Expected behavior: the trouser reward increases, samples become more pants-like, and prompt/class alignment degrades but is not completely erased.

## Reproduce JPEG-compressibility GRPO

Reward: terminal black-box reward `reward(x) = -jpeg_bits_per_pixel(x)`.

Reward config:

```yaml
# configs/reward/fashion_jpeg_compressibility.yaml
_target_: rl.reward.JpegCompressibilityReward
quality: 75
optimize: false
progressive: false
subsampling: null
```

Representative moderate-long run:

```bash
uv run python train_grpo.py \
  experiment=fashion_grpo_jpeg \
  device=mps \
  seed_checkpoint=runs/<fashion_cfg_seed>/checkpoints/latest.pt \
  rl_training.epochs=300 \
  rl_training.batch_size=4 \
  rl_training.G=8 \
  rl_training.num_inner=6 \
  rl_training.lr=2.0e-5 \
  rl_training.clip_eps=0.5 \
  rl_training.kl_beta=0.01 \
  rl_training.advantage_scale=5.0 \
  rl_training.save_every=100 \
  rl_training.run_prefix=fashion_grpo_jpeg_moderate_long
```

Evaluation with JPEG/PNG bpp metrics:

```bash
uv run python inference.py \
  experiment=fashion_cfg \
  metrics=jpeg_compressibility \
  inference.sampler.checkpoint=runs/<jpeg_grpo_run>/checkpoints/latest.pt \
  inference.n_samples=64 \
  inference.save_path=runs/<jpeg_grpo_run>/eval_compare/inference_grid.png \
  device=mps
```

Recorded fixed-grid metrics for the moderate-long run:

| Model | JPEG bpp ↓ | PNG bpp ↓ | Classifier acc ↑ |
|---|---:|---:|---:|
| Seed | 6.2736 | 5.3020 | 0.9531 |
| GRPO JPEG moderate-long | 4.9990 | 2.6207 | 0.6094 |

Detailed JPEG experiment notes and sample grids are in `experiments/jpeg_compression_grpo/experiment.md`.

## Code pointers

- `train.py` — supervised flow matching.
- `train_grpo.py` — GRPO training entrypoint.
- `rl/sde_sampler.py` — SDE rollouts and Gaussian transition log-probs.
- `rl/grpo.py` — group advantages, clipped surrogate, KL.
- `rl/reward.py` — reward implementations.
- `rl/compression.py` — JPEG/PNG bpp helpers.
- `metrics.py` — JPEG-compressibility metrics.
