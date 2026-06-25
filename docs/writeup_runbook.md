# NanoFlow write-up runbook

This document collects the main commands and code pointers for the NanoFlow project write-up. The project is intentionally minimal PyTorch/Hydra flow matching code; it avoids pre-built diffusion training frameworks, except for using pretrained VAEs for latent ImageNet experiments.

## Core training recipe

The default probability path is Conditional Optimal Transport (CondOT):

```text
x_t = (1 - t) * eps + t * x_0
v_t = x_0 - eps
```

Key files:

- `flow.py` — CondOT interpolation and target velocity.
- `train.py` — training loop, MSE / masked-MSE loss, checkpointing, post-train inference.
- `inference.py` — ODE sampling and classifier-free guidance sampling.
- `models.py` — MLP and U-Net backbones for moons, Fashion-MNIST, and CIFAR-10.
- `models_dit.py` — DiT, deferred token masking, patch mixer, and packed MoE modules for ImageNet latent runs.
- `configs/` — Hydra experiment, dataset, model, reward, solver, and training configs.

## Moons

Fast 2D sanity check.

```bash
uv run python train.py device=mps
```

Useful variants:

```bash
uv run python train.py experiment=moons_cfg device=mps
uv run python train.py experiment=moons device=mps training.epochs=200 inference.save_path=moons_samples.png
```

Pointers:

- `configs/experiment/moons.yaml`
- `configs/experiment/moons_cfg.yaml`
- `configs/dataset/moons.yaml`
- `configs/model/mlp.yaml`
- `datasets.py` (`MoonsDataset`)
- `models.py` (`MLP`, `ClassCondMLP`)

## Fashion-MNIST

Small image dataset; first U-Net and classifier-free guidance experiments.

```bash
uv run python train.py experiment=fashion device=mps
uv run python train.py experiment=fashion_cfg device=mps
```

Pointers:

- `configs/experiment/fashion.yaml`
- `configs/experiment/fashion_cfg.yaml`
- `configs/dataset/fashion.yaml`
- `configs/model/unet_fashion.yaml`
- `configs/model/classcond_unet_fashion.yaml`
- `datasets.py` (`FashionMNISTDataset`)
- `models.py` (`UNet`, `ClassCondUNet`)

## CIFAR-10

RGB scaling stage.

```bash
uv run python train.py experiment=cifar10 device=cuda
uv run python train.py experiment=cifar10_cfg device=cuda
```

Example override:

```bash
uv run python train.py experiment=cifar10_cfg device=cuda training.epochs=500 training.batch_size=256
```

Multi-GPU:

```bash
torchrun --nproc_per_node=N train.py experiment=cifar10_cfg device=cuda distributed=ddp
```

FID smoke/eval support exists through the metrics config:

```bash
uv run python inference.py \
  experiment=cifar10_cfg \
  metrics=fid_cifar10 \
  inference.sampler.checkpoint=runs/{cifar_run}/checkpoints/latest.pt \
  inference.n_samples=64 \
  inference.save_path=runs/{cifar_run}/fid_eval_samples.png \
  device=cuda
```

Pointers:

- `configs/experiment/cifar10.yaml`
- `configs/experiment/cifar10_cfg.yaml`
- `configs/dataset/cifar10.yaml`
- `configs/model/unet_cifar.yaml`
- `configs/model/classcond_unet_cifar.yaml`
- `configs/metrics/fid_cifar10.yaml`
- `datasets.py` (`CifarDataset`)
- `models.py` (`UNet`, `ClassCondUNet`)
- `metrics.py` (`CleanFIDMetric`)

## Fashion-MNIST Flow-GRPO

Train a reward classifier, then fine-tune a class-conditional Fashion-MNIST checkpoint with Flow-GRPO.

```bash
uv run python -m rl.classifier --epochs 5
```

Prompt-aligned reward:

```bash
uv run python train_grpo.py \
  experiment=fashion_grpo \
  seed_checkpoint=runs/{fashion_cfg_run}/checkpoints/latest.pt \
  device=mps
```

Fixed Trouser diagnostic reward:

```bash
uv run python train_grpo.py experiment=fashion_grpo_trouser device=mps
```

JPEG-compressibility reward:

```bash
uv run python train_grpo.py \
  experiment=fashion_grpo_jpeg \
  device=mps \
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

JPEG metric comparison:

```bash
uv run python inference.py \
  experiment=fashion_cfg \
  metrics=jpeg_compressibility \
  inference.sampler.checkpoint=runs/fashion_grpo_jpeg_moderate_long_20260516_213332/checkpoints/latest.pt \
  inference.n_samples=64 \
  inference.save_path=runs/fashion_grpo_jpeg_moderate_long_20260516_213332/eval_compare/inference_grid.png \
  device=mps
```

Pointers:

- `train_grpo.py` — GRPO training loop.
- `rl/sde_sampler.py` — ODE-to-SDE rollout and log-prob recomputation.
- `rl/grpo.py` — group-relative advantage, Gaussian KL, clipped surrogate loss.
- `rl/reward.py` — classifier, fixed-class, and JPEG rewards.
- `rl/compression.py` — JPEG/PNG byte and bpp helpers.
- `configs/experiment/fashion_grpo*.yaml`
- `configs/reward/fashion_*.yaml`
- `configs/rl_training/*.yaml`
- `experiments/jpeg_compression_grpo/experiment.md` — JPEG reward write-up and figures.

## ImageNet-256 latent DiT / MoE

ImageNet is trained in SD-VAE latent space (`[4, 32, 32]`) using an mmap latent cache. The large runs use RunPod/SkyPilot managed-job chains.

Data/cache docs:

- `docs/imagenet256_storage.md`
- `scripts/build_imagenet_latent_cache.py`
- `scripts/convert_imagenet_latent_shards_to_mmap.py`
- `scripts/preflight_imagenet_latent_mmap.py`
- `datasets.py` (`ImageNetLatentMMapDataset`)
- `vae.py`

Local smoke once a latent mmap cache exists:

```bash
uv run python train.py \
  experiment=imagenet256_latent_dit_masked_smoke \
  dataset.cache_root=/path/to/imagenet256/sd-vae-ft-ema-mmap \
  device=cuda
```

Large H1024/D20 masked + unmasked chain template:

```bash
PATH="$HOME/.local/skypilot-shims:$PATH" \
  scripts/sky_runpod_chain.py cloud/runpod/imagenet256-dit-h1024-d20-training-chain.yaml \
    --infra auto \
    --gpu-request H100-SXM:1 \
    --volume-name nf-imagenet256 \
    --volume-size 100Gi \
    --image-id docker:ghcr.io/tqhdesilva/nanoflow:runpod-cu124 \
    --artifact-cloud-uri gs://nanoflow/runs \
    --dataset-cloud-uri gs://nanoflow/imagenet256/latent/sd-vae-ft-ema-mmap \
    --gcp-credentials ~/.config/gcp/nanoflow-gcs-reader.json
```

For dry-run validation, add `--dry-run` and optionally `--output /tmp/chain.yaml`.

ImageNet sample generation / eval:

```bash
uv run python eval_imagenet.py \
  experiment=imagenet256_latent_dit_m2_moe_layerwise_h1024_d20_e16_c2_moew05_unmasked20 \
  device=cuda \
  eval.checkpoint=/path/to/checkpoints/latest.pt \
  eval.weights=raw \
  eval.generate=true \
  eval.compute_fid=false \
  eval.output_dir=/path/to/output \
  eval.generation.num_samples=64 \
  eval.generation.batch_size=16 \
  eval.generation.num_steps=1000 \
  eval.generation.guidance_scale=4.0 \
  eval.generation.grid_path=/path/to/grid.png
```

Pointers:

- `docs/runpod_imagenet256_dit_chain.md`
- `docs/imagenet_eval.md`
- `cloud/runpod/imagenet256-dit-h1024-d20-training-chain.yaml`
- `cloud/runpod/imagenet256-dit-h1024-d20-heun-eval.yaml`
- `configs/dataset/imagenet256_latent_mmap.yaml`
- `configs/vae/sd_vae_ft_ema.yaml`
- `configs/experiment/imagenet256_latent_dit_m2_moe_layerwise_h1024_d20_e16_c2_moew05_masked80.yaml`
- `configs/experiment/imagenet256_latent_dit_m2_moe_layerwise_h1024_d20_e16_c2_moew05_unmasked20.yaml`
- `configs/model/classcond_deferred_dit_imagenet256_latent_m2_moe_layerwise_h1024_d20_e16_c2_moew05.yaml`
- `models_dit.py`
- `eval_imagenet.py`
