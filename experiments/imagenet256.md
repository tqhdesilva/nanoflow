# Stage 4 — ImageNet-256 latent diffusion

This stage trains class-conditional ImageNet-256 flow models in SD-VAE latent space. It includes the initial latent UNet baseline and the final H1024/D20 deferred-masking DiT/MoE lineage.

## What this reproduces

- SD-VAE latent cache creation and mmap conversion.
- ImageNet-256 latent UNet baseline.
- ImageNet-256 latent DiT/MoE masked pretraining.
- ImageNet-256 latent DiT/MoE unmasked fine-tuning.
- Sample generation from the final checkpoint.

## Cost and runtime

Assumed cloud price for the recorded H100 runs: \$3.30/H100-hour.

| Run | Approx compute | Runtime | Cost |
|---|---:|---:|---:|
| Latent UNet baseline | ~8.4 EFLOPs | ~16.5 H100-hours | ~\$54.45 |
| H1024/D20 masked pretrain, initial | included below | 13h36m36s | \$44.91 |
| H1024/D20 masked pretrain, continuation | included below | 28h13m47s | \$93.16 |
| H1024/D20 low-LR unmasked finetune | included below | 21h55m54s | \$72.37 |
| H1024/D20 final-best lineage total | ~41.3 EFLOPs | 63h46m17s / 63.7714 H100-hours | \$210.45 |

## Data preparation

The production ImageNet runs use latents from `stabilityai/sd-vae-ft-ema`, shaped `[4, 32, 32]`, stored as an mmap cache.

Relevant docs/scripts:

- `docs/imagenet256_storage.md`
- `scripts/build_imagenet_latent_cache.py`
- `scripts/convert_imagenet_latent_shards_to_mmap.py`
- `scripts/preflight_imagenet_latent_mmap.py`

Build sharded latents from a local ImageNet-256 tree:

```bash
uv run python scripts/build_imagenet_latent_cache.py \
  --image-root /path/to/imagenet-256/ImageNet \
  --output-root /path/to/imagenet256_latents \
  --vae stabilityai/sd-vae-ft-ema \
  --device cuda \
  --batch-size 64 \
  --num-workers 8 \
  --torch-dtype float16
```

Convert to mmap format:

```bash
uv run python scripts/convert_imagenet_latent_shards_to_mmap.py \
  --input-root /path/to/imagenet256_latents \
  --output-root /path/to/sd-vae-ft-ema-mmap
```

Preflight the mmap cache:

```bash
uv run python scripts/preflight_imagenet_latent_mmap.py \
  --cache-root /path/to/sd-vae-ft-ema-mmap \
  --batch-size 8 \
  --num-workers 2
```

## Shared config excerpts

Latent dataset:

```yaml
# configs/dataset/imagenet256_latent_mmap.yaml
_target_: datasets.ImageNetLatentMMapDataset
name: imagenet256_latent_mmap
cache_root: /path/to/sd-vae-ft-ema-mmap
latent_shape: [4, 32, 32]
latent_dtype: float16
label_dtype: int64
vae: stabilityai/sd-vae-ft-ema
transform_image_size: 256
transform_crop: resize
storage_format: mmap_npy_v1
cache_version: 1
num_classes: 1000
```

VAE wrapper:

```yaml
# configs/vae/sd_vae_ft_ema.yaml
_target_: vae.VAEWrapper
model_id: stabilityai/sd-vae-ft-ema
backend: diffusers_autoencoder_kl
latent_shape: [4, 32, 32]
image_size: 256
sample_posterior: false
```

## Latent UNet baseline

### Config files

- `configs/experiment/imagenet256_latent_cfg.yaml`
- `configs/model/classcond_unet_imagenet256_latent.yaml`
- `configs/dataset/imagenet256_latent_mmap.yaml`
- `configs/vae/sd_vae_ft_ema.yaml`

### Effective config excerpt

```yaml
# configs/model/classcond_unet_imagenet256_latent.yaml
_target_: models.ClassCondUNet
in_ch: 4
base_ch: 128
depth: 4
time_dim: 512
use_attn: true
num_classes: 1000
```

Representative production hyperparameters:

```yaml
training:
  batch_size: 1024
  epochs: 400
  lr: 5.0e-4
  warmup_epochs: 2
  precision: bf16
  ema_decay: 0.0
  p_uncond: 0.1
inference:
  sampler:
    num_steps: 200
    latent_shape: [4, 32, 32]
  class_sampler:
    guidance_scale: 3.0
```

### Reproduce training

```bash
uv run python train.py \
  experiment=imagenet256_latent_cfg \
  device=cuda \
  dataset.cache_root=/path/to/sd-vae-ft-ema-mmap \
  training.batch_size=1024 \
  training.epochs=400 \
  training.precision=bf16 \
  training.ema_decay=0.0 \
  training.p_uncond=0.1
```

## H1024/D20 deferred-masking DiT/MoE

### Config files

- `configs/experiment/imagenet256_latent_dit_m2_moe_layerwise_h1024_d20_e16_c2_moew05_masked80.yaml`
- `configs/experiment/imagenet256_latent_dit_m2_moe_layerwise_h1024_d20_e16_c2_moew05_unmasked20.yaml`
- `configs/model/classcond_deferred_dit_imagenet256_latent_m2_moe_layerwise_h1024_d20_e16_c2_moew05.yaml`
- `cloud/runpod/imagenet256-dit-h1024-d20-training-chain.yaml`
- `docs/runpod_imagenet256_dit_chain.md`

### Effective model summary

```yaml
# configs/model/classcond_deferred_dit_imagenet256_latent_m2_moe_layerwise_h1024_d20_e16_c2_moew05.yaml
_target_: models_dit.ClassCondDeferredMaskingDiT
in_ch: 4
latent_size: 32
patch_size: 2
num_classes: 1000
masker:
  _target_: models_dit.RandomTokenMasker
  mask_ratio: 0.75
patch_mixer:
  hidden_size: 768
  blocks: 2
backbone:
  hidden_size: 1024
  blocks: 20
  attention_heads: 16
  moe_experts: 16
  expert_capacity: 2.0
```

Model size:

```yaml
total_parameters: 695.1M
token_average_active_parameters: 314.0M
```

### Masked pretraining config

Base config:

```yaml
# configs/experiment/..._masked80.yaml
training:
  batch_size: 320
  epochs: 80
  lr_schedule_epochs: 80
  lr_min_factor: 0.05
  lr: 1.0e-4
  optimizer: adamw
  weight_decay: 0.01
  warmup_epochs: 1
  precision: bf16
  compile_model: true
  compile_mode: reduce-overhead
  ema_decay: 0.0
  p_uncond: 0.1
  loss_mode: masked_mse
```

Final-best lineage used 240 total masked epochs. Continue from the same run directory with `training.resume=auto` and set the total epoch horizon to 240:

```bash
uv run python train.py \
  experiment=imagenet256_latent_dit_m2_moe_layerwise_h1024_d20_e16_c2_moew05_masked80 \
  device=cuda \
  dataset.cache_root=/path/to/sd-vae-ft-ema-mmap \
  training.run_dir=/path/to/runs/<chain_id>/masked_pretrain \
  training.resume=auto \
  training.epochs=240 \
  training.lr_schedule_epochs=240 \
  inference=null
```

Masked compute estimate:

```text
240 epochs × 1,281,167 images × 86.72G train FLOPs/image ≈ 26.7 EFLOPs
```

### Low-LR unmasked fine-tuning config

Base config:

```yaml
# configs/experiment/..._unmasked20.yaml
model:
  masker: null
training:
  batch_size: 80
  epochs: 20
  lr_schedule_epochs: 20
  lr_min_factor: 0.05
  lr: 1.0e-5
  optimizer: adamw
  weight_decay: 0.01
  warmup_epochs: 1
  precision: bf16
  compile_model: true
  compile_mode: reduce-overhead
  ema_decay: 0.0
  p_uncond: 0.1
  loss_mode: mse
```

Final-best lineage used 40 unmasked epochs initialized from the masked checkpoint:

```bash
uv run python train.py \
  experiment=imagenet256_latent_dit_m2_moe_layerwise_h1024_d20_e16_c2_moew05_unmasked20 \
  device=cuda \
  dataset.cache_root=/path/to/sd-vae-ft-ema-mmap \
  training.run_dir=/path/to/runs/<chain_id>/unmasked_finetune \
  training.resume=auto \
  training.init_from=/path/to/runs/<chain_id>/masked_pretrain/checkpoints/latest.pt \
  training.epochs=40 \
  training.lr_schedule_epochs=40 \
  inference=null
```

Unmasked compute estimate:

```text
40 epochs × 1,281,167 images × 284.86G train FLOPs/image ≈ 14.6 EFLOPs
```

Total final-best lineage estimate:

```text
26.7 EFLOPs + 14.6 EFLOPs ≈ 41.3 EFLOPs
```

## Cloud-chain reproduction

The reusable chain template is `cloud/runpod/imagenet256-dit-h1024-d20-training-chain.yaml`. Use placeholder values for cloud storage, volume names, credentials, and output paths; do not commit rendered chain YAMLs or secret env files.

Dry-run/render example:

```bash
uv run python scripts/sky_runpod_chain.py \
  cloud/runpod/imagenet256-dit-h1024-d20-training-chain.yaml \
  --dry-run \
  --chain-id <chain_id> \
  --infra auto \
  --gpu-request <gpu-request> \
  --volume-name <volume_name> \
  --volume-size 100Gi \
  --image-id <docker-image> \
  --dataset-cloud-uri <latent-cache-cloud-uri> \
  --artifact-cloud-uri <artifact-cloud-uri> \
  --gcp-credentials <path-to-service-account-json>
```

Launch by removing `--dry-run` after inspecting the rendered YAML.

## Generate samples from the final checkpoint

```bash
uv run python eval_imagenet.py \
  experiment=imagenet256_latent_dit_m2_moe_layerwise_h1024_d20_e16_c2_moew05_unmasked20 \
  device=cuda \
  eval.checkpoint=/path/to/runs/<chain_id>/unmasked_finetune/checkpoints/latest.pt \
  eval.output_dir=/path/to/runs/<chain_id>/eval_generate \
  eval.generate=true \
  eval.compute_fid=false \
  eval.make_stats=false \
  eval.generation.num_samples=64 \
  eval.generation.batch_size=16 \
  eval.generation.num_steps=200 \
  eval.generation.guidance_scale=3.0
```

For the sample grid shown in the writeup, the final visual sampling run used a longer Euler sampler and CFG scale 4.0.

## Code pointers

- `models_dit.py` — DiT, deferred masking, layerwise scaling, packed MoE.
- `train.py` — masked/unmasked training loop.
- `eval_imagenet.py` — sample generation and optional FID.
- `scripts/estimate_dit_complexity.py` — DiT/MoE compute estimates.
- `scripts/estimate_training_cost.py` — profile-to-cost estimates.
- `scripts/sky_runpod_chain.py` — cloud-chain renderer/launcher.
