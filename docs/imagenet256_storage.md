# ImageNet 256 GCS upload

Set env vars, or source `.env`:

```bash
export NANOFLOW_GCS_PROJECT=<your-gcp-project>
export NANOFLOW_GCS_BUCKET=<your-gcs-bucket>
```

Upload the extracted Kaggle dataset. The script downloads `imagenet_class_index.json` into the local root if it is missing, so the class ID mapping is uploaded too:

```bash
IMAGENET256_LOCAL_ROOT=/path/to/extracted/ImageNet \
  scripts/upload_imagenet256_to_gcs.sh
```

Defaults:

```bash
IMAGENET256_LOCAL_ROOT=/tmp/data/imagenet256-test/ImageNet
IMAGENET256_GCS_PATH=imagenet256/raw/kaggle-nguynhoitrung-imagenet256
```

Optional Kaggle CLI download command, if not downloading manually:

```bash
kaggle datasets download nguynhoitrung/imagenet256 -p /tmp/data --unzip
```

Smoke test without uploading:

```bash
source .env
DRY_RUN=1 scripts/upload_imagenet256_to_gcs.sh
```

## VAE smoke test

Stage 2 uses `stabilityai/sd-vae-ft-ema` by default for ImageNet 256 latent work. Run one small encode and decode pass against the local sample folder:

```bash
uv run python scripts/vae_smoke.py \
  --image-root /tmp/data/imagenet256-test/ImageNet \
  --batch-size 1 \
  --device mps \
  --output /tmp/nanoflow_vae_smoke.png
```

The smoke script writes originals and reconstructions to the output grid and prints the encoded latent shape.

## Latent cache build and sync

Build a sharded train and validation cache from the local ImageNet 256 tree:

```bash
uv run python scripts/build_imagenet_latent_cache.py \
  --image-root /tmp/data/imagenet-256/ImageNet \
  --output-root /tmp/nanoflow_imagenet256_latents \
  --device mps \
  --batch-size 8 \
  --num-workers 8 \
  --prefetch-factor 4 \
  --max-pending-writes 2 \
  --shard-size 8192 \
  --compile-vae
```

For smoke tests, force small shards and small split limits:

```bash
uv run python scripts/build_imagenet_latent_cache.py \
  --image-root /tmp/data/imagenet-256/ImageNet \
  --output-root /tmp/nanoflow_imagenet_latent_smoke \
  --device mps \
  --batch-size 1 \
  --shard-size 1 \
  --max-samples 2
```

Upload and hydrate through GCS with plain `gcloud`:

```bash
source .env
gcloud storage rsync -r /tmp/nanoflow_imagenet_latent_smoke \
  gs://${NANOFLOW_GCS_BUCKET}/imagenet256/latent/smoke/stage2.3-cache

gcloud storage rsync -r \
  gs://${NANOFLOW_GCS_BUCKET}/imagenet256/latent/smoke/stage2.3-cache \
  /tmp/nanoflow_imagenet_latent_hydrated
```

The cache metadata is written to `metadata.json`. It records the VAE id, transform, latent shape and dtype, source manifest hashes, split counts, and shard files.
