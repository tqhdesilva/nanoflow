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

The sharded cache metadata is written to `metadata.json`. It records the VAE id, transform, latent shape and dtype, source manifest hashes, split counts, and shard files.

## Training mmap latent cache

Convert the sharded `.pt` cache to the mmap training format before production-like random training:

```bash
uv run python scripts/convert_imagenet_latent_shards_to_mmap.py \
  --input-root /tmp/data/imagenet-256-latent-cache/sd-vae-ft-ema \
  --output-root /tmp/data/imagenet-256-latent-cache/sd-vae-ft-ema-mmap
```

The mmap cache layout is:

```text
metadata.json
train/latents.npy
train/labels.npy
train/source_paths.txt
val/latents.npy
val/labels.npy
val/source_paths.txt
```

Within each split, row `i` maps across all three files. `latents.npy[i]` is the `[4, 32, 32]` fp16 latent, `labels.npy[i]` is the numeric ImageNet class id, and line `i` of `source_paths.txt` is the original relative image path. The split is the parent directory, `train` or `val`.

Class ids are the standard ImageNet class indices from `imagenet_class_index.json` in the raw dataset root. That file maps string ids `"0"` through `"999"` to `[wnid, class_name]`. The cache builder infers labels from the WNID in each source filename, then stores the corresponding numeric id in `labels.npy`. To recover the class metadata for a sample, read the numeric label, convert it to a string key, and look it up in `imagenet_class_index.json`.

Upload and hydrate the mmap cache:

```bash
source .env
gcloud storage rsync -r \
  /tmp/data/imagenet-256-latent-cache/sd-vae-ft-ema-mmap \
  gs://${NANOFLOW_GCS_BUCKET}/imagenet256/latent/sd-vae-ft-ema-mmap

gcloud storage rsync -r \
  gs://${NANOFLOW_GCS_BUCKET}/imagenet256/latent/sd-vae-ft-ema-mmap \
  /tmp/data/imagenet-256-latent-cache/sd-vae-ft-ema-mmap
```
