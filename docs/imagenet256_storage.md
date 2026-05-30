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
