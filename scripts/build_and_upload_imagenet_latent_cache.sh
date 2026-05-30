#!/usr/bin/env bash
set -euo pipefail

: "${NANOFLOW_GCS_PROJECT:?set NANOFLOW_GCS_PROJECT}"
: "${NANOFLOW_GCS_BUCKET:?set NANOFLOW_GCS_BUCKET}"
: "${RAW_GCS_PATH:=imagenet256/raw/kaggle-nguynhoitrung-imagenet256}"
: "${CACHE_GCS_PATH:=imagenet256/latent/sd-vae-ft-ema-fp16}"
: "${RAW_ROOT:=/tmp/data/imagenet-256/ImageNet}"
: "${CACHE_ROOT:=/tmp/data/imagenet256-latent-cache/sd-vae-ft-ema-fp16}"
: "${DEVICE:=cuda}"
: "${BATCH_SIZE:=32}"
: "${NUM_WORKERS:=8}"
: "${PREFETCH_FACTOR:=2}"
: "${SHARD_SIZE:=8192}"
: "${TORCH_DTYPE:=float16}"

RAW_GCS_URI="gs://${NANOFLOW_GCS_BUCKET}/${RAW_GCS_PATH}"
CACHE_GCS_URI="gs://${NANOFLOW_GCS_BUCKET}/${CACHE_GCS_PATH}"

if [ "${DRY_RUN:-0}" = "1" ]; then
  echo gcloud --quiet --project "$NANOFLOW_GCS_PROJECT" storage rsync -r "$RAW_GCS_URI" "$RAW_ROOT"
  echo uv run python scripts/build_imagenet_latent_cache.py --image-root "$RAW_ROOT" --output-root "$CACHE_ROOT" --device "$DEVICE" --batch-size "$BATCH_SIZE" --num-workers "$NUM_WORKERS" --prefetch-factor "$PREFETCH_FACTOR" --shard-size "$SHARD_SIZE" --torch-dtype "$TORCH_DTYPE"
  echo gcloud --quiet --project "$NANOFLOW_GCS_PROJECT" storage rsync -r "$CACHE_ROOT" "$CACHE_GCS_URI"
  exit 0
fi

mkdir -p "$RAW_ROOT" "$CACHE_ROOT"
gcloud --quiet --project "$NANOFLOW_GCS_PROJECT" storage rsync -r "$RAW_GCS_URI" "$RAW_ROOT"

uv run python scripts/build_imagenet_latent_cache.py \
  --image-root "$RAW_ROOT" \
  --output-root "$CACHE_ROOT" \
  --device "$DEVICE" \
  --batch-size "$BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --prefetch-factor "$PREFETCH_FACTOR" \
  --shard-size "$SHARD_SIZE" \
  --torch-dtype "$TORCH_DTYPE"

gcloud --quiet --project "$NANOFLOW_GCS_PROJECT" storage rsync -r "$CACHE_ROOT" "$CACHE_GCS_URI"
echo "uploaded latent cache to $CACHE_GCS_URI"
