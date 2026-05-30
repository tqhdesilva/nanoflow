#!/usr/bin/env bash
set -euo pipefail
: "${NANOFLOW_GCS_PROJECT:?set NANOFLOW_GCS_PROJECT}"
: "${NANOFLOW_GCS_BUCKET:?set NANOFLOW_GCS_BUCKET}"
: "${IMAGENET256_LOCAL_ROOT:=/tmp/data/imagenet256-test/ImageNet}"
: "${IMAGENET256_GCS_PATH:=imagenet256/raw/kaggle-nguynhoitrung-imagenet256}"
: "${IMAGENET_CLASS_INDEX_URL:=https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json}"
[ -d "$IMAGENET256_LOCAL_ROOT" ] || { echo "missing $IMAGENET256_LOCAL_ROOT" >&2; exit 1; }
GCS_URI="gs://${NANOFLOW_GCS_BUCKET}/${IMAGENET256_GCS_PATH}"
# Optional download: kaggle datasets download nguynhoitrung/imagenet256 -p "$(dirname "$IMAGENET256_LOCAL_ROOT")" --unzip
# Optional metadata: kaggle datasets metadata nguynhoitrung/imagenet256 -p "${IMAGENET256_LOCAL_ROOT}/_kaggle_metadata"
if [ ! -f "${IMAGENET256_LOCAL_ROOT}/imagenet_class_index.json" ]; then
  curl -L "$IMAGENET_CLASS_INDEX_URL" -o "${IMAGENET256_LOCAL_ROOT}/imagenet_class_index.json"
fi
if [ "${DRY_RUN:-0}" = "1" ]; then
  echo gcloud --project "$NANOFLOW_GCS_PROJECT" storage rsync -r "$IMAGENET256_LOCAL_ROOT" "$GCS_URI"
  exit 0
fi
gcloud --project "$NANOFLOW_GCS_PROJECT" storage rsync -r "$IMAGENET256_LOCAL_ROOT" "$GCS_URI"
echo "uploaded $IMAGENET256_LOCAL_ROOT to $GCS_URI"
