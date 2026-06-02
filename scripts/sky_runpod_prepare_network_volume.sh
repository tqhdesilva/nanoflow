#!/usr/bin/env bash
set -euo pipefail

TASK_FILE="${TASK_FILE:-cloud/runpod/prepare-network-volume.yaml}"
CLUSTER="${CLUSTER:-nf-imagenet-prepare}"
CACHE_ROOT="${DATASET_CACHE_ROOT:-/workspace/latent-caches/imagenet256/current}"
EXTRA_ARGS=()
DATASET_URI="${DATASET_GCS_URI:-}"
RUNPOD_INFRA="${RUNPOD_INFRA:-}"

usage() {
  cat <<'EOF'
Usage:
  scripts/sky_runpod_prepare_network_volume.sh \
    --infra runpod/<country>/<data-center> \
    --dataset-gcs-uri gs://<bucket>/<prefix> \
    [--cache-root /workspace/latent-caches/imagenet256/current] \
    [--cluster nf-imagenet-prepare] \
    [--secret GCP_SERVICE_ACCOUNT_JSON_B64=...]

Any extra arguments are passed to sky launch.
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --infra)
      RUNPOD_INFRA="${2:?missing value for --infra}"
      shift 2
      ;;
    --dataset-gcs-uri)
      DATASET_URI="${2:?missing value for --dataset-gcs-uri}"
      shift 2
      ;;
    --cache-root)
      CACHE_ROOT="${2:?missing value for --cache-root}"
      shift 2
      ;;
    --cluster)
      CLUSTER="${2:?missing value for --cluster}"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

if [ -z "$RUNPOD_INFRA" ]; then
  echo "missing --infra, for example runpod/US/US-CA-2" >&2
  exit 1
fi
if [ -z "$DATASET_URI" ]; then
  echo "missing --dataset-gcs-uri, for example gs://bucket/path" >&2
  exit 1
fi
if [[ "$DATASET_URI" != gs://* ]]; then
  echo "--dataset-gcs-uri must start with gs://" >&2
  exit 1
fi

exec sky launch \
  -c "$CLUSTER" \
  --infra "$RUNPOD_INFRA" \
  --env "DATASET_GCS_URI=$DATASET_URI" \
  --env "DATASET_CACHE_ROOT=$CACHE_ROOT" \
  "${EXTRA_ARGS[@]}" \
  "$TASK_FILE"
