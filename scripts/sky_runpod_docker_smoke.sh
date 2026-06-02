#!/usr/bin/env bash
set -euo pipefail

IMAGE="${NANOFLOW_RUNPOD_IMAGE:-ghcr.io/tqhdesilva/nanoflow:runpod-cu124}"
INFRA="${RUNPOD_INFRA:-runpod}"
GPUS="${RUNPOD_GPUS:-A40:1}"
JOB_NAME="${JOB_NAME:-nf-docker-cuda-smoke}"
TASK_FILE="${TASK_FILE:-cloud/runpod/docker-cuda-smoke.yaml}"

case "$IMAGE" in
  docker:*) IMAGE_ID="$IMAGE" ;;
  *) IMAGE_ID="docker:$IMAGE" ;;
esac

export PATH="$HOME/.local/skypilot-shims:$PATH"
exec sky jobs launch \
  -n "$JOB_NAME" \
  --infra "$INFRA" \
  --gpus "$GPUS" \
  --image-id "$IMAGE_ID" \
  -y \
  "$@" \
  "$TASK_FILE"
