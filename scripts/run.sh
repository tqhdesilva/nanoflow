#!/usr/bin/env bash
# RunPod entrypoint. Saves artifacts to network volume at /workspace.
#
# Usage (single GPU):
#   ./scripts/run.sh dataset=cifar10 model=unet_cifar device=cuda
#
# Usage (multi-GPU):
#   NPROC=4 ./scripts/run.sh dataset=cifar10 model=unet_cifar device=cuda

set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"
CKPT_DIR="${WORKSPACE}/checkpoints"
LOG_DIR="${WORKSPACE}/runs"
NPROC="${NPROC:-1}"

mkdir -p "$CKPT_DIR" "$LOG_DIR"

COMMON_ARGS=(
    "training.checkpoint_dir=${CKPT_DIR}"
    "training.log_dir=${LOG_DIR}"
    "save=true"
    "$@"
)

if [ "$NPROC" -gt 1 ]; then
    exec torchrun --nproc_per_node="$NPROC" train.py "${COMMON_ARGS[@]}"
else
    exec uv run python train.py "${COMMON_ARGS[@]}"
fi
