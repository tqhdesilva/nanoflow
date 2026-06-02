#!/usr/bin/env bash
set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"
RUNPOD_ENV_FILE="${RUNPOD_ENV_FILE:-${WORKSPACE}/nanoflow_runpod.env}"
if [ -f "$RUNPOD_ENV_FILE" ] && [ "${LOAD_RUNPOD_ENV:-1}" = "1" ]; then
  set -a
  # shellcheck disable=SC1090
  source "$RUNPOD_ENV_FILE"
  set +a
fi

ROOT="${NANOFLOW_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
WORKSPACE="${WORKSPACE:-/workspace}"
LATENT_CACHE_LINK="${LATENT_CACHE_LINK:-${DATASET_CACHE_ROOT:-${WORKSPACE}/latent-caches/imagenet256/current}}"
RUNS_DIR="${RUNS_DIR:-${WORKSPACE}/runs}"

if [ "${SKIP_SETUP:-0}" != "1" ] || { [ "${USE_PREPARED_VENV:-1}" = "1" ] && [ -n "${UV_PROJECT_ENVIRONMENT:-}" ] && [ ! -x "${UV_PROJECT_ENVIRONMENT}/bin/python" ]; }; then
  "$ROOT/scripts/runpod_setup.sh"
fi

PYTHON_CMD=(uv run python)
TORCHRUN_CMD=(uv run torchrun)
if [ "${USE_PREPARED_VENV:-1}" = "1" ] && [ -n "${UV_PROJECT_ENVIRONMENT:-}" ] && [ -x "${UV_PROJECT_ENVIRONMENT}/bin/python" ]; then
  PYTHON_CMD=("${UV_PROJECT_ENVIRONMENT}/bin/python")
  if [ -x "${UV_PROJECT_ENVIRONMENT}/bin/torchrun" ]; then
    TORCHRUN_CMD=("${UV_PROJECT_ENVIRONMENT}/bin/torchrun")
  else
    TORCHRUN_CMD=("${UV_PROJECT_ENVIRONMENT}/bin/python" -m torch.distributed.run)
  fi
fi

if [ ! -f "${LATENT_CACHE_LINK}/metadata.json" ]; then
  echo "missing latent cache metadata at ${LATENT_CACHE_LINK}/metadata.json" >&2
  echo "run scripts/runpod_prepare_network_volume.sh first" >&2
  exit 1
fi

if [ -n "${NPROC:-}" ] && [ -z "${NPROC_PER_NODE:-}" ]; then
  NPROC_PER_NODE="$NPROC"
fi
if [ -z "${NPROC_PER_NODE:-}" ]; then
  NPROC_PER_NODE=$("${PYTHON_CMD[@]}" - <<'PY'
import torch
print(max(torch.cuda.device_count(), 1))
PY
)
fi

mkdir -p "$RUNS_DIR"

if [ "${SMOKE:-0}" = "1" ]; then
  batch_size="${BATCH_SIZE:-8}"
  num_workers="${NUM_WORKERS:-2}"
  eval_every="${EVAL_EVERY:-1}"
  max_steps="${MAX_STEPS:-20}"
  epochs="${EPOCHS:-1}"
else
  batch_size="${BATCH_SIZE:-64}"
  num_workers="${NUM_WORKERS:-8}"
  eval_every="${EVAL_EVERY:-10}"
  max_steps="${MAX_STEPS:-}"
  epochs="${EPOCHS:-100}"
fi

args=(
  experiment=imagenet256_latent_cfg
  device=cuda
  dataset.cache_root="$LATENT_CACHE_LINK"
  runs_dir="$RUNS_DIR"
  training.epochs="$epochs"
  training.batch_size="$batch_size"
  training.num_workers="$num_workers"
  training.precision="${PRECISION:-bf16}"
  training.ema_decay="${EMA_DECAY:-0.9999}"
  training.checkpoint_every="${CHECKPOINT_EVERY:-1}"
  training.eval_every="$eval_every"
)

if [ -n "$max_steps" ]; then
  args+=(training.max_steps="$max_steps")
fi

if [ "${SMOKE:-0}" = "1" ]; then
  args+=(
    inference.n_samples="${INFERENCE_N_SAMPLES:-4}"
    inference.sampler.num_steps="${INFERENCE_NUM_STEPS:-20}"
    sample_logger.n_samples="${SAMPLE_LOGGER_N_SAMPLES:-4}"
    sample_logger.num_steps="${SAMPLE_LOGGER_NUM_STEPS:-20}"
  )
fi

if [ "${DISABLE_INFERENCE:-0}" = "1" ]; then
  args+=(inference=null sample_logger=null)
fi

args+=("$@")

cd "$ROOT"
if [ "$NPROC_PER_NODE" -gt 1 ]; then
  if [ "${NNODES:-1}" -gt 1 ]; then
    exec "${TORCHRUN_CMD[@]}" \
      --nnodes="${NNODES}" \
      --node_rank="${NODE_RANK:?set NODE_RANK for multi node}" \
      --nproc_per_node="$NPROC_PER_NODE" \
      --master_addr="${MASTER_ADDR:?set MASTER_ADDR for multi node}" \
      --master_port="${MASTER_PORT:-29500}" \
      train.py distributed=ddp "${args[@]}"
  fi
  exec "${TORCHRUN_CMD[@]}" --standalone --nproc_per_node="$NPROC_PER_NODE" train.py distributed=ddp "${args[@]}"
fi

exec "${PYTHON_CMD[@]}" train.py "${args[@]}"
