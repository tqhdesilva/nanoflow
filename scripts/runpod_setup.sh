#!/usr/bin/env bash
set -euo pipefail

ROOT="${NANOFLOW_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
WORKSPACE="${WORKSPACE:-/workspace}"

export UV_CACHE_DIR="${UV_CACHE_DIR:-${WORKSPACE}/.cache/uv}"
export HF_HOME="${HF_HOME:-${WORKSPACE}/.cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-${WORKSPACE}/.cache/torch}"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-${WORKSPACE}/.venvs/nanoflow}"

mkdir -p "$UV_CACHE_DIR" "$HF_HOME" "$TORCH_HOME" "$(dirname "$UV_PROJECT_ENVIRONMENT")"

if ! command -v uv >/dev/null 2>&1; then
  python -m pip install --upgrade uv
fi

cd "$ROOT"
uv sync --frozen --no-dev --no-install-project
uv run python - <<'PY'
import torch
import torchvision
print(f"torch={torch.__version__} cuda_available={torch.cuda.is_available()} cuda={torch.version.cuda}")
print(f"torchvision={torchvision.__version__}")
PY
