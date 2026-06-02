#!/usr/bin/env bash
set -euo pipefail

ROOT="${NANOFLOW_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
WORKSPACE="${WORKSPACE:-/workspace}"

export UV_CACHE_DIR="${UV_CACHE_DIR:-${WORKSPACE}/.cache/uv}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-${WORKSPACE}/.cache/uv/python}"
export HF_HOME="${HF_HOME:-${WORKSPACE}/.cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-${WORKSPACE}/.cache/torch}"
export TMPDIR="${NANOFLOW_TMPDIR:-${WORKSPACE}/.tmp}"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-${WORKSPACE}/.venvs/nanoflow}"
RUNPOD_TORCH_INDEX_URL="${RUNPOD_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu124}"
RUNPOD_TORCH_PACKAGES="${RUNPOD_TORCH_PACKAGES:-torch==2.6.0+cu124 torchvision==0.21.0+cu124}"

mkdir -p "$UV_CACHE_DIR" "$UV_PYTHON_INSTALL_DIR" "$HF_HOME" "$TORCH_HOME" "$TMPDIR" "$(dirname "$UV_PROJECT_ENVIRONMENT")"

if ! command -v uv >/dev/null 2>&1; then
  python -m pip install --upgrade uv
fi

if [ -f "${UV_PROJECT_ENVIRONMENT}/pyvenv.cfg" ] && [ ! -x "${UV_PROJECT_ENVIRONMENT}/bin/python" ]; then
  echo "removing invalid virtualenv with missing Python: ${UV_PROJECT_ENVIRONMENT}" >&2
  rm -rf "$UV_PROJECT_ENVIRONMENT"
fi

install_runpod_torch_wheels() {
  if [ -z "$RUNPOD_TORCH_PACKAGES" ]; then
    return
  fi
  read -r -a torch_packages <<< "$RUNPOD_TORCH_PACKAGES"
  uv pip install --python "${UV_PROJECT_ENVIRONMENT}/bin/python" \
    --index-url "$RUNPOD_TORCH_INDEX_URL" \
    "${torch_packages[@]}"
}

cd "$ROOT"
uv sync --frozen --no-dev --no-install-project
install_runpod_torch_wheels
uv run python - <<'PY'
import torch
import torchvision
print(f"torch={torch.__version__} cuda_available={torch.cuda.is_available()} cuda={torch.version.cuda}")
print(f"torchvision={torchvision.__version__}")
PY
