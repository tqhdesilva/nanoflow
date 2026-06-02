#!/usr/bin/env bash
set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"
DATASET_CACHE_ROOT="${DATASET_CACHE_ROOT:-${WORKSPACE}/latent-caches/imagenet256/current}"
RUNPOD_ENV_FILE="${RUNPOD_ENV_FILE:-${WORKSPACE}/nanoflow_runpod.env}"
NANOFLOW_REPO_URL="${NANOFLOW_REPO_URL:-https://github.com/tqhdesilva/nanoflow.git}"
NANOFLOW_REPO_REF="${NANOFLOW_REPO_REF:-main}"
NANOFLOW_REPO_DIR="${NANOFLOW_REPO_DIR:-${WORKSPACE}/nanoflow}"
NANOFLOW_VENV="${NANOFLOW_VENV:-${WORKSPACE}/.venvs/nanoflow}"
UV_CACHE_DIR="${UV_CACHE_DIR:-${WORKSPACE}/.cache/uv}"
HF_HOME="${HF_HOME:-${WORKSPACE}/.cache/huggingface}"
TORCH_HOME="${TORCH_HOME:-${WORKSPACE}/.cache/torch}"
TMPDIR="${NANOFLOW_TMPDIR:-${WORKSPACE}/.tmp}"
INSTALL_GCLOUD="${INSTALL_GCLOUD:-1}"
INSTALL_SYSTEM_PACKAGES="${INSTALL_SYSTEM_PACKAGES:-1}"

export UV_CACHE_DIR HF_HOME TORCH_HOME TMPDIR
if [ "${DRY_RUN:-0}" != "1" ]; then
  mkdir -p "$UV_CACHE_DIR" "$HF_HOME" "$TORCH_HOME" "$TMPDIR" "$(dirname "$NANOFLOW_VENV")"
fi

: "${DATASET_GCS_URI:?set DATASET_GCS_URI to a fully qualified gs://bucket/prefix URI}"

if [[ "$DATASET_GCS_URI" != gs://* ]]; then
  echo "DATASET_GCS_URI must start with gs://" >&2
  exit 1
fi

gcs_without_scheme="${DATASET_GCS_URI#gs://}"
NANOFLOW_GCS_BUCKET="${gcs_without_scheme%%/*}"
LATENT_GCS_PATH="${gcs_without_scheme#*/}"
if [ -z "$NANOFLOW_GCS_BUCKET" ] || [ "$LATENT_GCS_PATH" = "$gcs_without_scheme" ] || [ -z "$LATENT_GCS_PATH" ]; then
  echo "DATASET_GCS_URI must include a bucket and prefix, for example gs://bucket/path" >&2
  exit 1
fi

sudo_cmd=()
if [ "$(id -u)" -ne 0 ]; then
  if command -v sudo >/dev/null 2>&1; then
    sudo_cmd=(sudo)
  else
    echo "This script needs root or sudo for package installation." >&2
    exit 1
  fi
fi

install_system_packages() {
  if [ "$INSTALL_SYSTEM_PACKAGES" != "1" ]; then
    return
  fi
  if ! command -v apt-get >/dev/null 2>&1; then
    return
  fi
  "${sudo_cmd[@]}" apt-get update
  "${sudo_cmd[@]}" apt-get install -y git curl ca-certificates gnupg python3 python3-pip python3-venv
}

ensure_uv() {
  if command -v uv >/dev/null 2>&1; then
    return
  fi
  python3 -m pip install --upgrade uv
}

install_gcloud() {
  if command -v gcloud >/dev/null 2>&1; then
    return
  fi
  if [ "$INSTALL_GCLOUD" != "1" ]; then
    echo "gcloud is missing. Set INSTALL_GCLOUD=1 or install google-cloud-cli." >&2
    exit 1
  fi
  if ! command -v apt-get >/dev/null 2>&1; then
    echo "Cannot auto install gcloud without apt-get." >&2
    exit 1
  fi
  "${sudo_cmd[@]}" install -d -m 0755 /etc/apt/keyrings
  curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg \
    | "${sudo_cmd[@]}" gpg --dearmor --yes -o /etc/apt/keyrings/cloud.google.gpg
  echo "deb [signed-by=/etc/apt/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" \
    | "${sudo_cmd[@]}" tee /etc/apt/sources.list.d/google-cloud-sdk.list >/dev/null
  "${sudo_cmd[@]}" apt-get update
  "${sudo_cmd[@]}" apt-get install -y google-cloud-cli
}

auth_gcloud() {
  local key_file=""
  if [ -n "${GCP_SERVICE_ACCOUNT_JSON_B64:-}" ]; then
    key_file="${RUNPOD_GCP_KEY_FILE:-/tmp/nanoflow-gcp-sa.json}"
    printf '%s' "$GCP_SERVICE_ACCOUNT_JSON_B64" | base64 -d > "$key_file"
  elif [ -n "${GCP_SERVICE_ACCOUNT_JSON:-}" ]; then
    key_file="${RUNPOD_GCP_KEY_FILE:-/tmp/nanoflow-gcp-sa.json}"
    printf '%s' "$GCP_SERVICE_ACCOUNT_JSON" > "$key_file"
  elif [ -n "${GOOGLE_APPLICATION_CREDENTIALS:-}" ] && [ -f "$GOOGLE_APPLICATION_CREDENTIALS" ]; then
    key_file="$GOOGLE_APPLICATION_CREDENTIALS"
  fi

  if [ -n "$key_file" ]; then
    gcloud auth activate-service-account --key-file "$key_file" >/dev/null
  elif ! gcloud auth list --filter=status:ACTIVE --format='value(account)' | grep -q .; then
    cat >&2 <<'EOF'
No active gcloud auth found.
Provide one of these before running this script:
  GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
  GCP_SERVICE_ACCOUNT_JSON_B64=$(base64 -w0 service-account.json)
  GCP_SERVICE_ACCOUNT_JSON='{"type":"service_account",...}'
EOF
    exit 1
  fi

  if [ -n "${NANOFLOW_GCS_PROJECT:-}" ]; then
    gcloud config set project "$NANOFLOW_GCS_PROJECT" >/dev/null
  fi
}

clone_repo() {
  mkdir -p "$(dirname "$NANOFLOW_REPO_DIR")"
  if [ -d "${NANOFLOW_REPO_DIR}/.git" ]; then
    git -C "$NANOFLOW_REPO_DIR" remote set-url origin "$NANOFLOW_REPO_URL"
    git -C "$NANOFLOW_REPO_DIR" fetch --depth 1 origin "$NANOFLOW_REPO_REF"
    git -C "$NANOFLOW_REPO_DIR" checkout --detach FETCH_HEAD
  else
    rm -rf "$NANOFLOW_REPO_DIR"
    git clone --filter=blob:none --no-checkout "$NANOFLOW_REPO_URL" "$NANOFLOW_REPO_DIR"
    git -C "$NANOFLOW_REPO_DIR" fetch --depth 1 origin "$NANOFLOW_REPO_REF"
    git -C "$NANOFLOW_REPO_DIR" checkout --detach FETCH_HEAD
  fi
}

install_python_deps() {
  mkdir -p "$UV_CACHE_DIR" "$HF_HOME" "$TORCH_HOME" "$TMPDIR" "$(dirname "$NANOFLOW_VENV")"
  export UV_PROJECT_ENVIRONMENT="$NANOFLOW_VENV"
  cd "$NANOFLOW_REPO_DIR"
  uv sync --frozen --no-dev --no-install-project
}

write_env_file() {
  mkdir -p "$(dirname "$RUNPOD_ENV_FILE")"
  : > "$RUNPOD_ENV_FILE"
  {
    printf '# NanoFlow RunPod environment. Source this in training pods.\n'
    printf 'export WORKSPACE=%q\n' "$WORKSPACE"
    printf 'export NANOFLOW_REPO_ROOT=%q\n' "$NANOFLOW_REPO_DIR"
    printf 'export UV_PROJECT_ENVIRONMENT=%q\n' "$NANOFLOW_VENV"
    printf 'export UV_CACHE_DIR=%q\n' "$UV_CACHE_DIR"
    printf 'export HF_HOME=%q\n' "$HF_HOME"
    printf 'export TORCH_HOME=%q\n' "$TORCH_HOME"
    printf 'export TMPDIR=%q\n' "$TMPDIR"
    printf 'export DATASET_GCS_URI=%q\n' "$DATASET_GCS_URI"
    printf 'export LATENT_CACHE_GCS_URI=%q\n' "$DATASET_GCS_URI"
    printf 'export NANOFLOW_GCS_BUCKET=%q\n' "$NANOFLOW_GCS_BUCKET"
    printf 'export LATENT_GCS_PATH=%q\n' "$LATENT_GCS_PATH"
    printf 'export DATASET_CACHE_ROOT=%q\n' "$DATASET_CACHE_ROOT"
    printf 'export LATENT_CACHE_ROOT=%q\n' "$DATASET_CACHE_ROOT"
    printf 'export LATENT_CACHE_LINK=%q\n' "$DATASET_CACHE_ROOT"
    printf 'export RUNS_DIR=%q\n' "${RUNS_DIR:-${WORKSPACE}/runs}"
    printf 'export SKIP_SETUP=%q\n' "${SKIP_SETUP_AFTER_PREPARE:-1}"
  } >> "$RUNPOD_ENV_FILE"
}

sync_cache() {
  mkdir -p "$DATASET_CACHE_ROOT"
  gcloud_args=()
  if [ -n "${NANOFLOW_GCS_PROJECT:-}" ]; then
    gcloud_args=(--project "$NANOFLOW_GCS_PROJECT")
  fi
  gcloud "${gcloud_args[@]}" storage rsync -r "$DATASET_GCS_URI" "$DATASET_CACHE_ROOT"
  if [ ! -f "${DATASET_CACHE_ROOT}/metadata.json" ]; then
    echo "metadata.json missing after sync: ${DATASET_CACHE_ROOT}" >&2
    exit 1
  fi
}

print_summary() {
  "${NANOFLOW_VENV}/bin/python" - <<PY
import json
from pathlib import Path
root = Path(${DATASET_CACHE_ROOT@Q})
meta = json.loads((root / "metadata.json").read_text())
counts = {k: v.get("count") for k, v in meta.get("splits", {}).items()}
print(f"repo={Path(${NANOFLOW_REPO_DIR@Q})}")
print(f"venv={Path(${NANOFLOW_VENV@Q})}")
print(f"cache={root}")
print(f"split_counts={counts}")
print(f"env_file={Path(${RUNPOD_ENV_FILE@Q})}")
PY
}

if [ "${DRY_RUN:-0}" = "1" ]; then
  echo "would clone $NANOFLOW_REPO_URL ref $NANOFLOW_REPO_REF to $NANOFLOW_REPO_DIR"
  echo "would create venv $NANOFLOW_VENV and run: UV_CACHE_DIR=$UV_CACHE_DIR TMPDIR=$TMPDIR UV_PROJECT_ENVIRONMENT=$NANOFLOW_VENV uv sync --frozen --no-dev --no-install-project"
  echo "would sync: gcloud storage rsync -r $DATASET_GCS_URI $DATASET_CACHE_ROOT"
  echo "would write env file: $RUNPOD_ENV_FILE"
  exit 0
fi

install_system_packages
ensure_uv
install_gcloud
auth_gcloud
clone_repo
install_python_deps
write_env_file
sync_cache
print_summary
