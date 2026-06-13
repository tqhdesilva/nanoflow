#!/usr/bin/env bash
set -euo pipefail

ROOT="${NANOFLOW_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ENV_FILE="${ENV_FILE:-${ROOT}/.env}"
_pre_project="${NANOFLOW_GCS_PROJECT-}"
_pre_project_set="${NANOFLOW_GCS_PROJECT+x}"
_pre_bucket="${NANOFLOW_GCS_BUCKET-}"
_pre_bucket_set="${NANOFLOW_GCS_BUCKET+x}"
if [ -f "$ENV_FILE" ] && [ "${LOAD_ENV_FILE:-1}" = "1" ]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi
if [ -n "$_pre_project_set" ]; then
  NANOFLOW_GCS_PROJECT="$_pre_project"
fi
if [ -n "$_pre_bucket_set" ]; then
  NANOFLOW_GCS_BUCKET="$_pre_bucket"
fi

WORKSPACE="${WORKSPACE:-/workspace}"
: "${LATENT_CACHE_ROOT:=${DATASET_CACHE_ROOT:-${WORKSPACE}/latent-caches/imagenet256/current}}"
: "${LATENT_CACHE_LINK:=${LATENT_CACHE_ROOT}}"
: "${INSTALL_GCLOUD:=1}"

if [ -z "${DATASET_GCS_URI:-}" ] && [ -n "${LATENT_CACHE_GCS_URI:-}" ]; then
  DATASET_GCS_URI="$LATENT_CACHE_GCS_URI"
fi
if [ -z "${DATASET_GCS_URI:-}" ]; then
  if [ -n "${NANOFLOW_GCS_BUCKET:-}" ] && [ -n "${LATENT_GCS_PATH:-}" ]; then
    DATASET_GCS_URI="gs://${NANOFLOW_GCS_BUCKET}/${LATENT_GCS_PATH}"
  else
    echo "set DATASET_GCS_URI to a fully qualified gs://bucket/prefix URI" >&2
    exit 1
  fi
fi
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

GCS_URI="$DATASET_GCS_URI"

cache_ready() {
  [ -f "${LATENT_CACHE_ROOT}/metadata.json" ] \
    && [ -f "${LATENT_CACHE_ROOT}/train/latents.npy" ] \
    && [ -f "${LATENT_CACHE_ROOT}/train/labels.npy" ] \
    && [ -f "${LATENT_CACHE_ROOT}/train/source_paths.txt" ] \
    && [ -f "${LATENT_CACHE_ROOT}/val/latents.npy" ] \
    && [ -f "${LATENT_CACHE_ROOT}/val/labels.npy" ] \
    && [ -f "${LATENT_CACHE_ROOT}/val/source_paths.txt" ]
}

sudo_cmd=()
if [ "$(id -u)" -ne 0 ]; then
  if command -v sudo >/dev/null 2>&1; then
    sudo_cmd=(sudo)
  else
    echo "gcloud install needs root or sudo" >&2
    exit 1
  fi
fi

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
  "${sudo_cmd[@]}" apt-get update
  "${sudo_cmd[@]}" apt-get install -y ca-certificates curl gnupg
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
  local created_key_file="0"
  local old_umask=""
  if [ -n "${GCP_SERVICE_ACCOUNT_JSON_B64:-}" ]; then
    key_file="${RUNPOD_GCP_KEY_FILE:-$(mktemp /tmp/nanoflow-gcp-sa.XXXXXX.json)}"
    created_key_file="1"
    old_umask="$(umask)"
    umask 077
    printf '%s' "$GCP_SERVICE_ACCOUNT_JSON_B64" | base64 -d > "$key_file"
    umask "$old_umask"
  elif [ -n "${GCP_SERVICE_ACCOUNT_JSON:-}" ]; then
    key_file="${RUNPOD_GCP_KEY_FILE:-$(mktemp /tmp/nanoflow-gcp-sa.XXXXXX.json)}"
    created_key_file="1"
    old_umask="$(umask)"
    umask 077
    printf '%s' "$GCP_SERVICE_ACCOUNT_JSON" > "$key_file"
    umask "$old_umask"
  elif [ -n "${GOOGLE_APPLICATION_CREDENTIALS:-}" ] && [ -f "$GOOGLE_APPLICATION_CREDENTIALS" ]; then
    key_file="$GOOGLE_APPLICATION_CREDENTIALS"
  fi

  if [ -n "$key_file" ]; then
    gcloud auth activate-service-account --key-file "$key_file" >/dev/null
    if [ "$created_key_file" = "1" ] && [ "${RUNPOD_KEEP_GCP_KEY:-0}" != "1" ]; then
      rm -f "$key_file"
    fi
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

if [ "${DRY_RUN:-0}" = "1" ]; then
  echo "gcloud storage rsync -r $GCS_URI $LATENT_CACHE_ROOT"
  if [ "$LATENT_CACHE_LINK" != "$LATENT_CACHE_ROOT" ]; then
    echo "ln -sfn $LATENT_CACHE_ROOT $LATENT_CACHE_LINK"
  fi
  exit 0
fi

if [ "${SKIP_SYNC_IF_PRESENT:-1}" = "1" ] && cache_ready; then
  echo "latent cache already present: ${LATENT_CACHE_ROOT}"
  exit 0
fi

install_gcloud
auth_gcloud
mkdir -p "$LATENT_CACHE_ROOT" "$(dirname "$LATENT_CACHE_LINK")"
gcloud_args=()
if [ -n "${NANOFLOW_GCS_PROJECT:-}" ]; then
  gcloud_args=(--project "$NANOFLOW_GCS_PROJECT")
fi
gcloud "${gcloud_args[@]}" storage rsync -r "$GCS_URI" "$LATENT_CACHE_ROOT"

if [ ! -f "${LATENT_CACHE_ROOT}/metadata.json" ]; then
  echo "metadata.json missing after sync: ${LATENT_CACHE_ROOT}" >&2
  exit 1
fi

if [ "$LATENT_CACHE_LINK" != "$LATENT_CACHE_ROOT" ]; then
  ln -sfn "$LATENT_CACHE_ROOT" "$LATENT_CACHE_LINK"
fi
python_bin=python
if ! command -v python >/dev/null 2>&1; then
  python_bin=python3
fi
"$python_bin" - <<PY
import json
from pathlib import Path
root = Path(${LATENT_CACHE_ROOT@Q})
meta = json.loads((root / "metadata.json").read_text())
counts = {k: v.get("count") for k, v in meta.get("splits", {}).items()}
print(f"hydrated {root}")
print(f"split_counts={counts}")
print(f"current_link={Path(${LATENT_CACHE_LINK@Q})}")
PY
