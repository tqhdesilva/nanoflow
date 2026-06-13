#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "usage: scripts/runpod_gcs_rsync.sh SOURCE DEST" >&2
  exit 2
fi

SOURCE="$1"
DEST="$2"
INSTALL_GCLOUD="${INSTALL_GCLOUD:-1}"

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

install_gcloud
auth_gcloud

gcloud_args=()
if [ -n "${NANOFLOW_GCS_PROJECT:-}" ]; then
  gcloud_args=(--project "$NANOFLOW_GCS_PROJECT")
fi

gcloud "${gcloud_args[@]}" storage rsync -r "$SOURCE" "$DEST"
