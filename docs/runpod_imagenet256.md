# RunPod setup for ImageNet 256 latent training

## Recommendation

Start without a custom Docker image. Use a RunPod PyTorch image, run `uv sync --frozen --no-dev`, and keep the uv cache plus virtualenv on the RunPod network volume. The first pod pays the install cost. Later pods reuse `/workspace/.cache/uv` and `/workspace/.venvs/nanoflow`.

Build a custom image only if startup time is still annoying after the network volume cache is warm, or if the exact CUDA, PyTorch, and system package set needs to be frozen.

RunPod network volumes mount at `/workspace`. SkyPilot can create and mount them with `type: runpod-network-volume`. SkyPilot can also use a Docker image as the runtime through `resources.image_id: docker:<image>`. Do not plan on running Docker inside the pod for this workflow.

## Network volume layout

```text
/workspace/
  .cache/uv/
  .cache/huggingface/
  .cache/torch/
  .venvs/nanoflow/
  latent-caches/imagenet256/
    sd-vae-ft-ema-fp16/
    current -> sd-vae-ft-ema-fp16
  runs/
```

Default GCS source for the latent cache:

```text
gs://${NANOFLOW_GCS_BUCKET}/imagenet256/latent/sd-vae-ft-ema-fp16
```

## GCS access

Use a service account key rather than interactive `gcloud auth login`.

For cache hydration only, grant the service account `roles/storage.objectViewer` on the bucket or the relevant prefix. If the pod will sync checkpoints or promoted artifacts back to GCS, add write permissions for those output prefixes.

Supported inputs for `scripts/runpod_hydrate_imagenet_latents.sh`:

```bash
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
```

or:

```bash
export GCP_SERVICE_ACCOUNT_JSON_B64="$(base64 -w0 /path/to/service-account.json)"
```

SkyPilot example:

```bash
sky launch -c nf-hydrate cloud/runpod/hydrate-latents.yaml \
  --secret GCP_SERVICE_ACCOUNT_JSON_B64="$(base64 -w0 /path/to/service-account.json)"
```

## Manual CPU pod hydration

Attach the RunPod network volume, open a shell, then run:

```bash
cd /workspace
git clone <nanoflow-repo-url> nanoflow
cd /workspace/nanoflow

export NANOFLOW_GCS_PROJECT=<gcp-project>
export NANOFLOW_GCS_BUCKET=<gcs-bucket>
export GCP_SERVICE_ACCOUNT_JSON_B64="$(base64 -w0 /path/to/service-account.json)"

bash scripts/runpod_hydrate_imagenet_latents.sh
```

The script installs `gcloud` when needed, authenticates, syncs the cache to `/workspace/latent-caches/imagenet256/sd-vae-ft-ema-fp16`, and updates `/workspace/latent-caches/imagenet256/current`.

Dry run:

```bash
DRY_RUN=1 NANOFLOW_GCS_BUCKET=<gcs-bucket> \
  bash scripts/runpod_hydrate_imagenet_latents.sh
```

## SkyPilot flow

Create the volume:

```bash
# Edit the data center and size first.
sky volumes apply cloud/runpod/runpod-volume.yaml
```

Hydrate cached latents:

```bash
sky launch -c nf-hydrate cloud/runpod/hydrate-latents.yaml \
  --env NANOFLOW_GCS_PROJECT=<gcp-project> \
  --env NANOFLOW_GCS_BUCKET=<gcs-bucket> \
  --secret GCP_SERVICE_ACCOUNT_JSON_B64="$(base64 -w0 /path/to/service-account.json)"
```

Short GPU smoke run:

```bash
sky launch -c nf-imagenet-ddp cloud/runpod/imagenet256-latent-ddp.yaml \
  --env SMOKE=1 \
  --env NPROC_PER_NODE=2 \
  --env BATCH_SIZE=8
```

Pilot run:

```bash
sky launch -c nf-imagenet-ddp cloud/runpod/imagenet256-latent-ddp.yaml
```

## Current decision

Do not build a custom image for the first RunPod pass. The initial setup already has to hydrate the latent cache onto the network volume, so the first `uv sync` overhead is acceptable. Keep dependency caches and the project venv on `/workspace` so follow-up pods reuse them.
