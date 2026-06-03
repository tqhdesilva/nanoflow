# RunPod setup for ImageNet 256 latent training

## Recommendation

Use the custom Docker image for Python dependencies and PyTorch CUDA. Use the RunPod network volume only as warm storage for latent caches, checkpoints, logs, and model caches. Run one CPU SkyPilot task in the target RunPod data center only when the latent cache needs hydration from GCS.

Pin the cache hydration pod and the later GPU training pod to the same RunPod data center as the network volume.

## Network volume layout

```text
/workspace/
  .cache/huggingface/
  .cache/torch/
  .tmp/
  latent-caches/imagenet256/current/
  runs/
```

The GCS cache source is passed at launch time as a full URI through `DATASET_GCS_URI`.

## Required launch inputs

- RunPod data center, for example `runpod/US/US-CA-2`.
- Dataset cache URI, for example `gs://<bucket>/<prefix>`.
- Optional network-volume destination path. The default is `/workspace/latent-caches/imagenet256/current`.
- Docker image, default `ghcr.io/tqhdesilva/nanoflow:runpod-cu124`.
- Optional `NANOFLOW_REPO_URL` and `NANOFLOW_REPO_REF` only for the legacy volume prep script.
- GCP service account credentials with read access to the cache prefix when hydrating data.

## RunPod launch workarounds

Use these checks before every RunPod launch:

1. Match the exact GPU SKU. In EU-NL-1, the working H100 request was `--gpus H100-SXM:2`. A generic `--gpus H100:2` can resolve to `2x_H100_SECURE`, which may fail even when H100 SXM stock is visible in the RunPod UI.
2. Keep the pod and volume in the same data center. The `nf-imagenet256` network volume is pinned to `runpod/NL/EU-NL-1`, so setup and training must use `--infra runpod/NL/EU-NL-1` unless a new volume is created and hydrated elsewhere.
3. Keep Python dependencies in the Docker image. Do not source `/workspace/nanoflow_runpod.env` for image-backed training, because it can point back to an old volume venv.
4. Use the `runpod-cu124` image unless a target host has been validated with a newer CUDA runtime. The image installs `torch==2.6.0+cu124` and `torchvision==0.21.0+cu124` from `https://download.pytorch.org/whl/cu124`.
5. Do not rely on SkyPilot autostop for RunPod. The current local SkyPilot CLI reports that autostop is unsupported for RunPod. After a non-managed smoke or pilot run, verify artifacts, then run `sky down <cluster> -y`. This preserves the network volume. Managed jobs clean up workers after completion.
6. When reusing an already launched cluster with `sky exec`, pass the same GPU request, for example `--gpus H100-SXM:1`, so the job receives the visible GPU.

## Choosing a RunPod data center

First confirm SkyPilot can use RunPod:

```bash
sky check runpod
```

Use RunPod's API to find live data center IDs, stock status, and network volume support. The SkyPilot CLI currently shows RunPod regions such as `CA` or `US`, while RunPod data center IDs are values such as `US-GA-1` or `CA-MTL-1`.

For a concise H100 table, use the repo helper:

```bash
RUNPOD_API_KEY=<runpod-api-key> scripts/runpod_list_h100_datacenters.sh
```

Useful option:

```bash
REQUIRE_STORAGE_SUPPORT=1 GPU_QUERY="H100" RUNPOD_API_KEY=<runpod-api-key> \
  scripts/runpod_list_h100_datacenters.sh
```

`REQUIRE_STORAGE_SUPPORT=1` keeps only data centers that support network volumes. It is the default, but keeping it explicit in launch notes avoids accidentally selecting a GPU-only data center. Set `GPU_QUERY` to the GPU family you want to search for, for example `H100`.

By default, the helper calls RunPod GraphQL directly and prints data centers whose `gpuAvailability` entry matches `GPU_QUERY`, has a non-empty `stockStatus`, and has `storageSupport=true`. This filters out data centers with H100 availability but no network volume support, for example locations like `AP-IN-1` when RunPod reports `storageSupport=false`. Results are sorted by `High`, `Medium`, then `Low`.

For raw RunPod CLI inspection, use:

```bash
RUNPOD_API_KEY=<runpod-api-key> runpodctl datacenter list -o json
```

Note that `runpodctl` may omit `storageSupport` from its output, so use the helper when filtering for network volume support.

Use SkyPilot's GPU catalog as a secondary price and shape check:

```bash
sky gpus list H100:8 --infra runpod --all-regions
sky gpus list H100:8 --infra runpod --all-regions -o json
```

For a quick optimizer view, let SkyPilot pick the cheapest country-level candidate without pinning a data center:

```bash
sky launch --dryrun --infra runpod --gpus H100:8 'echo ok'
```

After choosing a candidate RunPod data center ID, validate that SkyPilot accepts the full infra string:

```bash
sky launch --dryrun --infra runpod/US/US-CA-2 --gpus H100:8 'echo ok'
```

An invalid data center ID fails immediately. A successful dry run means the SkyPilot catalog accepts that GPU shape in that RunPod region. It does not reserve capacity and should not be treated as a live stock guarantee.

Network volumes require the full RunPod data center ID in the `infra` value. Use `scripts/runpod_list_h100_datacenters.sh` as the read-only filter for `storageSupport=true`, then create or reuse the project volume in the selected data center:

```bash
sky volumes apply --infra runpod/US/US-CA-2 cloud/runpod/runpod-volume.yaml
sky volumes ls --refresh -v
```

If you want a smaller disposable probe before creating the real project volume, use the minimum RunPod network volume size, then delete it:

```bash
sky volumes apply --name nf-rpvol-probe \
  --infra runpod/US/US-CA-2 \
  --type runpod-network-volume \
  --size 10Gi \
  -y
sky volumes delete nf-rpvol-probe -y
```

If `sky volumes apply` fails for the selected data center, choose another candidate and keep setup and training pinned to that new data center. Once a volume exists, the CPU setup task and GPU training task must use the same `--infra runpod/<country>/<data-center>` value.

SkyPilot's `gpus list` output is a catalog and pricing view. Use RunPod's `stockStatus` and `storageSupport` from the helper for live availability and network volume support, then feed the chosen data center ID back into SkyPilot for `sky volumes apply`, setup, and training.

## GCS access

Use a service account key rather than interactive `gcloud auth login`.

For cache hydration, grant the service account `roles/storage.objectViewer` on the bucket or relevant prefix.

Supported auth inputs:

```bash
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
```

or:

```bash
export GCP_SERVICE_ACCOUNT_JSON_B64="$(base64 < /path/to/service-account.json | tr -d '\n')"
```

## SkyPilot flow

Create or update the network volume in the chosen data center:

```bash
sky volumes apply --infra runpod/<country>/<data-center> cloud/runpod/runpod-volume.yaml
```

Prepare the volume from a CPU worker in the same data center only if the latent cache is not already present. The setup YAML keeps requirements small (`cpus: 2+`, `memory: 8+`, `disk_size: 5`) so SkyPilot can choose an available CPU pod. The network volume mounted at `/workspace` holds caches, latent data, checkpoints, and logs.

```bash
scripts/sky_runpod_prepare_network_volume.sh \
  --infra runpod/<country>/<data-center> \
  --dataset-gcs-uri gs://<bucket>/<prefix> \
  --secret GCP_SERVICE_ACCOUNT_JSON_B64="$(base64 < /path/to/service-account.json | tr -d '\n')"
```

If the default still has no stock, keep the request broad and let SkyPilot choose another CPU shape:

```bash
scripts/sky_runpod_prepare_network_volume.sh \
  --infra runpod/<country>/<data-center> \
  --dataset-gcs-uri gs://<bucket>/<prefix> \
  --cpus 2+ \
  --memory 4+ \
  --disk-size 5 \
  --retry-until-up \
  --secret GCP_SERVICE_ACCOUNT_JSON_B64="$(base64 < /path/to/service-account.json | tr -d '\n')"
```

Equivalent direct SkyPilot command:

```bash
sky launch -c nf-imagenet-prepare \
  --infra runpod/<country>/<data-center> \
  --env DATASET_GCS_URI=gs://<bucket>/<prefix> \
  --env DATASET_CACHE_ROOT=/workspace/latent-caches/imagenet256/current \
  --secret GCP_SERVICE_ACCOUNT_JSON_B64="$(base64 < /path/to/service-account.json | tr -d '\n')" \
  cloud/runpod/prepare-network-volume.yaml
```

Short GPU smoke run after the volume is prepared. In EU-NL-1, request the exact SKU shown by RunPod availability, for example `H100-SXM:1`.

```bash
PATH="$HOME/.local/skypilot-shims:$PATH" sky jobs launch \
  -n nf-imagenet-train-smoke \
  --infra runpod/<country>/<data-center> \
  --gpus H100-SXM:1 \
  --env SMOKE=1 \
  --env NPROC_PER_NODE=1 \
  --env BATCH_SIZE=8 \
  --env MAX_STEPS=20 \
  --env DISABLE_INFERENCE=1 \
  -y \
  cloud/runpod/imagenet256-latent-ddp.yaml
```

Managed jobs clean up workers after completion. If using `sky launch` instead of `sky jobs launch`, tear down the pod manually while preserving the network volume:

```bash
sky down nf-imagenet-ddp -y
```

Pilot run:

```bash
PATH="$HOME/.local/skypilot-shims:$PATH" sky jobs launch \
  -n nf-imagenet-train-pilot \
  --infra runpod/<country>/<data-center> \
  -y \
  cloud/runpod/imagenet256-latent-ddp.yaml
```

## Legacy manual CPU pod preparation

Attach the RunPod network volume, open a shell, then run:

```bash
cd /workspace
git clone <nanoflow-repo-url> nanoflow
cd /workspace/nanoflow

export DATASET_GCS_URI=gs://<bucket>/<prefix>
export GCP_SERVICE_ACCOUNT_JSON_B64="$(base64 < /path/to/service-account.json | tr -d '\n')"

bash scripts/runpod_prepare_network_volume.sh
```

Dry run:

```bash
DRY_RUN=1 DATASET_GCS_URI=gs://<bucket>/<prefix> \
  bash scripts/runpod_prepare_network_volume.sh
```

## Checkpoint retry policy

Training config includes a Hydra-instantiated `retryer` passed into `Trainer`. It retries only allowlisted transient IO failures during epoch-end and train-end callbacks, which covers checkpoint saves. The default allowlist includes transient `OSError` errno names such as `EIO`, `ESTALE`, `ETIMEDOUT`, network reset or unreachable errors, `EAGAIN`, and `EBUSY`, plus known PyTorch checkpoint writer messages such as `PytorchStreamWriter`, `file write failed`, `unexpected pos`, and `inline_container.cc`. Unknown exceptions and non-IO failures fail fast. Tune with overrides such as `retryer.max_retries=5`.

## Current decision

Move dependency setup into a custom Docker image for GPU training. Keep the RunPod network volume for latent caches, checkpoints, logs, and optional Hugging Face or torch caches. See `docs/runpod_docker.md` for the image build and managed smoke test.
