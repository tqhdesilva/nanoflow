# RunPod Docker image

## Why Docker now

The old RunPod path put uv, Python, and PyTorch wheels on the network volume. That worked, but it made each new pod depend on the volume venv and on RunPod setup scripts. The image path bakes Python deps into the container. The network volume can then focus on data, checkpoints, and logs.

## PyTorch CUDA choice

Default image build args use the previously working RunPod wheel set:

```text
PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu124
PYTORCH_PACKAGES="torch==2.6.0+cu124 torchvision==0.21.0+cu124"
```

This avoids the lockfile's PyPI torch path, which can pull CUDA 13 packages. NVIDIA documents CUDA 12.x as requiring driver `>=525` for minor version compatibility, while CUDA 13.x requires `>=580`. RunPod driver versions can vary by host, so CUDA 12.x is the safer first image target.

If a selected RunPod host is known to support CUDA 12.8 well, override the build args with a newer supported PyTorch pair, for example:

```bash
docker build \
  --build-arg PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu128 \
  --build-arg 'PYTORCH_PACKAGES=torch==2.11.0+cu128 torchvision==0.26.0+cu128' \
  -t <registry>/nanoflow:runpod-cu128 .
```

## Build and push

The default automated path is GitHub Actions. `.github/workflows/docker-runpod.yml` builds the image on PRs and pushes to GHCR on pushes to `main`:

```text
ghcr.io/tqhdesilva/nanoflow:runpod-cu124
ghcr.io/tqhdesilva/nanoflow:runpod-cu124-<commit-sha>
```

After the first push, make the GHCR package public, or configure RunPod registry auth for GHCR. Public is simpler for the first smoke.

Manual equivalent:

```bash
docker build -t ghcr.io/tqhdesilva/nanoflow:runpod-cu124 .
docker push ghcr.io/tqhdesilva/nanoflow:runpod-cu124
```

The Dockerfile installs uv project dependencies with:

```bash
uv sync --frozen --no-dev --no-install-project \
  --no-install-package torch \
  --no-install-package torchvision
```

Then it installs the selected CUDA PyTorch wheels explicitly. The project itself is not installed as a wheel. The source is copied to `/app`, and `PYTHONPATH=/app` is set. The image intentionally does not set `ENTRYPOINT`, because SkyPilot passes a `bash` bootstrap command to RunPod to install and start SSH before running the task.

## Smoke test as a managed job

Use the SkyPilot shim for managed jobs:

```bash
scripts/sky_runpod_docker_smoke.sh
```

Equivalent explicit command:

```bash
PATH="$HOME/.local/skypilot-shims:$PATH" sky jobs launch \
  -n nf-docker-cuda-smoke \
  --infra runpod \
  --gpus A40:1 \
  --image-id docker:ghcr.io/tqhdesilva/nanoflow:runpod-cu124 \
  -y \
  cloud/runpod/docker-cuda-smoke.yaml
```

The job runs `scripts/runpod_cuda_smoke.py`. It verifies imports for the main NanoFlow deps, creates a CUDA tensor, instantiates `Trainer`, runs one tiny optimizer step, prints JSON, and exits. Managed jobs should tear down the worker after completion.

## Optional RunPod template

A RunPod template is useful for console-driven Pods, but it is not required for the SkyPilot managed smoke job. If you want one after the image is public:

```bash
RUNPOD_API_KEY=<key> runpodctl template create \
  --name nanoflow-runpod-cu124 \
  --image ghcr.io/tqhdesilva/nanoflow:runpod-cu124 \
  --container-disk-in-gb 20 \
  --docker-entrypoint python \
  --docker-start-cmd scripts/runpod_cuda_smoke.py
```

If you create a Pod from this template, stop or delete it manually after the smoke. For cost safety, prefer the SkyPilot managed job above.

Check status:

```bash
PATH="$HOME/.local/skypilot-shims:$PATH" sky jobs queue
PATH="$HOME/.local/skypilot-shims:$PATH" sky jobs logs nf-docker-cuda-smoke
```

## What Bazel is doing in the RunPod doc

Bazel is a build system. With `rules_oci`, it can assemble and push OCI container images without a Docker daemon. That matters on RunPod Pods because plain Docker builds and Docker Compose are not available by default.

For this project, Bazel helps only if we need to build the image from inside a RunPod Pod. It is not automatically simpler for Python dependency images because `rules_oci` does not interpret Dockerfile `RUN` steps. A normal Dockerfile build from a local machine or CI is simpler. If local Docker is unavailable and we must build on RunPod, use the RunPod Bazel pattern or a CI builder to push the image, then run the managed smoke job above.
