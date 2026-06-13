# RunPod ImageNet-256 DiT SkyPilot chain

This runbook covers launching the ImageNet-256 latent DiT chain on RunPod with SkyPilot managed jobs. The YAML defines all job stages and commands. The launcher only fills runtime plumbing, applies the RunPod network volume, then submits the rendered chain.

## Prerequisites

Install and configure:

- SkyPilot with RunPod support.
- `runpodctl` for RunPod availability and cleanup checks.
- `RUNPOD_API_KEY` in the environment for `--infra auto` and RunPod CLI checks.
- GCS service account JSON at `~/.config/gcp/nanoflow-gcs-reader.json`, or pass `--gcp-credentials`.
- A Docker image with NanoFlow dependencies, for example `docker:ghcr.io/tqhdesilva/nanoflow:runpod-cu124`.

Check RunPod access:

```bash
PATH="$HOME/.local/skypilot-shims:$PATH" sky check runpod
RUNPOD_API_KEY=<runpod-api-key> runpodctl datacenter list -o json >/tmp/runpod-datacenters.json
```

## SkyPilot managed jobs setup

Use local consolidation mode for RunPod managed jobs. Do not run the managed jobs controller on RunPod. A RunPod controller pod may not be able to reach worker pods through public SSH endpoints, while local consolidation lets the laptop connect to each worker.

When SkyPilot is installed with `uv tool`, consolidation mode needs a `python` executable on `PATH` that imports the same SkyPilot package as the active `sky` command. Create a wrapper script. Do not use a symlink.

```bash
mkdir -p ~/.local/skypilot-shims
cat > ~/.local/skypilot-shims/python <<'EOF'
#!/usr/bin/env bash
exec "$HOME/.local/share/uv/tools/skypilot/bin/python" "$@"
EOF
chmod +x ~/.local/skypilot-shims/python
PATH="$HOME/.local/skypilot-shims:$PATH" python -c 'import sky; print(sky.__version__)'
```

Set `~/.sky/config.yaml`:

```yaml
jobs:
  controller:
    consolidation_mode: true
    resources:
      cloud: runpod
      cpus: 1+
      memory: 4+
      disk_size: 5
```

Restart the local SkyPilot API server after editing the config:

```bash
PATH="$HOME/.local/skypilot-shims:$PATH" sky api stop
PATH="$HOME/.local/skypilot-shims:$PATH" sky api start
```

Use the shim path for every SkyPilot command in this runbook:

```bash
export PATH="$HOME/.local/skypilot-shims:$PATH"
```

## Region and volume selection

The chain uses a RunPod network volume mounted at `/workspace`. The volume and every worker pod must be in the same data center.

The recommended path is `--infra auto`. The launcher queries RunPod GraphQL, keeps data centers with network volume support, filters for the requested GPU stock, skips countries unsupported by SkyPilot, then emits a concrete infra string such as `runpod/US/US-CA-2`.

```bash
export RUNPOD_API_KEY=<runpod-api-key>
```

Optional preference:

```bash
export RUNPOD_PREFERRED_DATACENTER=EU-NL-1
```

Manual availability checks:

```bash
RUNPOD_API_KEY=<runpod-api-key> scripts/runpod_list_h100_datacenters.sh
REQUIRE_STORAGE_SUPPORT=1 GPU_QUERY="H100" RUNPOD_API_KEY=<runpod-api-key> \
  scripts/runpod_list_h100_datacenters.sh
```

Manual infra validation:

```bash
PATH="$HOME/.local/skypilot-shims:$PATH" \
  sky launch --dryrun --infra runpod/US/US-CA-2 --gpus H100-SXM:1 'echo ok'
```

## Confirm regional capacity

There is no known no-create API that confirms RunPod CPU availability in a specific data center. RunPod data center queries expose GPU stock and network volume support, but not CPU stock. SkyPilot dry runs only check YAML validity, volume references, infra syntax, and catalog feasibility. They do not check live scheduling or reserve capacity.

Use this decision rule:

- If CPU-only tasks are required, the first CPU task in the real managed chain is the CPU capacity check. Watch it closely and cancel within a few minutes if it remains stuck.
- If pre-launch CPU probes are not acceptable, do not depend on CPU-only RunPod tasks. Use GPU-backed sync stages or merge sync into GPU stages so region selection only depends on GPU stock and volume support.

Candidate selection still starts with GPU plus volume support:

```bash
REQUIRE_STORAGE_SUPPORT=1 GPU_QUERY="H100" RUNPOD_API_KEY=<runpod-api-key> \
  scripts/runpod_list_h100_datacenters.sh
```

Then apply or reuse the candidate volume in that exact data center:

```bash
PATH="$HOME/.local/skypilot-shims:$PATH" \
  sky volumes apply --infra runpod/US/US-CA-2 \
    --name nf-imagenet256-usca2 \
    --type runpod-network-volume \
    --size 100Gi \
    -y
```

Render the chain for config validation:

```bash
tmpdir=$(mktemp -d)
scripts/sky_runpod_chain.py cloud/runpod/imagenet256-dit-smoke-chain.yaml \
  --dry-run \
  --chain-id schedule-check \
  --infra runpod/US/US-CA-2 \
  --gpu-request H100-SXM:1 \
  --volume-name nf-imagenet256-usca2 \
  --volume-size 100Gi \
  --output "$tmpdir/chain.yaml" \
  --gcp-credentials ~/.config/gcp/nanoflow-gcs-reader.json
```

SkyPilot cannot dry-run a multi-document managed DAG with `sky launch`. If you want per-task config validation, split representative tasks and run dry runs on each task file:

```bash
uv run python - <<'PY' "$tmpdir/chain.yaml" "$tmpdir"
import sys
from pathlib import Path
import yaml

chain_path = Path(sys.argv[1])
outdir = Path(sys.argv[2])
docs = list(yaml.safe_load_all(chain_path.read_text()))
for doc in docs[1:]:
    if doc["name"] in {"sync_inputs", "masked_pretrain", "sync_artifacts"}:
        (outdir / f"{doc['name']}.yaml").write_text(
            yaml.safe_dump(doc, sort_keys=False)
        )
PY

PATH="$HOME/.local/skypilot-shims:$PATH" sky launch --dryrun "$tmpdir/sync_inputs.yaml"
PATH="$HOME/.local/skypilot-shims:$PATH" sky launch --dryrun "$tmpdir/masked_pretrain.yaml"
PATH="$HOME/.local/skypilot-shims:$PATH" sky launch --dryrun "$tmpdir/sync_artifacts.yaml"
```

These dry runs are useful for catching bad YAML, bad volume references, unsupported regions, and unsupported GPU SKUs. They are not capacity checks.

For a workflow that avoids unknown CPU capacity, make the sync stages GPU-backed in a copied YAML:

```yaml
resources:
  infra: ${RUNPOD_INFRA}
  accelerators: ${GPU_REQUEST}
  disk_size: 100
  image_id: ${IMAGE_ID}
```

Use that resource block for `sync_inputs` and `sync_artifacts` instead of `cpus`, `memory`, and the CPU image. This costs more for sync stages, but it removes the separate CPU capacity dependency. The real confirmation then becomes the first GPU task starting successfully in the selected region.

During launch, inspect status and pods:

```bash
PATH="$HOME/.local/skypilot-shims:$PATH" sky jobs queue --refresh
RUNPOD_API_KEY=<runpod-api-key> runpodctl pod list --all --since 2h -o json
```

If a task remains `STARTING` for several minutes, treat the region as not currently usable for that task shape and cancel the job.

## Dry run

Dry run renders the chain and prints the volume and launch commands. It does not write a secret file, apply the volume, or launch a job.

```bash
scripts/sky_runpod_chain.py cloud/runpod/imagenet256-dit-smoke-chain.yaml \
  --dry-run \
  --chain-id imagenet256-dit-smoke-test \
  --infra auto \
  --gpu-request H100-SXM:1 \
  --volume-name nf-imagenet256 \
  --volume-size 100Gi \
  --image-id docker:ghcr.io/tqhdesilva/nanoflow:runpod-cu124 \
  --artifact-cloud-uri gs://nanoflow/runs \
  --gcp-credentials ~/.config/gcp/nanoflow-gcs-reader.json
```

For dry run only, `--infra auto` can render without `RUNPOD_API_KEY`; it falls back to `runpod/NL/EU-NL-1`.

## Launch

Omit `--dry-run` to launch. This applies the network volume and submits the managed chain.

```bash
PATH="$HOME/.local/skypilot-shims:$PATH" \
  scripts/sky_runpod_chain.py cloud/runpod/imagenet256-dit-smoke-chain.yaml \
    --infra auto \
    --gpu-request H100-SXM:1 \
    --volume-name nf-imagenet256 \
    --volume-size 100Gi \
    --image-id docker:ghcr.io/tqhdesilva/nanoflow:runpod-cu124 \
    --artifact-cloud-uri gs://nanoflow/runs \
    --gcp-credentials ~/.config/gcp/nanoflow-gcs-reader.json
```

For a manually chosen data center, pass the full infra string:

```bash
PATH="$HOME/.local/skypilot-shims:$PATH" \
  scripts/sky_runpod_chain.py cloud/runpod/imagenet256-dit-smoke-chain.yaml \
    --infra runpod/US/US-CA-2 \
    --gpu-request H100-SXM:1 \
    --volume-name nf-imagenet256-usca2 \
    --volume-size 100Gi \
    --image-id docker:ghcr.io/tqhdesilva/nanoflow:runpod-cu124 \
    --artifact-cloud-uri gs://nanoflow/runs \
    --gcp-credentials ~/.config/gcp/nanoflow-gcs-reader.json
```

## Launcher behavior, provenance, and overrides

The launcher does four things:

1. Resolve runtime values such as chain id, infra, GPU SKU, image, volume, and cloud URIs.
2. Render the chain template to `.nanoflow_runpod_chains/<chain_id>.yaml`, or to `--output` when supplied.
3. Apply the RunPod network volume.
4. Submit the rendered multi-document YAML with `sky jobs launch`.

The template file only needs to exist locally when the launcher runs. It does not need to be present on the worker for provenance capture. For reproducible team runs, keep the template checked in. If launching from a remote git source rather than local SkyPilot workdir sync, the template and task scripts must be committed at the selected ref.

CLI overrides are captured in the rendered YAML. The launcher injects the rendered YAML as a SkyPilot `file_mounts` source for `sync_inputs`; the task copies it from `/tmp/nanoflow-rendered-chain.yaml` to `/workspace/runs/$CHAIN_ID/chain.yaml`. That exported `chain.yaml` is the exact rendered YAML submitted to SkyPilot, including launcher overrides such as `--infra`, `--gpu-request`, `--volume-name`, `--image-id`, and cloud URIs.

The launcher passes GCS credentials via `sky jobs launch --secret-file`. The secret value is not printed inline. A local `.nanoflow_runpod_chains/<chain_id>.secrets.env` file is written for non-dry launches. Delete it after submission if you want no local secret copy:

```bash
rm -f .nanoflow_runpod_chains/*.secrets.env
```

## Chain stages

`cloud/runpod/imagenet256-dit-smoke-chain.yaml` contains:

1. `sync_inputs`: copy the exact rendered chain YAML into the run directory, sync latent mmap data from GCS if missing, then run mmap preflight.
2. `masked_pretrain`: run tiny masked DiT training with `training.resume=auto`.
3. `unmasked_finetune`: initialize from the masked checkpoint and train unmasked MSE.
4. `eval_generate`: generate a tiny PNG sample set from the finetune checkpoint.
5. `sync_artifacts`: sync the chain directory to GCS.

Smoke success means the cloud output contains:

- `chain.yaml`
- `masked_pretrain/checkpoints/latest.pt`
- `unmasked_finetune/checkpoints/latest.pt`
- `masked_pretrain/metadata.yaml`
- `unmasked_finetune/metadata.yaml`
- `eval_generate/metadata.yaml`
- generated PNG files from `eval_generate`

For longer runs, copy the chain YAML and edit the training or eval commands in YAML. Keep stage logic in YAML rather than adding launcher presets.

## Monitoring

Check status:

```bash
PATH="$HOME/.local/skypilot-shims:$PATH" sky jobs queue --refresh
```

Stream managed job logs:

```bash
PATH="$HOME/.local/skypilot-shims:$PATH" sky jobs logs <job-id>
PATH="$HOME/.local/skypilot-shims:$PATH" sky jobs logs --controller <job-id>
```

Local logs are often faster for diagnosis:

```bash
ls ~/sky_logs/jobs_controller/
find ~/sky_logs/managed_jobs/job-id-<job-id> -type f -name run.log -print
```

If a job stays `PENDING` for several minutes with no `submitted_at`, no controller pid, and no worker cluster, controller acceptance failed before RunPod provisioning. Check:

```bash
sed -n '1,200p' ~/sky_logs/managed_jobs/submit-job-<job-id>.log
```

A `python: command not found` error means the SkyPilot shim path was not active when the job was submitted.

If a task stays `STARTING` too long, inspect RunPod pods and cancel if it is stuck:

```bash
RUNPOD_API_KEY=<runpod-api-key> runpodctl pod list --all --since 2h -o json
PATH="$HOME/.local/skypilot-shims:$PATH" sky jobs cancel <job-id> -y
```

## Artifact verification

Check the artifact sink after `sync_artifacts` succeeds:

```bash
GOOGLE_APPLICATION_CREDENTIALS="$HOME/.config/gcp/nanoflow-gcs-reader.json" \
  gcloud storage ls --recursive gs://nanoflow/runs/<chain_id>
```

Expected smoke outputs are listed in the chain stages section.

## Cleanup checks

Managed jobs should clean up worker pods after each task. Verify no live SkyPilot resources remain:

```bash
PATH="$HOME/.local/skypilot-shims:$PATH" sky status --refresh
PATH="$HOME/.local/skypilot-shims:$PATH" sky jobs queue --refresh
```

Verify RunPod pods:

```bash
RUNPOD_API_KEY=<runpod-api-key> runpodctl pod list --all --since 2h -o json
```

Network volumes are caches. Keep reusable hydrated volumes, or delete disposable volumes after artifacts are in GCS:

```bash
PATH="$HOME/.local/skypilot-shims:$PATH" sky volumes ls --refresh
PATH="$HOME/.local/skypilot-shims:$PATH" sky volumes delete <volume-name> -y
```

## Data preflight

The sync stage runs:

```bash
python scripts/preflight_imagenet_latent_mmap.py \
  --cache-root /workspace/latent-caches/imagenet256/current \
  --batch-size 2 \
  --num-workers 0
```

It validates `metadata.json`, opens train and val mmap arrays, checks `[*, 4, 32, 32]` fp16 latents, checks int64 labels in `[0, 999]`, then loads one train and one val batch.

## Eval pieces

Eval generation, stats creation, and FID calculation are independent commands.

Generate PNGs:

```bash
python eval_imagenet.py \
  experiment=imagenet256_latent_dit_m2_unmasked_finetune \
  model.masker=null \
  device=cuda \
  eval.checkpoint=/workspace/runs/<chain_id>/unmasked_finetune/checkpoints/latest.pt \
  eval.output_dir=/workspace/runs/<chain_id>/eval_generate \
  eval.generation.num_samples=10000 \
  eval.generation.num_steps=200 \
  eval.compute_fid=false
```

Create FID reference stats from real RGB ImageNet validation images:

```bash
python eval_imagenet.py \
  eval.generate=false \
  eval.make_stats=true \
  eval.output_dir=/workspace/runs/<chain_id>/fid_stats \
  eval.stats.real_dir=/workspace/data/imagenet-256/ImageNet/val \
  eval.stats.custom_stats_name=nanoflow_imagenet256_val_real_tf_legacy \
  eval.stats.mode=legacy_tensorflow \
  eval.stats.device=cuda
```

Compute FID from generated PNGs and existing stats:

```bash
python eval_imagenet.py \
  eval.generate=false \
  eval.compute_fid=true \
  eval.output_dir=/workspace/runs/<chain_id>/eval_generate \
  eval.fid.custom_stats_name=nanoflow_imagenet256_val_real_tf_legacy \
  eval.fid.num_samples=10000 \
  eval.fid.device=cuda
```

## Resolved Hydra config

Before editing a serious chain YAML, inspect exact configs locally:

```bash
uv run python train.py experiment=imagenet256_latent_dit_m2_masked --cfg job
uv run python train.py experiment=imagenet256_latent_dit_m2_unmasked_finetune --cfg job
uv run python eval_imagenet.py experiment=imagenet256_latent_dit_m2_masked --cfg job
```

## Notes

- Use exact GPU SKUs such as `H100-SXM:1`.
- CPU sync tasks use `cpus: 1+`, `memory: 4+`, and `disk_size: 5` to avoid RunPod CPU provisioning failures from oversized pod storage or scarce CPU shapes.
- GPU tasks use `disk_size: 100` and route caches plus outputs to `/workspace`.
- The volume is a cache. GCS is the durable source and sink.
- Do not pass managed jobs controller resources through `sky jobs launch --config` for multi-document YAMLs. Put controller settings in `~/.sky/config.yaml`.
