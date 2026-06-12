# ImageNet-256 eval

`eval_imagenet.py` writes individual PNG samples, label sidecars, sample metadata, Clean-FID stats metadata, and metric YAML files for ImageNet-256 latent runs.

## Generate samples

```bash
uv run python eval_imagenet.py \
  experiment=imagenet256_latent_dit_m2_masked \
  eval.checkpoint=runs/<run_id>/checkpoints/latest.pt \
  eval.output_dir=/workspace/runs/<chain_id>/eval_10k \
  eval.generation.num_samples=10000 \
  eval.generation.batch_size=16 \
  eval.generation.num_steps=200 \
  eval.generation.guidance_scale=2.0 \
  device=cuda
```

For FID-50K, use `eval.generation.num_samples=50000`. The label schedule cycles over 1000 classes, so 50K writes exactly 50 samples per class.

Generation is resumable by default. A rerun skips valid `000000.png` style files when `metadata.yaml` has the same eval config hash. The hash includes the checkpoint SHA256 when the script loads the checkpoint, so overwriting `latest.pt` does not silently mix old and new samples. If the output directory contains files from a different config, the script raises. Use `eval.generation.clean_output_dir=true` only when you want to delete and rebuild the directory.

## Create ImageNet validation stats

Use real RGB ImageNet-256 validation images, not VAE reconstructions, for the primary reference stats.

```bash
uv run python eval_imagenet.py \
  eval.generate=false \
  eval.make_stats=true \
  eval.output_dir=/workspace/runs/<chain_id>/fid_stats \
  eval.stats.real_dir=/workspace/data/imagenet-256/ImageNet/val \
  eval.stats.custom_stats_name=nanoflow_imagenet256_val_real_tf_legacy \
  eval.stats.mode=legacy_tensorflow \
  eval.stats.device=cuda
```

The Clean-FID stats file is stored in Clean-FID's package stats cache. The script records the stats path, SHA256 hash, real image source, mode, feature model, and Clean-FID version in `stats_metadata.yaml`.

## Compute FID

```bash
uv run python eval_imagenet.py \
  eval.generate=false \
  eval.compute_fid=true \
  eval.output_dir=/workspace/runs/<chain_id>/eval_10k \
  eval.fid.custom_stats_name=nanoflow_imagenet256_val_real_tf_legacy \
  eval.fid.mode=legacy_tensorflow \
  eval.fid.device=cuda
```

`metrics.yaml` is written under `eval.output_dir` unless `eval.fid.output_path` is set. `eval.fid.num_samples` is an expected PNG count check. It defaults to `${eval.generation.num_samples}`. For FID-only scoring of an existing 50K directory, set `eval.fid.num_samples=50000`.

## Generate then compute FID

```bash
uv run python eval_imagenet.py \
  experiment=imagenet256_latent_dit_m2_masked \
  eval.checkpoint=runs/<run_id>/checkpoints/latest.pt \
  eval.output_dir=/workspace/runs/<chain_id>/eval_50k \
  eval.generation.num_samples=50000 \
  eval.compute_fid=true \
  device=cuda \
  eval.fid.device=cuda
```

Primary reported FID should use `nanoflow_imagenet256_val_real_tf_legacy` with Clean-FID `mode=legacy_tensorflow`.
