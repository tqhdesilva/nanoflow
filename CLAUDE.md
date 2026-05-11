# NanoFlow

Minimal from-scratch flow matching. Modular, nanoGPT-style.

## Running

```bash
uv run python train.py                                    # 2D moons (default, CPU)
uv run python train.py device=mps                         # moons on MPS
uv run python train.py experiment=fashion device=mps      # FashionMNIST
uv run python train.py experiment=cifar10 device=cuda     # CIFAR-10
```

Override any config field via CLI:
```bash
uv run python train.py experiment=cifar10 device=cuda training.epochs=200 training.batch_size=256
uv run python train.py training.resume=runs/moons_20260413_120000/checkpoints/latest.pt
```

Multi-GPU:
```bash
torchrun --nproc_per_node=N train.py experiment=cifar10 device=cuda distributed=ddp
```

## Structure

```
train.py          # Hydra entry — Trainer class + training loop + post-train inference
inference.py      # Hydra entry — FlowSampler class for standalone sampling from a checkpoint
config.py         # Structured config dataclasses (schema validation)
callbacks.py      # RunDir, Checkpoint, EpochSummary, StepLoss, LRMonitor, SampleLogger
datasets.py       # MoonsDataset, FashionMNISTDataset, CifarDataset + build_dataloader
models.py         # SinusoidalEmbedding, MLP, ResBlock, UNet (+ class-conditioned variants)
flow.py           # NoisePath base, CondOT
viz.py            # plot_samples(), plot_image_samples()
configs/
  config.yaml                            # top-level (device, distributed, runs_dir, _target_ wiring)
  experiment/{moons,fashion,cifar10}[_cfg]  # bundles dataset + model + inference (+ CFG variants)
  dataset/{moons,fashion,cifar10}        # dataset-specific params
  model/{mlp,unet_fashion,unet_cifar,classcond_*}  # model architecture params
  training/default.yaml                  # epochs, lr, batch_size, etc.
  flow/condot.yaml                       # flow path selection
  metrics/{none,fid_cifar10}.yaml        # post-train metrics to compute
```

## Config system

Before any run, sanity-check the fully materialized config with:
```bash
uv run python train.py experiment=cifar10 device=cuda --cfg job
```
Replace the experiment/overrides to match your intended command. Hydra resolves all interpolations and prints the final values.

Hydra experiment configs are the primary interface. `experiment=X` sets dataset, model, and inference together:

| Experiment | Dataset | Model | n_samples | num_workers |
|------------|---------|-------|-----------|-------------|
| `moons` (default) | 2D moons | MLP | 1000 | 0 |
| `fashion` | FashionMNIST | UNet (small) | 64 | 4 |
| `cifar10` | CIFAR-10 | UNet (large) | 64 | 4 |

Structured config dataclasses in `config.py` validate all YAML configs at startup — wrong types or extra keys error immediately.

Sample PNGs are written when `inference.save_path` is set (each experiment YAML sets one by default).

## Conventions

- Data scaled to [-1, 1]. Noise is N(0, I). Time t ∈ [0, 1] where t=0 is noise, t=1 is data.
- UNet uses GroupNorm (not BatchNorm).
- Use `uv run python` to run (not bare `python`).

## Run layout

All run artifacts live under `runs_dir` (default: `runs`). Each run gets `{run_prefix}_{timestamp}/`:
```
runs/{prefix}_{timestamp}/
  checkpoints/          # latest.pt, preempted.pt
  tensorboard/          # TensorBoard event files
  metadata.yaml         # resolved config + git info (commit, branch, dirty, diff)
```

Checkpoints saved every `training.save_every` epochs. SIGTERM saves `preempted.pt`.
Resume: `training.resume=runs/.../checkpoints/latest.pt`.

## Inference

`train.py` runs post-training inference automatically when `cfg.inference` is set (every experiment sets it). For standalone sampling from a saved checkpoint:

```bash
# Sample from a checkpoint:
uv run python inference.py experiment=moons \
    inference.sampler.checkpoint=runs/.../checkpoints/latest.pt

# No checkpoint (random weights, just to smoke-test the path):
uv run python inference.py experiment=moons
```

## TensorBoard

```bash
uv run tensorboard --logdir runs/
```

Logs: train/loss, train/epoch_loss, train/lr, val/loss, timing/*, samples/generated.
