# NanoFlow

Minimal from-scratch flow matching. Modular, nanoGPT-style.

## Running

```bash
uv run python main.py                                    # 2D moons (default, CPU)
uv run python main.py device=mps                         # moons on MPS
uv run python main.py experiment=fashion device=mps      # FashionMNIST
uv run python main.py experiment=cifar10 device=cuda      # CIFAR-10
```

Override any config field via CLI:
```bash
uv run python main.py experiment=cifar10 device=cuda training.epochs=200 training.batch_size=256 save=true
uv run python main.py training.resume=runs/moons_20260413_120000/checkpoints/latest.pt
```

Multi-GPU:
```bash
torchrun --nproc_per_node=N main.py experiment=cifar10 device=cuda
```

## Structure

```
main.py           # Hydra entry point — training + inference
inference.py      # Standalone inference — load from run_dir or experiment+checkpoint
config.py         # Structured config dataclasses (schema validation)
unit.py           # FlowMatchingUnit(AutoUnit), SamplingUnit(AutoPredictUnit), euler_sample
datasets.py       # MoonsDataset, FashionMNISTDataset, CifarDataset (classes with num_classes)
models.py         # SinusoidalEmbedding, MLP, ResBlock, UNet
flow.py           # NoisePath base, CondOT
viz.py            # plot_samples(), plot_image_samples()
configs/
  config.yaml                        # top-level defaults + device, save
  experiment/{moons,fashion,cifar10}  # bundles dataset + model + inference
  dataset/{moons,fashion,cifar10}     # dataset-specific params
  model/{mlp,unet_fashion,unet_cifar} # model architecture params
  inference/{moons,fashion,cifar10}   # sampling params (n_samples, num_steps, image_shape)
  training/default.yaml               # epochs, lr, batch_size, etc.
  flow/condot.yaml                    # flow path selection
```

## Config system

Before any run, sanity-check the fully materialized config with:
```bash
uv run python main.py experiment=cifar10 device=cuda --cfg job
```
Replace the experiment/overrides to match your intended command. Hydra resolves all interpolations and prints the final values.

Hydra experiment configs are the primary interface. `experiment=X` sets dataset, model, and inference together:

| Experiment | Dataset | Model | n_samples | num_workers |
|------------|---------|-------|-----------|-------------|
| `moons` (default) | 2D moons | MLP | 1000 | 0 |
| `fashion` | FashionMNIST | UNet (small) | 64 | 4 |
| `cifar10` | CIFAR-10 | UNet (large) | 64 | 4 |

Structured config dataclasses in `config.py` validate all YAML configs at startup — wrong types or extra keys error immediately.

`save=false` (default): no plots generated. `save=true`: samples are generated and saved as PNG.

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

```bash
# From run dir (auto-finds checkpoint + config):
uv run python inference.py run_dir=runs/moons_20260413_120000 save=true

# Manual (experiment + checkpoint):
uv run python inference.py experiment=moons inference.checkpoint=runs/.../checkpoints/latest.pt save=true

# No checkpoint (random weights, warns):
uv run python inference.py experiment=moons save=true
```

## TensorBoard

```bash
uv run tensorboard --logdir runs/
```

Logs: train/loss, train/lr, train/grad_norm, val/loss, timing/*, samples/generated.
