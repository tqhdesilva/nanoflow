# NanoFlow

Minimal from-scratch flow matching. Modular, nanoGPT-style.

## Running

```bash
uv run python train.py                                    # 2D moons (default, CPU)
uv run python train.py device=mps                         # moons on MPS
uv run python train.py experiment=fashion device=mps      # FashionMNIST
uv run python train.py experiment=cifar10 device=cuda      # CIFAR-10
```

Override any config field via CLI:
```bash
uv run python train.py experiment=cifar10 device=cuda training.epochs=200 training.batch_size=256 save=true
uv run python train.py training.resume=checkpoints/moons_20260411_153000_latest.pt
```

Multi-GPU:
```bash
torchrun --nproc_per_node=N train.py experiment=cifar10 device=cuda
```

## Structure

```
train.py          # Hydra entry point, post-training sampling
config.py         # Structured config dataclasses (schema validation)
unit.py           # FlowMatchingUnit(AutoUnit) — training loop, DDP, TensorBoard, EMA
datasets.py       # moons_dataset(), fashion_dataset(), cifar_dataset()
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

## Checkpointing

Checkpoints saved every `training.save_every` epochs to `checkpoints/{run_name}_latest.pt`.
SIGTERM saves `{run_name}_preempted.pt` before exit.
Resume: `training.resume=checkpoints/{run_name}_latest.pt`.

## TensorBoard

```bash
uv run tensorboard --logdir runs/
```

Run dirs are auto-named `{experiment}_{timestamp}` (e.g. `runs/moons_20260411_153000/`).
Logs: train/loss, train/lr, train/grad_norm, val/loss, timing/*, samples/generated.
