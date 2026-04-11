# NanoFlow

Minimal from-scratch flow matching. Modular, nanochat-style.

## Running

```bash
uv run python train.py                                                  # 2D moons (default, CPU)
uv run python train.py device=mps                                       # 2D moons on MPS
uv run python train.py dataset=fashion model=unet_fashion device=mps    # FashionMNIST
uv run python train.py dataset=cifar10 model=unet_cifar device=mps      # CIFAR-10
```

Hydra CLI overrides:
```bash
uv run python train.py dataset=cifar10 model=unet_cifar device=cuda training.epochs=200 save=true
uv run python train.py dataset=cifar10 model=unet_cifar training.resume=checkpoints/run_latest.pt
```

Multi-GPU (via torchrun):
```bash
torchrun --nproc_per_node=N train.py dataset=cifar10 model=unet_cifar device=cuda
```

RunPod (via scripts/run.sh, saves to network volume):
```bash
./scripts/run.sh dataset=cifar10 model=unet_cifar device=cuda
NPROC=4 ./scripts/run.sh dataset=cifar10 model=unet_cifar device=cuda
```

## Structure

```
datasets.py      # moons_dataset(), fashion_dataset(), cifar_dataset() — return Dataset objects
models.py        # SinusoidalEmbedding, MLP, ResBlock, UNet (generalized depth)
flow.py          # NoisePath base, CondOT (interpolation, target, sample_step)
unit.py          # FlowMatchingUnit(AutoUnit) — training, DDP, DataLoaders, TensorBoard, EMA, checkpointing
train.py         # Hydra entry point, post-training sampling + plots
viz.py           # plot_loss(), plot_samples(), plot_image_samples()
Dockerfile       # NGC PyTorch base + uv
scripts/run.sh   # RunPod entrypoint (network volume artifacts)
configs/
  config.yaml           # defaults list (dataset=moons, model=mlp, training=default, flow=condot)
  dataset/{moons,fashion,cifar10}.yaml
  model/{mlp,unet_fashion,unet_cifar}.yaml
  training/default.yaml
  flow/condot.yaml
```

## Conventions

- Data scaled to [-1, 1]. Noise is N(0, I). Time t ∈ [0, 1] where t=0 is noise, t=1 is data.
- UNet uses GroupNorm (not BatchNorm) — BN misbehaves with varying noise levels across a batch.
- UNet attention is optional (`use_attn`); recommended for ≥ 32×32 images.
- Use `uv run python` to run (not bare `python`).
- Training uses TorchTNT AutoUnit. Noise path is swappable via `flow.py` (CondOT default).

## Training config

Key fields in `configs/training/default.yaml`:

| Field | Default | Description |
|-------|---------|-------------|
| `epochs` | 100 | Training epochs |
| `batch_size` | 128 | Batch size |
| `lr` | 1e-3 | Learning rate |
| `grad_clip` | 1.0 | Max gradient norm |
| `ema_decay` | 0.9999 | EMA weight decay (0 = disabled) |
| `precision` | null | fp32/fp16/bf16 (null = fp32) |
| `log_dir` | runs/{run_name} | TensorBoard log directory |
| `log_every` | 50 | TensorBoard scalar logging interval (steps) |
| `save_every` | 10 | Checkpoint + validation + sample interval (epochs) |
| `num_workers` | 4 | DataLoader workers |
| `warmup_epochs` | 0 | Linear LR warmup epochs |

## CIFAR-10 training notes

Tested config: `batch_size=256`, `lr=1e-3`, 100 epochs, `unet_cifar` (5.8M params, base_ch=64, depth=3, use_attn=true). Trains in ~17 min on a single 4080 GPU (~11s/epoch). Produces recognizable but blurry 32×32 samples at 100 Euler steps.

```bash
uv run python train.py dataset=cifar10 model=unet_cifar device=cuda training.batch_size=256 save=true
```

## Checkpointing

Checkpoints saved every `training.save_every` epochs to `checkpoints/{run_name}_latest.pt`.
SIGTERM (RunPod preemption) saves `{run_name}_preempted.pt` before exit.
Resume with `training.resume=checkpoints/run_latest.pt`.
Checkpoints include: model, optimizer, epoch, step, losses, EMA weights, scheduler state.

## Multi-GPU

DDP is supported via `torchrun`. Logging, checkpointing, and sampling are rank-0 only.
`DistributedSampler` is auto-applied when launched with torchrun.
`hydra.job.chdir: false` in `configs/config.yaml` prevents per-rank directory conflicts.

## TensorBoard

```bash
uv run tensorboard --logdir runs/
```

Logs: train/loss, train/lr, train/grad_norm, train/epoch_loss, val/loss, timing/*, samples/generated.

## Docker (RunPod)

```bash
docker build -t nanoflow .
docker run nanoflow uv run python train.py training.epochs=2  # smoke test
```
