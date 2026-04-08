# NanoFlow

Minimal from-scratch flow matching. Modular, nanochat-style.

## Running

```bash
uv run python train.py                                                  # 2D moons (default)
uv run python train.py dataset=fashion model=unet_fashion device=mps   # FashionMNIST
uv run python train.py dataset=cifar10 model=unet_cifar device=mps     # CIFAR-10
```

Hydra CLI overrides:
```bash
uv run python train.py dataset=cifar10 model=unet_cifar device=cuda training.epochs=200 save=true
uv run python train.py dataset=cifar10 model=unet_cifar training.resume=checkpoints/run_latest.pt
```

Remote (SSH to single-GPU node):
```bash
ssh user@host "cd nanoflow && uv run python train.py dataset=cifar10 model=unet_cifar device=cuda"
```

## Structure

```
datasets.py      # moons_dataset(), fashion_dataset(), cifar_dataset()
models.py        # SinusoidalEmbedding, MLP, ResBlock, UNet (generalized depth)
flow.py          # interpolate(), target_velocity()
trainer.py       # train(), sample(), checkpointing, SIGTERM handler
train.py         # Hydra entry point
viz.py           # plot_loss(), plot_samples(), plot_image_samples()
configs/
  config.yaml           # defaults list (dataset=moons, model=mlp, training=default)
  dataset/{moons,fashion,cifar10}.yaml
  model/{mlp,unet_fashion,unet_cifar}.yaml
  training/default.yaml
```

## Conventions

- Data scaled to [-1, 1]. Noise is N(0, I). Time t ∈ [0, 1] where t=0 is noise, t=1 is data.
- UNet uses GroupNorm (not BatchNorm) — BN misbehaves with varying noise levels across a batch.
- UNet attention is optional (`use_attn`); recommended for ≥ 32×32 images.
- Use `uv run python` to run (not bare `python`).

## CIFAR-10 training notes

Tested config: `batch_size=256`, `lr=1e-3`, 100 epochs, `unet_cifar` (5.8M params, base_ch=64, depth=3, use_attn=true). Trains in ~17 min on a single 4080 GPU (~11s/epoch). Produces recognizable but blurry 32×32 samples at 100 Euler steps.

```bash
uv run python train.py dataset=cifar10 model=unet_cifar device=cuda training.batch_size=256 save=true
```

## Checkpointing

Checkpoints saved every `training.save_every` epochs to `checkpoints/{run_name}_latest.pt`.
SIGTERM (RunPod preemption) saves `{run_name}_preempted.pt` before exit.
Resume with `training.resume=checkpoints/run_latest.pt`.

## Multi-GPU (future)

torchrun is the launcher; add ~15 lines to `trainer.py` for DDP when needed:
```bash
torchrun --nproc_per_node=N train.py dataset=cifar10 model=unet_cifar device=cuda
```
`hydra.job.chdir: false` in `configs/config.yaml` prevents per-rank directory conflicts.
