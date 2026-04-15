# NanoFlow

Minimal from-scratch flow matching, in the spirit of nanoGPT. Pure PyTorch.

## Quickstart

```bash
# 2D moons (default — fast, good for debugging)
uv run python main.py device=mps

# FashionMNIST
uv run python main.py experiment=fashion device=mps

# CIFAR-10
uv run python main.py experiment=cifar10 device=cuda
```

Each experiment bundles the right dataset, model, and inference config. Override anything via CLI:

```bash
uv run python main.py experiment=cifar10 device=cuda training.epochs=200 training.batch_size=256 save=true
```

## Experiments

| Experiment | Dataset | Model | Params | Notes |
|------------|---------|-------|--------|-------|
| `moons` (default) | 2D moons | MLP | ~71K | ~10s on MPS |
| `fashion` | FashionMNIST 28x28 | UNet (depth=2) | ~347K | ~10 min on MPS |
| `cifar10` | CIFAR-10 32x32 | UNet (depth=3, attn) | ~5.8M | ~17 min on RTX 4080 (16GB) |

## Config system

Configs use [Hydra](https://hydra.cc/) with structured config validation. The experiment config is the main knob — it sets dataset, model, and inference defaults together.

Sanity-check the fully materialized config before a run:
```bash
uv run python main.py experiment=cifar10 device=cuda --cfg job
```

```
configs/
  config.yaml                          # top-level (device, save)
  experiment/{moons,fashion,cifar10}   # bundles dataset + model + inference
  training/default.yaml                # epochs, lr, batch_size, etc.
```

Key top-level overrides:
- `device={cpu,mps,cuda}` — default: cpu
- `save={true,false}` — save sample plots (default: false)
- `training.epochs=N`, `training.lr=X`, `training.batch_size=N`
- `inference.num_steps=N` — Euler integration steps (default: 100)

## TensorBoard

```bash
uv run tensorboard --logdir runs/
```

Each run logs to `runs/{experiment}_{timestamp}/`.

## Checkpointing

Checkpoints save every `training.save_every` epochs. Resume with:

```bash
uv run python main.py training.resume=checkpoints/{run_name}_latest.pt
```

SIGTERM (e.g. RunPod preemption) triggers a checkpoint save before exit.

## Flow matching in brief

| Concept | Detail |
|---------|--------|
| Interpolation | `x_t = (1-t)*noise + t*data` |
| Target | Predict velocity `v = data - noise` |
| Time | `t ∈ [0,1]`, continuous (t=0 is noise, t=1 is data) |
| Sampling | Deterministic Euler ODE: `x += v*dt` |
| Loss | MSE on predicted vs target velocity |

## DDPM vs Flow Matching

| DDPM | Flow Matching |
|------|---------------|
| `x_t = sqrt(a_t)*x_0 + sqrt(1-a_t)*eps` | `x_t = (1-t)*eps + t*x_0` |
| Predict noise `eps` | Predict velocity `v = x_0 - eps` |
| Integer timesteps, beta schedule | Continuous `t ∈ [0,1]`, no schedule |
| Stochastic reverse chain | Deterministic Euler ODE |
