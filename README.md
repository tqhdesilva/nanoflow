# NanoFlow

Minimal from-scratch flow matching, in the spirit of nanoGPT. Pure PyTorch.

## Quickstart

```bash
# 2D moons (default, fast, good for debugging)
uv run python train.py device=mps

# FashionMNIST
uv run python train.py experiment=fashion device=mps

# CIFAR-10
uv run python train.py experiment=cifar10 device=cuda
```

Each experiment bundles the right dataset, model, and inference config. Override anything via CLI:

```bash
uv run python train.py experiment=cifar10 device=cuda training.epochs=200 training.batch_size=256 inference.save_path=cifar10_samples.png
```

Multi-GPU:
```bash
torchrun --nproc_per_node=N train.py experiment=cifar10 device=cuda distributed=ddp
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
uv run python train.py experiment=cifar10 device=cuda --cfg job
```

```
configs/
  config.yaml                          # top-level (device, distributed, runs_dir)
  experiment/{moons,fashion,cifar10}   # bundles dataset + model + inference
  training/default.yaml                # epochs, lr, batch_size, etc.
```

Key overrides:
- `device={cpu,mps,cuda}` — default: cpu
- `training.epochs=N`, `training.lr=X`, `training.batch_size=N`
- `inference.save_path=path.png` — write a sample-grid plot
- `inference.sampler.num_steps=N` — Euler integration steps

## RL fine-tuning (Flow-GRPO)

Fine-tune a CFG checkpoint with [Flow-GRPO](https://github.com/yifan123/flow_grpo) (ODE→SDE conversion, group-relative advantage, clipped IS surrogate, closed-form Gaussian KL to a frozen reference). Single-GPU, plain torch.

```bash
# 1. Train the reward classifier (small CNN, ~1-2 min on MPS)
uv run python -m rl.classifier --epochs 5
# → runs/reward_models/fashion_classifier.pt

# 2. RL fine-tune a CFG-trained Fashion checkpoint
uv run python train_grpo.py experiment=fashion_grpo \
    seed_checkpoint=runs/{fashion_cfg_run}/checkpoints/latest.pt
```

The reward is `log p(target_class | sample)` under the frozen classifier.

Sanity-check the resolved config before a run:
```bash
uv run python train_grpo.py --cfg job
```

Common overrides:
- `rl_training.G=N` — group size (rollout batch = `batch_size * G`)
- `rl_training.num_inner=N` — PPO-style inner loop count
- `rl_training.kl_beta=X`, `rl_training.clip_eps=X`
- `rl_training.T_rollout=N` — SDE steps per rollout (default 10)
- `rl_training.sigma_a=X` — noise schedule scale (paper default 0.7)

Rollout transport is behind a `RolloutClient` Protocol (`rl/rollout_client.py`); the trainer never imports the SDE sampler directly. The only impl shipped is `InProcessRolloutClient`.

## TensorBoard

```bash
uv run tensorboard --logdir runs/
```

Each run logs to `runs/{experiment}_{timestamp}/`.

## Checkpointing

Checkpoints save every `training.save_every` epochs to `runs/{prefix}_{timestamp}/checkpoints/latest.pt`. Resume with:

```bash
uv run python train.py training.resume=runs/{prefix}_{timestamp}/checkpoints/latest.pt
```

SIGTERM (e.g. RunPod preemption) triggers a `preempted.pt` save before exit.

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
