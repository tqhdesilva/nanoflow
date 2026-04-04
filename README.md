# NanoFlow

Minimal from-scratch flow matching, in the spirit of nanoGPT. Pure PyTorch, single-file.

## Quickstart

```bash
# 2D moons (fast, good for debugging)
python flow_matching.py --dataset moons --save

# FashionMNIST (28x28 grayscale)
python flow_matching.py --dataset fashion --device mps --save
```

## DDPM → Flow Matching cheat sheet

| DDPM | Flow Matching |
|------|---------------|
| `x_t = sqrt(a_t)*x_0 + sqrt(1-a_t)*eps` | `x_t = (1-t)*eps + t*x_0` |
| Predict noise `eps` | Predict velocity `v = x_0 - eps` |
| `t` ∈ {0,...,T-1} integers | `t` ∈ [0,1] continuous |
| Reverse chain with variance | Euler ODE: `x += v*dt` |
| Beta schedule, alpha cumprod | Nothing — just lerp |

## Datasets & models

| Dataset | Model | Params | Default epochs | Notes |
|---------|-------|--------|----------------|-------|
| `moons` | MLP (4 residual blocks) | ~50K | 300 | 2D toy data from sklearn |
| `fashion` | UNet (2 levels, 28→14→7) | ~321K | 20 | FashionMNIST, pixel space |

Both share the same flow matching core — linear interpolation path, MSE velocity loss, Euler ODE sampling.

## CLI flags

```
--dataset   {moons,fashion}  default: moons
--epochs    int               default: 300 (moons) / 20 (fashion)
--batch_size int              default: 256 (moons) / 128 (fashion)
--lr        float             default: 1e-3
--num_steps int               Euler integration steps (default: 100)
--n_samples int               default: 1000 (moons) / 64 (fashion)
--device    str               default: cpu
--save                        save plots to PNG instead of showing
```

## DDPM → Flow Matching cheat sheet

| DDPM | Flow Matching |
|------|---------------|
| `x_t = sqrt(a_t)*x_0 + sqrt(1-a_t)*eps` | `x_t = (1-t)*eps + t*x_0` |
| Predict noise `eps` | Predict velocity `v = x_0 - eps` |
| `t` ∈ {0,...,T-1} integers | `t` ∈ [0,1] continuous |
| Reverse chain with variance | Euler ODE: `x += v*dt` |
| Beta schedule, alpha cumprod | Nothing — just lerp |

## Pitfalls

- **Time direction:** t=0 is noise, t=1 is data (opposite from DDPM)
- **Continuous time:** scale t by 1000 before sinusoidal embedding for better frequency coverage
- **Broadcasting:** t shape must match data dims — `(B,1)` for 2D, `(B,1,1,1)` for images
- **Sampling is deterministic:** Euler ODE has no stochastic term (unlike DDPM reverse steps)
- **Don't clamp during integration:** intermediate x_t can exceed [-1,1]; only clamp at visualization
