# NanoFlow

Minimal from-scratch flow matching, in the spirit of nanoGPT. Pure PyTorch, single-file.

## Quickstart

```bash
pip install -r requirements.txt
python flow_matching.py
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
- **Broadcasting:** t is `(B,)`, data is `(B,2)` — need `t.unsqueeze(-1)` for interpolation
- **Sampling is deterministic:** Euler ODE has no stochastic term (unlike DDPM reverse steps)

## Log

_Fill in during implementation._
