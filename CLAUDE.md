# NanoFlow

Minimal from-scratch flow matching. Single file (`flow_matching.py`), nanoGPT-style.

## Running

```bash
uv run python flow_matching.py --dataset moons --save       # 2D toy
uv run python flow_matching.py --dataset fashion --device mps --save  # FashionMNIST
```

## Structure

Everything lives in `flow_matching.py`, organized by section comments (A–G):
- **A** — Datasets: `moons_dataset()`, `fashion_dataset()`
- **B** — `SinusoidalEmbedding` (shared by MLP and UNet)
- **C** — Models: `MLP` (2D), `ResBlock` + `UNet` (images)
- **D** — Flow matching core: `interpolate()`, `target_velocity()` (shape-agnostic)
- **E** — `train()` loop (works for both datasets)
- **F** — `sample()` via Euler ODE integration
- **G** — Visualization + `__main__`

## Conventions

- Data scaled to [-1, 1]. Noise is N(0, I). Time t ∈ [0, 1] where t=0 is noise, t=1 is data.
- UNet uses GroupNorm (not BatchNorm) — BN misbehaves with varying noise levels across a batch.
- No attention in UNet — unnecessary at 28×28.
- Use `uv run python` to run (not bare `python`).
