"""Standalone inference: load a checkpoint, run Euler integration, save plots/metrics.

Also re-used by `train.py` for post-train sampling. See `run_inference`.
"""

import os
from pathlib import Path
from typing import Optional

import hydra
import torch
import yaml
from omegaconf import OmegaConf
from torch import Tensor

import config as _config  # noqa: F401, registers structured config schema
from callbacks import make_run_dir


@torch.no_grad()
def euler_sample(model, noise, num_steps):
    """Generate samples via Euler integration of the learned velocity field."""
    xt = noise
    dt = 1.0 / num_steps
    for t_val in torch.linspace(0, 1, num_steps, device=noise.device):
        vt = model(xt, t_val.expand(xt.size(0)))
        xt = xt + vt * dt
    return xt


@torch.no_grad()
def guided_euler_sample(model, noise, num_steps, cond, guidance_scale):
    """Euler sampling with classifier-free guidance."""
    null_cond = torch.full_like(cond, model.null_token)
    xt = noise
    dt = 1.0 / num_steps
    for t_val in torch.linspace(0, 1, num_steps, device=noise.device):
        t_batch = t_val.expand(xt.size(0))
        v_cond = model(xt, t_batch, cond)
        v_uncond = model(xt, t_batch, null_cond)
        vt = v_uncond + guidance_scale * (v_cond - v_uncond)
        xt = xt + vt * dt
    return xt


class FlowSampler:
    """Flow-matching trajectory sampler. Distinct from torch.utils.data.Sampler."""

    def __init__(
        self,
        model: torch.nn.Module,
        num_steps: int = 100,
        latent_shape: Optional[list] = None,
        device: str = "cpu",
        checkpoint: Optional[str] = None,
    ):
        self.device = torch.device(device)
        self.module = model.to(self.device)
        if checkpoint:
            ckpt = torch.load(checkpoint, weights_only=True, map_location=self.device)
            state = ckpt.get("ema_state") or ckpt["model_state"]
            self.module.load_state_dict(state)
            tag = "EMA" if ckpt.get("ema_state") else "model"
            epoch = ckpt.get("train_progress", {}).get("num_epochs_completed", "?")
            print(f"Loaded {tag} weights from {checkpoint} (epoch {epoch})")
        self.module.eval()
        self.num_steps = num_steps
        self.latent_shape = latent_shape

    def sample_labels(self, n_samples: int, class_sampler=None) -> Optional[Tensor]:
        if class_sampler is not None:
            if class_sampler.probs is not None:
                probs = torch.tensor(class_sampler.probs)
                labels = torch.multinomial(probs, n_samples, replacement=True)
            else:
                labels = torch.arange(n_samples) % class_sampler.num_classes
            return labels.to(self.device)
        return None

    @torch.no_grad()
    def generate(
        self,
        n_samples: int,
        class_sampler=None,
        labels: Optional[Tensor] = None,
    ) -> Tensor:
        shape = tuple(self.latent_shape) if self.latent_shape else (2,)
        noise = torch.randn(n_samples, *shape, device=self.device)
        if class_sampler is not None:
            labels = labels if labels is not None else self.sample_labels(
                n_samples, class_sampler
            )
            return guided_euler_sample(
                self.module,
                noise,
                self.num_steps,
                labels,
                class_sampler.guidance_scale,
            )
        return euler_sample(self.module, noise, self.num_steps)


def run_inference(
    cfg,
    sampler: FlowSampler,
    run_dir: Optional[Path] = None,
    train_data=None,
) -> Tensor:
    """Run sampling, save plots, compute metrics. Shared by CLI + post-train path."""
    from viz import plot_image_samples, plot_samples

    icfg = cfg.inference
    cs = OmegaConf.select(icfg, "class_sampler", default=None)
    labels = sampler.sample_labels(icfg.n_samples, class_sampler=cs)
    samples = sampler.generate(icfg.n_samples, class_sampler=cs, labels=labels).cpu()
    labels_cpu = labels.cpu() if labels is not None else None
    class_names = list(cs.class_names) if cs is not None and cs.class_names else None

    metrics_cfg = OmegaConf.select(icfg, "metrics", default=None)
    if metrics_cfg:
        if run_dir is None:
            run_dir = make_run_dir(cfg.runs_dir, cfg.training.run_prefix)
            print(f"Run dir: {run_dir}")
        results = []
        for m_cfg in metrics_cfg:
            metric = hydra.utils.instantiate(m_cfg)
            results.append(metric(samples))
        metrics_path = run_dir / "metrics.yaml"
        with open(metrics_path, "w") as f:
            yaml.dump(results, f, sort_keys=False)
        print(f"Wrote metrics → {metrics_path}", flush=True)

    if icfg.save_path:
        title = f"{sampler.num_steps} Euler steps, {icfg.n_samples} samples"
        if sampler.latent_shape is None:
            real = train_data.data if train_data is not None else None
            if real is not None:
                plot_samples(real, samples, title, icfg.save_path)
            else:
                print(
                    f"Generated {samples.shape[0]} samples"
                    " (no real data for comparison)"
                )
        else:
            plot_image_samples(
                samples,
                title,
                icfg.save_path,
                labels=labels_cpu,
                class_names=class_names,
            )
    else:
        print(
            f"Generated {samples.shape[0]} samples (set save_path to write plot)"
        )
    return samples


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg) -> None:
    if int(os.environ.get("RANK", 0)) != 0:
        return
    if cfg.inference is None:
        raise ValueError("inference config is required for standalone inference")
    sampler = hydra.utils.instantiate(cfg.inference.sampler)
    run_inference(cfg, sampler)


if __name__ == "__main__":
    main()
