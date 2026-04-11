"""NanoFlow — training entry point."""

import torch
import hydra
from torchtnt.framework import train as tnt_train

import config as _config  # noqa: F401 — registers structured config schema
from config import NanoFlowConfig
from unit import FlowMatchingUnit, sample, load_ema
from viz import plot_samples, plot_image_samples


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: NanoFlowConfig) -> None:
    unit = FlowMatchingUnit(cfg)

    tnt_train(unit, unit.train_loader, max_epochs=cfg.training.epochs)
    unit.cleanup()

    # Post-training: sampling + plots (rank 0 only)
    if unit.rank == 0:
        raw = unit._unwrap()
        raw.eval()
        if unit.ema_ready():
            load_ema(raw, unit.ema_params)

        if cfg.save:
            inf = cfg.inference
            shape = tuple(inf.image_shape) if inf.image_shape else (2,)
            generated = sample(raw, unit.path, inf.n_samples, inf.num_steps, unit.device, shape=shape)

            title = f"After {cfg.training.epochs} epochs, {inf.num_steps} Euler steps"
            if cfg.dataset.name == "moons":
                real = torch.stack([unit.train_ds[i][0] for i in range(min(1000, len(unit.train_ds)))])
                plot_samples(real.cpu(), generated.cpu(), title, inf.samples_plot)
            else:
                plot_image_samples(generated, title, inf.samples_plot)


if __name__ == "__main__":
    main()
