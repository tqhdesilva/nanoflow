"""NanoFlow — training entry point."""

import torch
import hydra
from omegaconf import DictConfig
from torchtnt.framework import train as tnt_train

from unit import FlowMatchingUnit, sample, load_ema
from viz import plot_loss, plot_samples, plot_image_samples


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    unit = FlowMatchingUnit(cfg)

    tnt_train(unit, unit.train_loader, max_epochs=cfg.training.epochs)
    unit.cleanup()

    # Post-training: sampling + plots (rank 0 only)
    if unit.rank == 0:
        raw = unit._unwrap()
        raw.eval()
        if unit.ema_ready():
            load_ema(raw, unit.ema_params)

        shape = unit.image_shape or (2,)
        generated = sample(raw, unit.path, cfg.dataset.n_samples, cfg.num_steps, unit.device, shape=shape)

        plot_loss(unit.losses, cfg.loss_plot if cfg.save else None)
        if cfg.dataset.name == "moons":
            real = torch.stack([unit.train_ds[i][0] for i in range(min(1000, len(unit.train_ds)))])
            plot_samples(
                real.cpu(), generated.cpu(),
                f"After {cfg.training.epochs} epochs, {cfg.num_steps} Euler steps",
                cfg.samples_plot if cfg.save else None,
            )
        else:
            plot_image_samples(
                generated,
                f"After {cfg.training.epochs} epochs, {cfg.num_steps} Euler steps",
                cfg.samples_plot if cfg.save else None,
            )


if __name__ == "__main__":
    main()
