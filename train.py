"""NanoFlow — training entry point."""

import hydra
from omegaconf import DictConfig

from datasets import moons_dataset, fashion_dataset, cifar_dataset
from trainer import train, sample
from viz import plot_loss, plot_samples, plot_image_samples


def build_dataset(cfg):
    name = cfg.dataset.name
    if name == "moons":
        return moons_dataset(n=cfg.dataset.n, noise=cfg.dataset.noise)
    elif name == "fashion":
        return fashion_dataset(root=cfg.dataset.root)
    elif name == "cifar10":
        return cifar_dataset(root=cfg.dataset.root)
    raise ValueError(f"Unknown dataset: {name}")


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    dataset = build_dataset(cfg)
    model = hydra.utils.instantiate(cfg.model).to(cfg.device)

    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    losses = train(model, dataset, cfg.training, cfg.device)

    image_shape = tuple(cfg.dataset.image_shape) if cfg.dataset.image_shape else (2,)
    generated = sample(model, cfg.dataset.n_samples, cfg.num_steps, cfg.device, shape=image_shape)

    plot_loss(losses, cfg.loss_plot if cfg.save else None)
    if cfg.dataset.name == "moons":
        plot_samples(
            dataset[:1000].cpu(), generated.cpu(),
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
