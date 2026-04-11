"""NanoFlow — training entry point."""

import os

import hydra
import torch
import torch.distributed as dist
from omegaconf import DictConfig
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from torchtnt.framework import train as tnt_train

from datasets import moons_dataset, fashion_dataset, cifar_dataset
from flow import CondOT
from trainer import FlowMatchingUnit, sample, _load_ema
from viz import plot_loss, plot_samples, plot_image_samples


# ---------------------------------------------------------------------------
# Distributed setup
# ---------------------------------------------------------------------------

def setup_distributed():
    """Initialize DDP if launched via torchrun. Returns (rank, world_size)."""
    if "RANK" not in os.environ:
        return 0, 1
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    return rank, world_size


# ---------------------------------------------------------------------------
# Dataset / DataLoader
# ---------------------------------------------------------------------------

def build_dataset(cfg, train=True):
    name = cfg.dataset.name
    if name == "moons":
        return moons_dataset(n=cfg.dataset.n, noise=cfg.dataset.noise, train=train)
    elif name == "fashion":
        return fashion_dataset(root=cfg.dataset.root, train=train)
    elif name == "cifar10":
        return cifar_dataset(root=cfg.dataset.root, train=train)
    raise ValueError(f"Unknown dataset: {name}")


def build_dataloader(dataset, cfg, rank=0, world_size=1, shuffle=True):
    sampler = None
    if world_size > 1:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
        shuffle = False
    return DataLoader(
        dataset,
        batch_size=cfg.training.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=cfg.training.num_workers,
        pin_memory=True,
        drop_last=True,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    rank, world_size = setup_distributed()

    # Device
    if world_size > 1:
        device = torch.device(f"cuda:{rank}")
    else:
        device = torch.device(cfg.device)

    # Flow path
    path = hydra.utils.instantiate(cfg.flow)

    # Dataset + DataLoader
    train_ds = build_dataset(cfg, train=True)
    val_ds = build_dataset(cfg, train=False)
    train_loader = build_dataloader(train_ds, cfg, rank, world_size, shuffle=True)
    val_loader = build_dataloader(val_ds, cfg, rank, world_size, shuffle=False)

    # Model
    model = hydra.utils.instantiate(cfg.model).to(device)
    if rank == 0:
        print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    if world_size > 1:
        model = DDP(model, device_ids=[rank])

    # Image shape
    image_shape = tuple(cfg.dataset.image_shape) if cfg.dataset.image_shape else None

    # TensorBoard writer (rank 0 only)
    writer = None
    if rank == 0:
        writer = SummaryWriter(log_dir=cfg.training.log_dir)

    # Training unit
    unit = FlowMatchingUnit(
        module=model,
        path=path,
        cfg=cfg.training,
        device=device,
        rank=rank,
        writer=writer,
        val_loader=val_loader,
        image_shape=image_shape,
    )

    # Resume
    if cfg.training.resume:
        from trainer import load_checkpoint
        raw = model.module if hasattr(model, "module") else model
        ckpt = load_checkpoint(cfg.training.resume, raw, unit.optimizer)
        unit.losses = ckpt.get("losses", [])
        if "ema" in ckpt and unit.ema_params is not None:
            for ep, cp in zip(unit.ema_params, ckpt["ema"]):
                ep.data.copy_(cp)
        if rank == 0:
            print(f"Resumed from {cfg.training.resume} at epoch {ckpt['epoch']}")

    # Train
    tnt_train(unit, train_loader, max_epochs=cfg.training.epochs)

    # Cleanup DDP
    if world_size > 1:
        dist.destroy_process_group()

    # Post-training: sampling + plots (rank 0 only)
    if rank == 0:
        raw = model.module if hasattr(model, "module") else model
        raw.eval()
        # Use EMA for sampling if it's had enough steps to track the model.
        # Rule of thumb: need ~5 half-lives = 5 * ln(2)/(1-decay) steps.
        ema_min_steps = 5 * 0.693 / (1 - unit.ema_decay) if unit.ema_decay > 0 else float("inf")
        ema_ready = (
            unit.ema_params is not None
            and unit.train_progress.num_steps_completed > ema_min_steps
        )
        if ema_ready:
            _load_ema(raw, unit.ema_params)

        sample_shape = image_shape if image_shape else (2,)
        generated = sample(raw, path, cfg.dataset.n_samples, cfg.num_steps, device, shape=sample_shape)

        plot_loss(unit.losses, cfg.loss_plot if cfg.save else None)
        if cfg.dataset.name == "moons":
            # Get some real samples for comparison
            real = torch.stack([train_ds[i][0] for i in range(min(1000, len(train_ds)))])
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
