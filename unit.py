"""FlowMatchingUnit — TorchTNT AutoUnit subclass for flow matching training."""

import os
import signal
import sys
import time
from pathlib import Path
from typing import Any, Optional, Tuple

import hydra
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from torchtnt.framework import AutoUnit, State
from torchtnt.framework.auto_unit import TrainStepResults
from torchvision.utils import make_grid

from datasets import moons_dataset, fashion_dataset, cifar_dataset
from flow import NoisePath


# ---------------------------------------------------------------------------
# EMA helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def _init_ema(model):
    return [p.clone() for p in model.parameters()]


@torch.no_grad()
def _update_ema(ema_params, model, decay):
    for ema_p, p in zip(ema_params, model.parameters()):
        ema_p.mul_(decay).add_(p.data, alpha=1 - decay)


@torch.no_grad()
def load_ema(model, ema_params):
    """Swap model params with EMA params. Call again to swap back."""
    for p, ema_p in zip(model.parameters(), ema_params):
        p.data, ema_p.data = ema_p.data.clone(), p.data.clone()


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def save_checkpoint(path, model, optimizer, epoch, step, losses, **extra):
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "step": step,
        "losses": losses,
    }
    state.update(extra)
    torch.save(state, path)


def load_checkpoint(path, model, optimizer):
    ckpt = torch.load(path, weights_only=True)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

@torch.no_grad()
def sample(model, path: NoisePath, n_samples=1000, num_steps=100, device="cpu", shape=(2,)):
    """Generate samples via the path's sampling algorithm."""
    xt = torch.randn(n_samples, *shape, device=device)
    dt = 1.0 / num_steps
    for t_val in torch.linspace(0, 1, num_steps, device=device):
        xt = path.sample_step(model, xt, t_val, dt)
    return xt


# ---------------------------------------------------------------------------
# Dataset / DataLoader helpers
# ---------------------------------------------------------------------------

def _build_dataset(cfg: DictConfig, train=True):
    name = cfg.dataset.name
    if name == "moons":
        return moons_dataset(n=cfg.dataset.n, noise=cfg.dataset.noise, train=train)
    elif name == "fashion":
        return fashion_dataset(root=cfg.dataset.root, train=train)
    elif name == "cifar10":
        return cifar_dataset(root=cfg.dataset.root, train=train)
    raise ValueError(f"Unknown dataset: {name}")


def _build_dataloader(dataset, batch_size, num_workers, rank, world_size, shuffle=True):
    sampler = None
    if world_size > 1:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
        shuffle = False
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )


# ---------------------------------------------------------------------------
# FlowMatchingUnit
# ---------------------------------------------------------------------------

class FlowMatchingUnit(AutoUnit):
    """Self-contained training unit. Builds its own model, path, loaders, and writer."""

    def __init__(self, cfg: DictConfig):
        # Distributed setup
        self.rank, self.world_size = self._setup_distributed()
        if self.world_size > 1:
            device = torch.device(f"cuda:{self.rank}")
        else:
            device = torch.device(cfg.device)

        # Flow path
        self.path: NoisePath = hydra.utils.instantiate(cfg.flow)

        # Model
        module = hydra.utils.instantiate(cfg.model).to(device)
        if self.rank == 0:
            print(f"Model params: {sum(p.numel() for p in module.parameters()):,}")
        if self.world_size > 1:
            module = DDP(module, device_ids=[self.rank])

        # AutoUnit init
        tcfg = cfg.training
        super().__init__(
            module=module,
            device=device,
            precision=getattr(tcfg, "precision", None),
            clip_grad_norm=tcfg.grad_clip if tcfg.grad_clip > 0 else None,
            step_lr_interval="epoch",
        )

        self.cfg = cfg
        self.tcfg = tcfg

        # Image shape
        self.image_shape = tuple(cfg.dataset.image_shape) if cfg.dataset.image_shape else None

        # DataLoaders
        self.train_ds = _build_dataset(cfg, train=True)
        val_ds = _build_dataset(cfg, train=False)
        self.train_loader = _build_dataloader(
            self.train_ds, tcfg.batch_size, tcfg.num_workers,
            self.rank, self.world_size, shuffle=True,
        )
        self.val_loader = _build_dataloader(
            val_ds, tcfg.batch_size, tcfg.num_workers,
            self.rank, self.world_size, shuffle=False,
        )

        # TensorBoard (rank 0 only)
        self.writer = SummaryWriter(log_dir=tcfg.log_dir) if self.rank == 0 else None

        # EMA — initialized lazily after first training step (not from random init)
        self.ema_params = None
        self.ema_decay = getattr(tcfg, "ema_decay", 0)

        # Tracking
        self.losses = []
        self._epoch_loss = 0.0
        self._epoch_steps = 0
        self._epoch_start = 0.0

        # Checkpointing
        self._ckpt_dir = Path(tcfg.checkpoint_dir)
        self._ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Resume
        if tcfg.resume:
            raw = self._unwrap()
            ckpt = load_checkpoint(tcfg.resume, raw, self.optimizer)
            self.losses = ckpt.get("losses", [])
            if "ema" in ckpt and self.ema_params is not None:
                for ep, cp in zip(self.ema_params, ckpt["ema"]):
                    ep.data.copy_(cp)
            if self.rank == 0:
                print(f"Resumed from {tcfg.resume} at epoch {ckpt['epoch']}")

        # SIGTERM handler
        self._setup_sigterm()

    @staticmethod
    def _setup_distributed():
        if "RANK" not in os.environ:
            return 0, 1
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        torch.cuda.set_device(rank)
        return rank, world_size

    def _unwrap(self):
        return self.module.module if hasattr(self.module, "module") else self.module

    # --- AutoUnit interface ---

    def configure_optimizers_and_lr_scheduler(self, module):
        optimizer = torch.optim.Adam(module.parameters(), lr=self.tcfg.lr)
        warmup = getattr(self.tcfg, "warmup_epochs", 0)
        if warmup > 0:
            scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [
                torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-3, total_iters=warmup),
                torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.tcfg.epochs - warmup),
            ], milestones=[warmup])
        else:
            scheduler = None
        return optimizer, scheduler

    def move_data_to_device(self, state: State, data, non_blocking: bool):
        dev = self.device
        if isinstance(data, (tuple, list)):
            return tuple(d.to(dev) if isinstance(d, torch.Tensor) else d for d in data)
        if isinstance(data, torch.Tensor):
            return data.to(dev)
        return data

    def _prefetch_next_batch(self, state, data_iter):
        # Override: torch.cuda.stream(None) is broken on MPS.
        if self.device.type != "cuda":
            try:
                next_batch = next(data_iter)
            except StopIteration:
                self._phase_to_next_batch[state.active_phase] = None
                self._is_last_batch = True
                return
            self._phase_to_next_batch[state.active_phase] = self.move_data_to_device(
                state, next_batch, non_blocking=False,
            )
        else:
            super()._prefetch_next_batch(state, data_iter)

    def compute_loss(self, state: State, data) -> Tuple[torch.Tensor, Any]:
        x_0 = data[0] if isinstance(data, (tuple, list)) else data
        eps = torch.randn_like(x_0)
        t = torch.rand(x_0.size(0), *([1] * (x_0.dim() - 1)), device=x_0.device)
        xt = self.path.interpolate(x_0, eps, t)
        v_pred = self.module(xt, t.view(-1))
        vt = self.path.target(x_0, eps, t)
        loss = F.mse_loss(v_pred, vt)
        return loss, v_pred

    # --- Callbacks ---

    def on_train_epoch_start(self, state: State) -> None:
        self._epoch_loss = 0.0
        self._epoch_steps = 0
        self._epoch_start = time.perf_counter()

    def on_train_step_end(self, state: State, data, step: int, results: TrainStepResults) -> None:
        loss_val = results.loss.item()
        self._epoch_loss += loss_val
        self._epoch_steps += 1
        global_step = self.train_progress.num_steps_completed

        # EMA: lazy init on first step, then update
        if self.ema_decay > 0:
            raw = self._unwrap()
            if self.ema_params is None:
                self.ema_params = _init_ema(raw)
            else:
                _update_ema(self.ema_params, raw, self.ema_decay)

        # Per-step TensorBoard logging
        if self.rank == 0 and self.writer and global_step % self.tcfg.log_every == 0:
            self.writer.add_scalar("train/loss", loss_val, global_step)
            lr = self.optimizer.param_groups[0]["lr"]
            self.writer.add_scalar("train/lr", lr, global_step)
            if results.total_grad_norm is not None:
                self.writer.add_scalar("train/grad_norm", results.total_grad_norm.item(), global_step)

    def on_train_epoch_end(self, state: State) -> None:
        epoch = self.train_progress.num_epochs_completed
        epoch_time = time.perf_counter() - self._epoch_start
        avg_loss = self._epoch_loss / max(self._epoch_steps, 1)
        self.losses.append(avg_loss)

        # Validation
        val_loss = None
        if self.rank == 0 and self.val_loader and (epoch % self.tcfg.save_every == 0 or epoch == self.tcfg.epochs):
            val_loss = self._run_validation()

        # Per-epoch logging
        if self.rank == 0:
            lr = self.optimizer.param_groups[0]["lr"]
            n_samples = self._epoch_steps * self.tcfg.batch_size
            throughput = n_samples / max(epoch_time, 1e-6)

            summary = f"Epoch {epoch}/{self.tcfg.epochs} | loss={avg_loss:.4f}"
            if val_loss is not None:
                summary += f" | val={val_loss:.4f}"
            summary += f" | lr={lr:.2e} | {throughput:.0f} sam/s | {epoch_time:.1f}s"
            print(summary)

            if self.writer:
                self.writer.add_scalar("train/epoch_loss", avg_loss, epoch)
                self.writer.add_scalar("timing/epoch_sec", epoch_time, epoch)
                self.writer.add_scalar("timing/samples_per_sec", throughput, epoch)
                if val_loss is not None:
                    self.writer.add_scalar("val/loss", val_loss, epoch)

                # GPU memory (once, after first epoch)
                if epoch == 1 and torch.cuda.is_available():
                    mem_mb = torch.cuda.max_memory_allocated() / 1e6
                    self.writer.add_scalar("system/gpu_mem_peak_mb", mem_mb, epoch)

            # Checkpoint
            if epoch % self.tcfg.save_every == 0:
                self._save(epoch)

            # Sample images
            if self.image_shape and epoch % self.tcfg.save_every == 0 and self.writer:
                self._log_samples(epoch)

    def on_train_end(self, state: State) -> None:
        if self.rank == 0:
            self._save(self.train_progress.num_epochs_completed)
            if self.writer:
                self.writer.close()

    def cleanup(self):
        if self.world_size > 1:
            dist.destroy_process_group()

    # --- Helpers ---

    def ema_ready(self):
        """Whether EMA has had enough steps to be useful for sampling."""
        if self.ema_params is None or self.ema_decay <= 0:
            return False
        min_steps = 5 * 0.693 / (1 - self.ema_decay)
        return self.train_progress.num_steps_completed > min_steps

    @torch.no_grad()
    def _run_validation(self):
        raw = self._unwrap()
        raw.eval()
        total_loss = 0.0
        n = 0
        for data in self.val_loader:
            x_0 = data[0] if isinstance(data, (tuple, list)) else data
            x_0 = x_0.to(self.device)
            eps = torch.randn_like(x_0)
            t = torch.rand(x_0.size(0), *([1] * (x_0.dim() - 1)), device=x_0.device)
            xt = self.path.interpolate(x_0, eps, t)
            v_pred = raw(xt, t.view(-1))
            vt = self.path.target(x_0, eps, t)
            total_loss += F.mse_loss(v_pred, vt).item()
            n += 1
        raw.train()
        return total_loss / max(n, 1)

    def _save(self, epoch):
        raw = self._unwrap()
        extra = {}
        if self.ema_params is not None:
            extra["ema"] = [p.data.cpu() for p in self.ema_params]
        save_checkpoint(
            self._ckpt_dir / f"{self.tcfg.run_name}_latest.pt",
            raw, self.optimizer, epoch,
            self.train_progress.num_steps_completed,
            self.losses, **extra,
        )

    @torch.no_grad()
    def _log_samples(self, epoch, n_samples=64, num_steps=100):
        raw = self._unwrap()
        raw.eval()
        if self.ema_ready():
            load_ema(raw, self.ema_params)
        samples = sample(raw, self.path, n_samples, num_steps, self.device, shape=self.image_shape)
        if self.ema_ready():
            load_ema(raw, self.ema_params)  # swap back
        raw.train()
        grid = make_grid(samples.clamp(-1, 1) * 0.5 + 0.5, nrow=8)
        self.writer.add_image("samples/generated", grid, epoch)

    def _setup_sigterm(self):
        if self.rank != 0:
            return
        def handler(sig, frame):
            epoch = self.train_progress.num_epochs_completed
            path = self._ckpt_dir / f"{self.tcfg.run_name}_preempted.pt"
            print(f"\nSIGTERM at epoch {epoch}, saving to {path}")
            self._save(epoch)
            sys.exit(0)
        signal.signal(signal.SIGTERM, handler)
