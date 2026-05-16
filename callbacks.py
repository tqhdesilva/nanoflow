"""NanoFlow training callbacks.

Plain-Python callback protocol. Each hook takes `trainer` with no framework base class.
Hooks invoked by `Trainer.fit`: on_train_start, on_train_epoch_start, on_train_step_end,
on_train_epoch_end, on_eval_epoch_start, on_eval_epoch_end, on_train_end.
"""

from __future__ import annotations

import os
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
import yaml
from omegaconf import DictConfig, OmegaConf
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid


def _rank() -> int:
    return int(os.environ.get("RANK", 0))


def make_run_dir(runs_dir: str, run_prefix: str) -> Path:
    """Create `runs/{prefix}_{timestamp}/` and return its path."""
    run_id = f"{run_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = Path(runs_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _git_info() -> dict:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode()
        branch = (
            subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"])
            .strip()
            .decode()
        )
        dirty = subprocess.call(["git", "diff", "--quiet"]) != 0
        info = {"commit": commit, "branch": branch, "dirty": dirty}
        if dirty:
            info["diff"] = subprocess.check_output(["git", "diff"]).decode()
        return info
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {}


class RunDirCallback:
    """Rank-0 run directory + TensorBoard writer + metadata.yaml.

    Other callbacks read `.run_dir`, `.ckpt_dir`, `.writer` after init.
    """

    def __init__(self, runs_dir: str, run_prefix: str, cfg: DictConfig):
        self.rank = _rank()
        run_id = f"{run_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.run_dir = Path(runs_dir) / run_id
        self.ckpt_dir = self.run_dir / "checkpoints"
        self.tb_dir = self.run_dir / "tensorboard"
        self.writer: Optional[SummaryWriter] = None
        if self.rank == 0:
            self.run_dir.mkdir(parents=True, exist_ok=True)
            self.ckpt_dir.mkdir(parents=True, exist_ok=True)
            self.tb_dir.mkdir(parents=True, exist_ok=True)
            print(f"Run dir: {self.run_dir}")
            meta = {
                "config": OmegaConf.to_container(cfg, resolve=True),
                "git": _git_info(),
            }
            with open(self.run_dir / "metadata.yaml", "w") as f:
                yaml.dump(meta, f, default_flow_style=False, sort_keys=False)
            self.writer = SummaryWriter(log_dir=str(self.tb_dir))

    def on_train_end(self, trainer) -> None:
        if self.writer is not None:
            self.writer.close()


class CheckpointCallback:
    """Save model + optimizer + scheduler + EMA + scaler + train_progress.

    Saves every `save_every` epochs and at end of training. On `on_train_start`,
    restores from `resume` path if provided. `save_path(name)` is exposed so a
    SIGTERM handler can write `preempted.pt`.
    """

    def __init__(
        self,
        ckpt_dir: Path,
        save_every: int,
        resume: Optional[str] = None,
    ):
        self.rank = _rank()
        self.ckpt_dir = ckpt_dir
        self.save_every = save_every
        self.resume = resume

    def save_path(self, name: str = "latest") -> Path:
        return self.ckpt_dir / f"{name}.pt"

    def save(self, trainer, name: str = "latest") -> None:
        if self.rank != 0:
            return
        torch.save(trainer.state_dict(), self.save_path(name))

    def on_train_start(self, trainer) -> None:
        if not self.resume:
            return
        ckpt = torch.load(self.resume, weights_only=True, map_location=trainer.device)
        trainer.load_state_dict(ckpt)
        if self.rank == 0:
            epoch = ckpt.get("train_progress", {}).get("num_epochs_completed", "?")
            print(f"Resumed from {self.resume} at epoch {epoch}")

    def on_train_epoch_end(self, trainer) -> None:
        epoch = trainer.epoch
        if epoch % self.save_every == 0:
            self.save(trainer, "latest")

    def on_train_end(self, trainer) -> None:
        self.save(trainer, "latest")


class EpochSummaryCallback:
    """Rank-0 console summary + epoch-level scalars (loss, val/loss, throughput)."""

    def __init__(self, writer: Optional[SummaryWriter], total_epochs: int, batch_size: int):
        self.rank = _rank()
        self.writer = writer
        self.total_epochs = total_epochs
        self.batch_size = batch_size
        self._epoch_start = 0.0

    def on_train_epoch_start(self, trainer) -> None:
        trainer.train_loss_sum = 0.0
        trainer.train_loss_steps = 0
        self._epoch_start = time.perf_counter()

    def on_eval_epoch_start(self, trainer) -> None:
        trainer.val_loss_sum = 0.0
        trainer.val_loss_steps = 0

    def on_train_epoch_end(self, trainer) -> None:
        epoch = trainer.epoch
        epoch_time = time.perf_counter() - self._epoch_start
        avg_loss = trainer.train_loss_sum / max(trainer.train_loss_steps, 1)
        trainer.losses.append(avg_loss)
        n = trainer.train_loss_steps * self.batch_size
        throughput = n / max(epoch_time, 1e-6)
        if self.rank != 0 or self.writer is None:
            return
        print(
            f"Epoch {epoch}/{self.total_epochs} | loss={avg_loss:.4f} | "
            f"{throughput:.0f} sam/s | {epoch_time:.1f}s"
        )
        self.writer.add_scalar("train/epoch_loss", avg_loss, epoch)
        self.writer.add_scalar("timing/epoch_sec", epoch_time, epoch)
        self.writer.add_scalar("timing/samples_per_sec", throughput, epoch)

    def on_eval_epoch_end(self, trainer) -> None:
        if self.rank != 0 or self.writer is None or trainer.val_loss_steps == 0:
            return
        val = trainer.val_loss_sum / trainer.val_loss_steps
        epoch = trainer.epoch
        print(f"  eval @ epoch {epoch} | val={val:.4f}")
        self.writer.add_scalar("val/loss", val, epoch)


class StepLossCallback:
    """Log per-step training loss to TB at `log_every` cadence (rank 0 only)."""

    def __init__(self, writer: Optional[SummaryWriter], log_every: int):
        self.rank = _rank()
        self.writer = writer
        self.log_every = log_every

    def on_train_step_end(self, trainer) -> None:
        if self.rank != 0 or self.writer is None:
            return
        if trainer.step % self.log_every != 0:
            return
        if trainer.train_loss_steps > 0:
            self.writer.add_scalar(
                "train/loss",
                trainer.train_loss_sum / trainer.train_loss_steps,
                trainer.step,
            )


class LRMonitorCallback:
    """Log `train/lr` at end of each train epoch (rank 0 only)."""

    def __init__(self, writer: Optional[SummaryWriter]):
        self.rank = _rank()
        self.writer = writer

    def on_train_epoch_end(self, trainer) -> None:
        if self.rank != 0 or self.writer is None or trainer.optimizer is None:
            return
        lr = trainer.optimizer.param_groups[0]["lr"]
        self.writer.add_scalar("train/lr", lr, trainer.epoch)


class SampleLoggerCallback:
    """Rank-0: every `save_every` epochs, generate samples and log a TB image grid."""

    def __init__(
        self,
        writer: Optional[SummaryWriter],
        save_every: int,
        latent_shape: list,
        n_samples: int,
        num_steps: int,
        guidance_scale: float = 1.0,
        p_uncond: Optional[float] = None,
    ):
        self.rank = _rank()
        self.writer = writer
        self.save_every = save_every
        self.latent_shape = tuple(latent_shape)
        self.n_samples = n_samples
        self.num_steps = num_steps
        self.guidance_scale = guidance_scale
        self.p_uncond = p_uncond

    def on_train_epoch_end(self, trainer) -> None:
        if self.rank != 0 or self.writer is None:
            return
        epoch = trainer.epoch
        if epoch % self.save_every != 0:
            return

        from inference import euler_sample, guided_euler_sample

        sample_model = trainer.eval_model
        was_training = sample_model.training
        sample_model.eval()
        with torch.no_grad():
            noise = torch.randn(self.n_samples, *self.latent_shape, device=trainer.device)
            if self.p_uncond is not None and hasattr(sample_model, "num_classes"):
                labels = torch.randint(
                    0,
                    sample_model.num_classes,
                    (self.n_samples,),
                    device=trainer.device,
                )
                samples = guided_euler_sample(
                    sample_model, noise, self.num_steps, labels, self.guidance_scale
                )
            else:
                samples = euler_sample(sample_model, noise, self.num_steps)
        if was_training:
            sample_model.train()
        grid = make_grid(samples.clamp(-1, 1) * 0.5 + 0.5, nrow=8)
        self.writer.add_image("samples/generated", grid, epoch)
