"""NanoFlow training callbacks.

Plain-Python callback protocol. Each hook takes `trainer` with no framework base class.
Hooks invoked by `Trainer.fit`: on_train_start, on_train_epoch_start, on_train_step_end,
on_train_epoch_end, on_eval_epoch_start, on_eval_epoch_end, on_train_end,
on_train_cleanup.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
import torch.distributed as dist
import yaml
from omegaconf import DictConfig, OmegaConf
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid


def _rank() -> int:
    return int(os.environ.get("RANK", 0))


def _world_size() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return int(os.environ.get("WORLD_SIZE", 1))


def _dist_ready() -> bool:
    return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1


def _distributed_loss_stats(loss_sum: float, sample_count: int, device: torch.device):
    if not _dist_ready():
        return loss_sum, sample_count
    backend = dist.get_backend()
    tensor_device = device if backend == "nccl" else torch.device("cpu")
    stats = torch.tensor(
        [float(loss_sum), float(sample_count)],
        dtype=torch.float64,
        device=tensor_device,
    )
    dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    return float(stats[0].item()), int(stats[1].item())


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


class SummaryWriterHandle:
    """Small proxy so callbacks can share a writer opened after resume."""

    def __init__(self):
        self.writer: Optional[SummaryWriter] = None

    def open(self, log_dir: Path, purge_step: Optional[int] = None) -> None:
        kwargs = {}
        if purge_step is not None and purge_step > 0:
            kwargs["purge_step"] = purge_step
        self.writer = SummaryWriter(log_dir=str(log_dir), **kwargs)

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()
            self.writer = None

    def add_scalar(self, *args, **kwargs) -> None:
        if self.writer is None:
            raise RuntimeError("TensorBoard writer has not been opened")
        self.writer.add_scalar(*args, **kwargs)

    def add_image(self, *args, **kwargs) -> None:
        if self.writer is None:
            raise RuntimeError("TensorBoard writer has not been opened")
        self.writer.add_image(*args, **kwargs)


class RunDirCallback:
    """Rank-0 run directory + TensorBoard writer + metadata.yaml.

    Other callbacks read `.run_dir`, `.ckpt_dir`, `.writer` after init.
    """

    def __init__(
        self,
        runs_dir: str,
        run_prefix: str,
        cfg: DictConfig,
        run_dir: Optional[str] = None,
    ):
        self.rank = _rank()
        self.stable_run_dir = run_dir is not None
        if self.stable_run_dir:
            self.run_dir = Path(run_dir)
        else:
            run_id = f"{run_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.run_dir = Path(runs_dir) / run_id
        self.ckpt_dir = self.run_dir / "checkpoints"
        self.tb_dir = self.run_dir / "tensorboard"
        self.writer: Optional[SummaryWriterHandle] = (
            SummaryWriterHandle() if self.rank == 0 else None
        )
        if self.rank == 0:
            self.run_dir.mkdir(parents=True, exist_ok=True)
            self.ckpt_dir.mkdir(parents=True, exist_ok=True)
            self.tb_dir.mkdir(parents=True, exist_ok=True)
            print(f"Run dir: {self.run_dir}")
            metadata_path = self.run_dir / "metadata.yaml"
            if not self.stable_run_dir or not metadata_path.exists():
                meta = {
                    "config": OmegaConf.to_container(cfg, resolve=True),
                    "git": _git_info(),
                }
                with open(metadata_path, "w") as f:
                    yaml.dump(meta, f, default_flow_style=False, sort_keys=False)

    def on_train_start(self, trainer) -> None:
        if self.rank != 0 or self.writer is None:
            return
        step = getattr(trainer, "step", 0)
        purge_step = step if self.stable_run_dir and step > 0 else None
        self.writer.open(self.tb_dir, purge_step=purge_step)

    def on_train_cleanup(self, trainer) -> None:
        if self.writer is not None:
            self.writer.close()


class CheckpointCallback:
    """Save model + optimizer + scheduler + EMA + scaler + train_progress.

    Saves every `checkpoint_every` epochs and at end of training. On
    `on_train_start`, restores from `resume` path if provided. `save_path(name)`
    is exposed for callers that need a named checkpoint.
    """

    def __init__(
        self,
        ckpt_dir: Path,
        checkpoint_every: int,
        resume: Optional[str] = None,
    ):
        self.rank = _rank()
        self.ckpt_dir = ckpt_dir
        self.checkpoint_every = checkpoint_every
        self.resume = resume

    def save_path(self, name: str = "latest") -> Path:
        return self.ckpt_dir / f"{name}.pt"

    @staticmethod
    def _atomic_save(obj, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
        )
        tmp_path = Path(tmp_name)
        try:
            with os.fdopen(fd, "wb") as f:
                torch.save(obj, f)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, path)
            try:
                dir_fd = os.open(path.parent, os.O_RDONLY)
                try:
                    os.fsync(dir_fd)
                finally:
                    os.close(dir_fd)
            except OSError:
                pass
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    def save(self, trainer, name: str = "latest") -> None:
        if self.rank != 0:
            return
        self._atomic_save(trainer.state_dict(), self.save_path(name))

    def _resume_path(self) -> Optional[Path]:
        if not self.resume:
            return None
        if self.resume == "auto":
            path = self.save_path("latest")
            return path if path.exists() else None
        return Path(self.resume)

    def on_train_start(self, trainer) -> None:
        resume_path = self._resume_path()
        if resume_path is None:
            return
        ckpt = torch.load(resume_path, weights_only=True, map_location=trainer.device)
        trainer.load_state_dict(ckpt)
        if self.rank == 0:
            epoch = ckpt.get("train_progress", {}).get("num_epochs_completed", "?")
            print(f"Resumed from {resume_path} at epoch {epoch}")

    def on_train_epoch_end(self, trainer) -> None:
        epoch = trainer.epoch
        if self.checkpoint_every > 0 and epoch % self.checkpoint_every == 0:
            self.save(trainer, "latest")

    def on_train_end(self, trainer) -> None:
        self.save(trainer, "latest")


class EpochSummaryCallback:
    """Rank-0 console summary + epoch-level scalars (loss, val/loss, throughput)."""

    def __init__(
        self, writer: Optional[SummaryWriter], total_epochs: int, batch_size: int
    ):
        self.rank = _rank()
        self.writer = writer
        self.total_epochs = total_epochs
        self.batch_size = batch_size
        self.world_size = _world_size()
        self.global_batch_size = batch_size * self.world_size
        self._epoch_start = 0.0

    def on_train_start(self, trainer) -> None:
        if self.rank != 0 or self.writer is None:
            return
        print(f"Effective global batch size: {self.global_batch_size}")
        self.writer.add_scalar(
            "train/effective_batch_size",
            self.global_batch_size,
            getattr(trainer, "step", 0),
        )

    def on_train_epoch_start(self, trainer) -> None:
        self._epoch_start = time.perf_counter()

    def on_eval_epoch_start(self, trainer) -> None:
        pass

    def on_train_epoch_end(self, trainer) -> None:
        epoch = trainer.epoch
        epoch_time = time.perf_counter() - self._epoch_start
        loss_sum, sample_count = _distributed_loss_stats(
            trainer.train_loss_sum,
            trainer.train_loss_samples,
            trainer.device,
        )
        avg_loss = loss_sum / max(sample_count, 1)
        trainer.losses.append(avg_loss)
        throughput = sample_count / max(epoch_time, 1e-6)
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
        loss_sum, sample_count = _distributed_loss_stats(
            trainer.val_loss_sum,
            trainer.val_loss_samples,
            trainer.device,
        )
        if self.rank != 0 or self.writer is None or sample_count == 0:
            return
        val = loss_sum / sample_count
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
        self.writer.add_scalar("train/loss", trainer.last_train_loss, trainer.step)


class LRMonitorCallback:
    """Log `train/lr` at the same step cadence as train loss, rank 0 only."""

    def __init__(self, writer: Optional[SummaryWriter], log_every: int):
        self.rank = _rank()
        self.writer = writer
        self.log_every = log_every

    def on_train_start(self, trainer) -> None:
        if self.rank != 0 or self.writer is None or trainer.optimizer is None:
            return
        lr = trainer.optimizer.param_groups[0]["lr"]
        self.writer.add_scalar("train/lr", lr, getattr(trainer, "step", 0))

    def on_train_step_end(self, trainer) -> None:
        if self.rank != 0 or self.writer is None or trainer.optimizer is None:
            return
        if trainer.step % self.log_every != 0:
            return
        lr = trainer.optimizer.param_groups[0]["lr"]
        self.writer.add_scalar("train/lr", lr, trainer.step)


class SampleLoggerCallback:
    """Rank-0: periodically generate samples and log a TB image grid."""

    def __init__(
        self,
        writer: Optional[SummaryWriter],
        every: int,
        latent_shape: list,
        n_samples: int,
        num_steps: int,
        guidance_scale: float = 1.0,
        p_uncond: Optional[float] = None,
        vae_cfg=None,
    ):
        self.rank = _rank()
        self.writer = writer
        self.every = every
        self.latent_shape = tuple(latent_shape)
        self.n_samples = n_samples
        self.num_steps = num_steps
        self.guidance_scale = guidance_scale
        self.p_uncond = p_uncond
        self.vae_cfg = vae_cfg
        self.vae = None

    def on_train_epoch_end(self, trainer) -> None:
        if self.rank != 0 or self.writer is None:
            return
        epoch = trainer.epoch
        if self.every <= 0 or epoch % self.every != 0:
            return

        from inference import euler_sample, guided_euler_sample

        sample_model = trainer.eval_model
        was_training = sample_model.training
        sample_model.eval()
        with torch.no_grad():
            noise = torch.randn(
                self.n_samples, *self.latent_shape, device=trainer.device
            )
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
        if self.vae_cfg is not None:
            if self.vae is None:
                import hydra

                self.vae = hydra.utils.instantiate(
                    self.vae_cfg, device=str(trainer.device)
                )
            samples = self.vae.decode(samples)
        samples = samples.detach().cpu()
        grid = make_grid(samples.clamp(-1, 1) * 0.5 + 0.5, nrow=8)
        self.writer.add_image("samples/generated", grid, epoch)
