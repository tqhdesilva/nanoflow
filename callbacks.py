"""NanoFlow training callbacks.

Custom: RunDir (run dir + metadata + tb logger), Checkpoint (save/restore),
EpochSummary (rank-0 prints + epoch loss/throughput scalars), SampleLogger
(generated-sample grids). Built-in `LearningRateMonitor` from torchtnt covers
LR logging.
"""

from __future__ import annotations

import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
import yaml
from omegaconf import DictConfig, OmegaConf
from torch.utils.tensorboard import SummaryWriter
from torchtnt.framework.callback import Callback
from torchtnt.framework.state import State
from torchtnt.utils.distributed import get_global_rank
from torchtnt.utils.loggers import TensorBoardLogger
from torchvision.utils import make_grid

from unit import euler_sample, guided_euler_sample


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


class RunDirCallback(Callback):
    """Rank-0 run directory + TensorBoard logger + metadata.yaml.

    Other callbacks read `.run_dir`, `.ckpt_dir`, `.tb_logger` after init.
    """

    def __init__(self, runs_dir: str, run_prefix: str, cfg: DictConfig):
        self.rank = get_global_rank()
        run_id = f"{run_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.run_dir = Path(runs_dir) / run_id
        self.ckpt_dir = self.run_dir / "checkpoints"
        self.tb_dir = self.run_dir / "tensorboard"
        self.tb_logger: Optional[TensorBoardLogger] = None
        self._cfg = cfg
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
        self.tb_logger = TensorBoardLogger(path=str(self.tb_dir))

    def on_train_end(self, state: State, unit) -> None:
        if self.tb_logger is not None:
            self.tb_logger.close()


class CheckpointCallback(Callback):
    """Save model + optimizer + EMA + train_progress every `save_every` epochs.

    On `on_train_start`, restore from `resume` path if provided.
    `save_path(name)` is exposed so a SIGTERM handler can write `preempted.pt`.
    """

    def __init__(
        self,
        ckpt_dir: Path,
        save_every: int,
        resume: Optional[str] = None,
    ):
        self.rank = get_global_rank()
        self.ckpt_dir = ckpt_dir
        self.save_every = save_every
        self.resume = resume
        self.losses: list = []

    def save_path(self, name: str = "latest") -> Path:
        return self.ckpt_dir / f"{name}.pt"

    def _build_state(self, unit, name: str) -> dict:
        ckpt = {
            "model_state": unit._raw_module.state_dict(),
            "optimizer_state": unit.optimizer.state_dict() if unit.optimizer else None,
            "lr_scheduler_state": (
                unit.lr_scheduler.state_dict() if unit.lr_scheduler else None
            ),
            "train_progress": unit.train_progress.state_dict(),
            "losses": list(self.losses),
        }
        if getattr(unit, "swa_model", None) is not None:
            ckpt["ema_state"] = unit.swa_model.module.state_dict()
        return ckpt

    def save(self, unit, name: str = "latest") -> None:
        if self.rank != 0:
            return
        path = self.save_path(name)
        torch.save(self._build_state(unit, name), path)

    def on_train_start(self, state: State, unit) -> None:
        if not self.resume:
            return
        ckpt = torch.load(self.resume, weights_only=True, map_location=unit.device)
        unit._raw_module.load_state_dict(ckpt["model_state"])
        if ckpt.get("optimizer_state") and unit.optimizer:
            unit.optimizer.load_state_dict(ckpt["optimizer_state"])
        if ckpt.get("lr_scheduler_state") and unit.lr_scheduler:
            unit.lr_scheduler.load_state_dict(ckpt["lr_scheduler_state"])
        if ckpt.get("ema_state") and getattr(unit, "swa_model", None) is not None:
            unit.swa_model.module.load_state_dict(ckpt["ema_state"])
        prog = ckpt.get("train_progress", {})
        if prog:
            unit.train_progress.load_state_dict(prog)
        self.losses = list(ckpt.get("losses", []))
        if self.rank == 0:
            print(
                f"Resumed from {self.resume} at epoch "
                f"{prog.get('num_epochs_completed', '?')}"
            )

    def on_train_epoch_end(self, state: State, unit) -> None:
        if unit.train_loss_steps > 0:
            self.losses.append(unit.train_loss_sum / unit.train_loss_steps)
        epoch = unit.train_progress.num_epochs_completed
        if epoch % self.save_every == 0:
            self.save(unit, "latest")

    def on_train_end(self, state: State, unit) -> None:
        self.save(unit, "latest")


class EpochSummaryCallback(Callback):
    """Rank-0 console summary + epoch-level scalars (loss, val/loss, throughput)."""

    def __init__(self, tb_logger: TensorBoardLogger, total_epochs: int, batch_size: int):
        self.rank = get_global_rank()
        self.tb_logger = tb_logger
        self.total_epochs = total_epochs
        self.batch_size = batch_size
        self._epoch_start = 0.0

    def on_train_epoch_start(self, state: State, unit) -> None:
        unit.train_loss_sum = 0.0
        unit.train_loss_steps = 0
        self._epoch_start = time.perf_counter()

    def on_eval_epoch_start(self, state: State, unit) -> None:
        unit.val_loss_sum = 0.0
        unit.val_loss_steps = 0

    def on_train_epoch_end(self, state: State, unit) -> None:
        epoch = unit.train_progress.num_epochs_completed
        epoch_time = time.perf_counter() - self._epoch_start
        avg_loss = unit.train_loss_sum / max(unit.train_loss_steps, 1)
        n = unit.train_loss_steps * self.batch_size
        throughput = n / max(epoch_time, 1e-6)
        if self.rank != 0:
            return
        print(
            f"Epoch {epoch}/{self.total_epochs} | loss={avg_loss:.4f} | "
            f"{throughput:.0f} sam/s | {epoch_time:.1f}s"
        )
        self.tb_logger.log("train/epoch_loss", avg_loss, epoch)
        self.tb_logger.log("timing/epoch_sec", epoch_time, epoch)
        self.tb_logger.log("timing/samples_per_sec", throughput, epoch)

    def on_eval_epoch_end(self, state: State, unit) -> None:
        if self.rank != 0 or unit.val_loss_steps == 0:
            return
        val = unit.val_loss_sum / unit.val_loss_steps
        epoch = unit.train_progress.num_epochs_completed
        print(f"  eval @ epoch {epoch} | val={val:.4f}")
        self.tb_logger.log("val/loss", val, epoch)


class StepLossCallback(Callback):
    """Log per-step training loss to TB at `log_every` cadence (rank 0 only)."""

    def __init__(self, tb_logger: TensorBoardLogger, log_every: int):
        self.rank = get_global_rank()
        self.tb_logger = tb_logger
        self.log_every = log_every

    def on_train_step_end(self, state: State, unit) -> None:
        if self.rank != 0:
            return
        step = unit.train_progress.num_steps_completed
        if step % self.log_every != 0:
            return
        if unit.train_loss_steps > 0:
            self.tb_logger.log(
                "train/loss",
                unit.train_loss_sum / unit.train_loss_steps,
                step,
            )


class SampleLoggerCallback(Callback):
    """Rank-0: every `save_every` epochs, generate samples and log a TB image grid."""

    def __init__(
        self,
        tb_logger: TensorBoardLogger,
        save_every: int,
        latent_shape: list,
        n_samples: int,
        num_steps: int,
        guidance_scale: float = 1.0,
        p_uncond: Optional[float] = None,
    ):
        self.rank = get_global_rank()
        self.tb_logger = tb_logger
        self.save_every = save_every
        self.latent_shape = tuple(latent_shape)
        self.n_samples = n_samples
        self.num_steps = num_steps
        self.guidance_scale = guidance_scale
        self.p_uncond = p_uncond

    def on_train_epoch_end(self, state: State, unit) -> None:
        if self.rank != 0:
            return
        epoch = unit.train_progress.num_epochs_completed
        if epoch % self.save_every != 0:
            return
        writer = self.tb_logger.writer
        if writer is None:
            return

        # Use EMA weights if available, else live model.
        sample_model = (
            unit.swa_model.module
            if getattr(unit, "swa_model", None) is not None
            else unit._raw_module
        )
        was_training = sample_model.training
        sample_model.eval()
        with torch.no_grad():
            noise = torch.randn(self.n_samples, *self.latent_shape, device=unit.device)
            if self.p_uncond is not None and hasattr(sample_model, "num_classes"):
                labels = torch.randint(
                    0,
                    sample_model.num_classes,
                    (self.n_samples,),
                    device=unit.device,
                )
                samples = guided_euler_sample(
                    sample_model, noise, self.num_steps, labels, self.guidance_scale
                )
            else:
                samples = euler_sample(sample_model, noise, self.num_steps)
        if was_training:
            sample_model.train()
        grid = make_grid(samples.clamp(-1, 1) * 0.5 + 0.5, nrow=8)
        writer.add_image("samples/generated", grid, epoch)
