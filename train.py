"""NanoFlow training entry point.

Owns the `Trainer` class (model + optimizer + scheduler + scaler + EMA) and the
training loop. Hydra-decorated `main` runs training, then (if `cfg.inference` is
set) runs post-train sampling via `inference.run_inference`.
"""

from __future__ import annotations

import os
import signal
import sys
from typing import Any, Optional

import hydra
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import clip_grad_norm_
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torch.utils.data import DistributedSampler

import config as _config  # noqa: F401, registers structured config schema
from callbacks import (
    CheckpointCallback,
    EpochSummaryCallback,
    LRMonitorCallback,
    RunDirCallback,
    SampleLoggerCallback,
    StepLossCallback,
)
from flow import NoisePath


def setup_device_and_dist(device_type: str, distributed: Optional[str]) -> torch.device:
    """Resolve device and (optionally) initialize the distributed process group."""
    if device_type == "mps":
        return torch.device("mps")

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size > 1:
        if distributed != "ddp":
            raise ValueError(
                f"WORLD_SIZE={world_size} but distributed={distributed!r} "
                "(expected 'ddp')"
            )
        backend = "nccl" if device_type == "cuda" else "gloo"
        if not dist.is_initialized():
            dist.init_process_group(backend=backend)
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if device_type == "cuda":
            torch.cuda.set_device(local_rank)
            return torch.device(f"cuda:{local_rank}")
        return torch.device(device_type)

    return torch.device(device_type)


class Trainer:
    """Flow-matching trainer. Plain PyTorch with no framework dependency."""

    def __init__(
        self,
        *,
        model: nn.Module,
        flow: NoisePath,
        training,
        device: torch.device,
        distributed: Optional[str] = None,
    ):
        self.device = device
        self.training = training
        self.flow = flow

        self.raw_module = model.to(device)

        if training.p_uncond is not None and not hasattr(self.raw_module, "null_token"):
            raise ValueError(
                "p_uncond is set but model has no null_token. "
                "Use a class-conditioned model (ClassCondMLP, ClassCondUNet)."
            )

        if distributed == "ddp":
            ddp_kwargs = {}
            if device.type == "cuda":
                ddp_kwargs["device_ids"] = [device.index]
            self.module: nn.Module = DDP(self.raw_module, **ddp_kwargs)
        elif distributed is None:
            self.module = self.raw_module
        else:
            raise ValueError(f"Unknown distributed strategy: {distributed!r}")

        self.optimizer = torch.optim.Adam(self.raw_module.parameters(), lr=training.lr)
        self.lr_scheduler = self._build_scheduler(self.optimizer, training)

        ema_decay = getattr(training, "ema_decay", 0) or 0
        if ema_decay > 0:
            self.ema_model: Optional[AveragedModel] = AveragedModel(
                self.raw_module,
                multi_avg_fn=get_ema_multi_avg_fn(ema_decay),
            )
        else:
            self.ema_model = None

        self._amp, self._amp_dtype = self._resolve_precision(training.precision)
        if training.precision == "fp16" and device.type == "cuda":
            self.scaler: Optional[torch.amp.GradScaler] = torch.amp.GradScaler("cuda")
        else:
            self.scaler = None

        self.epoch = 0
        self.step = 0
        self.losses: list[float] = []
        self.train_loss_sum = 0.0
        self.train_loss_steps = 0
        self.val_loss_sum = 0.0
        self.val_loss_steps = 0

    @staticmethod
    def _build_scheduler(optimizer, training):
        warmup = getattr(training, "warmup_epochs", 0)
        if warmup <= 0:
            return None
        return torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            [
                torch.optim.lr_scheduler.LinearLR(
                    optimizer, start_factor=1e-3, total_iters=warmup
                ),
                torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=training.epochs - warmup
                ),
            ],
            milestones=[warmup],
        )

    @staticmethod
    def _resolve_precision(precision: Optional[str]):
        if precision == "fp16":
            return True, torch.float16
        if precision == "bf16":
            return True, torch.bfloat16
        return False, None

    @property
    def eval_model(self) -> nn.Module:
        """Module used for sampling/eval. EMA if enabled, else live weights."""
        if self.ema_model is not None:
            return self.ema_model.module
        return self.raw_module

    def _to_device(self, data):
        if isinstance(data, (tuple, list)):
            return tuple(
                d.to(self.device) if isinstance(d, torch.Tensor) else d for d in data
            )
        if isinstance(data, torch.Tensor):
            return data.to(self.device)
        return data

    def _compute_loss(self, batch) -> tuple[torch.Tensor, Any]:
        x_0 = batch[0]
        eps = torch.randn_like(x_0)
        t = torch.rand(x_0.size(0), *([1] * (x_0.dim() - 1)), device=x_0.device)
        xt = self.flow.interpolate(x_0, eps, t)
        if self.training.p_uncond is not None:
            cond = batch[1].clone()
            if self.training.p_uncond > 0:
                mask = torch.rand(cond.size(0), device=cond.device) < self.training.p_uncond
                cond[mask] = self.raw_module.null_token
            v_pred = self.module(xt, t.view(-1), cond)
        else:
            v_pred = self.module(xt, t.view(-1))
        vt = self.flow.target(x_0, eps, t)
        return F.mse_loss(v_pred, vt), v_pred

    def fit(self, train_loader, val_loader, callbacks=None) -> None:
        callbacks = callbacks or []
        for cb in callbacks:
            if hasattr(cb, "on_train_start"):
                cb.on_train_start(self)
        try:
            while self.epoch < self.training.epochs:
                for cb in callbacks:
                    if hasattr(cb, "on_train_epoch_start"):
                        cb.on_train_epoch_start(self)
                self._train_epoch(train_loader, callbacks)
                self.epoch += 1
                for cb in callbacks:
                    if hasattr(cb, "on_train_epoch_end"):
                        cb.on_train_epoch_end(self)

                if self.epoch % self.training.save_every == 0:
                    for cb in callbacks:
                        if hasattr(cb, "on_eval_epoch_start"):
                            cb.on_eval_epoch_start(self)
                    self._eval_epoch(val_loader)
                    for cb in callbacks:
                        if hasattr(cb, "on_eval_epoch_end"):
                            cb.on_eval_epoch_end(self)

                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()
        finally:
            for cb in callbacks:
                if hasattr(cb, "on_train_end"):
                    cb.on_train_end(self)

    def _train_epoch(self, loader, callbacks) -> None:
        self.module.train()
        if isinstance(loader.sampler, DistributedSampler):
            loader.sampler.set_epoch(self.epoch)
        for batch in loader:
            batch = self._to_device(batch)
            self.optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(
                device_type=self.device.type,
                dtype=self._amp_dtype,
                enabled=self._amp,
            ):
                loss, _ = self._compute_loss(batch)
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                if self.training.grad_clip > 0:
                    clip_grad_norm_(self.module.parameters(), self.training.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if self.training.grad_clip > 0:
                    clip_grad_norm_(self.module.parameters(), self.training.grad_clip)
                self.optimizer.step()
            if self.ema_model is not None:
                self.ema_model.update_parameters(self.raw_module)
            self.train_loss_sum += loss.item()
            self.train_loss_steps += 1
            self.step += 1
            for cb in callbacks:
                if hasattr(cb, "on_train_step_end"):
                    cb.on_train_step_end(self)

    @torch.no_grad()
    def _eval_epoch(self, loader) -> None:
        self.module.eval()
        if isinstance(loader.sampler, DistributedSampler):
            loader.sampler.set_epoch(self.epoch)
        for batch in loader:
            batch = self._to_device(batch)
            with torch.amp.autocast(
                device_type=self.device.type,
                dtype=self._amp_dtype,
                enabled=self._amp,
            ):
                loss, _ = self._compute_loss(batch)
            self.val_loss_sum += loss.item()
            self.val_loss_steps += 1

    def state_dict(self) -> dict:
        ckpt = {
            "model_state": self.raw_module.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "lr_scheduler_state": (
                self.lr_scheduler.state_dict() if self.lr_scheduler is not None else None
            ),
            "train_progress": {
                "num_epochs_completed": self.epoch,
                "num_steps_completed": self.step,
            },
            "losses": list(self.losses),
        }
        if self.ema_model is not None:
            ckpt["ema_state"] = self.ema_model.module.state_dict()
        if self.scaler is not None:
            ckpt["scaler_state"] = self.scaler.state_dict()
        return ckpt

    def load_state_dict(self, ckpt: dict) -> None:
        self.raw_module.load_state_dict(ckpt["model_state"])
        if ckpt.get("optimizer_state"):
            self.optimizer.load_state_dict(ckpt["optimizer_state"])
        if ckpt.get("lr_scheduler_state") and self.lr_scheduler is not None:
            self.lr_scheduler.load_state_dict(ckpt["lr_scheduler_state"])
        prog = ckpt.get("train_progress", {})
        self.epoch = prog.get("num_epochs_completed", 0)
        self.step = prog.get("num_steps_completed", 0)
        self.losses = list(ckpt.get("losses", []))
        if ckpt.get("ema_state") and self.ema_model is not None:
            self.ema_model.module.load_state_dict(ckpt["ema_state"])
        if ckpt.get("scaler_state") and self.scaler is not None:
            self.scaler.load_state_dict(ckpt["scaler_state"])


def _build_callbacks(cfg, run_dir_cb: RunDirCallback) -> list:
    writer = run_dir_cb.writer
    ckpt_cb = CheckpointCallback(
        ckpt_dir=run_dir_cb.ckpt_dir,
        save_every=cfg.training.save_every,
        resume=cfg.training.resume,
    )
    callbacks: list = [
        run_dir_cb,
        ckpt_cb,
        EpochSummaryCallback(
            writer=writer,
            total_epochs=cfg.training.epochs,
            batch_size=cfg.training.batch_size,
        ),
        StepLossCallback(writer=writer, log_every=cfg.training.log_every),
        LRMonitorCallback(writer=writer),
    ]
    if cfg.get("sample_logger") is not None:
        scfg = cfg.sample_logger
        callbacks.append(
            SampleLoggerCallback(
                writer=writer,
                save_every=cfg.training.save_every,
                latent_shape=list(scfg.latent_shape),
                n_samples=scfg.n_samples,
                num_steps=scfg.num_steps,
                guidance_scale=OmegaConf.select(scfg, "guidance_scale", default=1.0),
                p_uncond=cfg.training.p_uncond,
            )
        )
    return callbacks


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg) -> None:
    device = setup_device_and_dist(cfg.device, cfg.distributed)

    trainer: Optional[Trainer] = None
    train_loader = None

    if cfg.trainer is not None:
        trainer = hydra.utils.instantiate(cfg.trainer, device=device)
        train_loader = hydra.utils.instantiate(cfg.train_loader)
        val_loader = hydra.utils.instantiate(cfg.val_loader)

        run_dir_cb = RunDirCallback(
            runs_dir=cfg.runs_dir,
            run_prefix=cfg.training.run_prefix,
            cfg=cfg,
        )
        callbacks = _build_callbacks(cfg, run_dir_cb)
        ckpt_cb = next(cb for cb in callbacks if isinstance(cb, CheckpointCallback))

        def _handler(sig, frame):
            print(
                f"\nSIGTERM caught, saving preempted checkpoint to "
                f"{ckpt_cb.save_path('preempted')}"
            )
            ckpt_cb.save(trainer, "preempted")
            sys.exit(0)

        signal.signal(signal.SIGTERM, _handler)

        trainer.fit(train_loader, val_loader, callbacks=callbacks)

    if cfg.inference is not None and int(os.environ.get("RANK", 0)) == 0:
        from inference import run_inference

        icfg = cfg.inference
        if trainer is not None:
            sampler = hydra.utils.instantiate(
                icfg.sampler,
                model=trainer.eval_model,
                checkpoint=None,
            )
        else:
            sampler = hydra.utils.instantiate(icfg.sampler)

        train_data = getattr(train_loader, "dataset", None) if train_loader else None
        run_dir = run_dir_cb.run_dir if trainer is not None else None
        run_inference(cfg, sampler, run_dir=run_dir, train_data=train_data)

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
