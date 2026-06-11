"""NanoFlow training entry point.

Owns the `Trainer` class (model + optimizer + scheduler + scaler + EMA) and the
training loop. Hydra-decorated `main` runs training, then (if `cfg.inference` is
set) runs post-train sampling via `inference.run_inference`.
"""

from __future__ import annotations

import math
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
from config import LossMode
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
    if world_size > 1 or distributed == "ddp":
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
        self.lr_scheduler = None

        ema_decay = getattr(training, "ema_decay", 0) or 0
        if ema_decay > 0:
            self.ema_model: Optional[AveragedModel] = AveragedModel(
                self.raw_module,
                multi_avg_fn=get_ema_multi_avg_fn(ema_decay),
            )
        else:
            self.ema_model = None

        if training.max_steps is not None and training.max_steps <= 0:
            raise ValueError("training.max_steps must be positive or null")

        self._amp, self._amp_dtype = self._resolve_precision(training.precision)
        if training.precision == "fp16" and device.type == "cuda":
            self.scaler: Optional[torch.amp.GradScaler] = torch.amp.GradScaler("cuda")
        else:
            self.scaler = None

        self.epoch = 0
        self.step = 0
        self.losses: list[float] = []
        self._reset_train_epoch_metrics()
        self._reset_val_epoch_metrics()

    @staticmethod
    def _build_scheduler(optimizer, training, steps_per_epoch: int):
        warmup_epochs = getattr(training, "warmup_epochs", 0)
        if warmup_epochs <= 0:
            return None
        total_steps = training.epochs * steps_per_epoch
        if total_steps <= 0:
            return None
        warmup_steps = min(warmup_epochs * steps_per_epoch, total_steps)
        start_factor = 1e-3

        def lr_lambda(step: int) -> float:
            if warmup_steps > 0 and step < warmup_steps:
                progress = step / max(warmup_steps, 1)
                return start_factor + (1.0 - start_factor) * progress
            progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            progress = min(max(progress, 0.0), 1.0)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    @staticmethod
    def _resolve_precision(precision: Optional[str]):
        if precision == "fp16":
            return True, torch.float16
        if precision == "bf16":
            return True, torch.bfloat16
        return False, None

    def _reset_train_epoch_metrics(self) -> None:
        self.last_train_loss = 0.0
        self.train_loss_sum = 0.0
        self.train_loss_steps = 0
        self.train_loss_samples = 0

    def _reset_val_epoch_metrics(self) -> None:
        self.val_loss_sum = 0.0
        self.val_loss_steps = 0
        self.val_loss_samples = 0

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

    def _labels_for_batch(self, batch):
        if self.training.p_uncond is None:
            return None
        cond = batch[1].clone()
        if self.training.p_uncond > 0:
            mask = torch.rand(cond.size(0), device=cond.device) < self.training.p_uncond
            cond[mask] = self.raw_module.null_token
        return cond

    def _forward_model(
        self,
        xt: torch.Tensor,
        t: torch.Tensor,
        labels: Optional[torch.Tensor],
        *,
        return_aux: bool,
    ):
        args = (xt, t.view(-1)) if labels is None else (xt, t.view(-1), labels)
        if return_aux:
            try:
                return self.module(*args, return_aux=True)
            except TypeError as exc:
                raise ValueError(
                    "training.loss_mode=masked_mse requires model forward "
                    "to accept return_aux=True"
                ) from exc
        return self.module(*args)

    def _has_active_training_masker(self) -> bool:
        return bool(
            self.raw_module.training
            and getattr(self.raw_module, "masker", None) is not None
        )

    @staticmethod
    def _masked_mse(pred: torch.Tensor, target: torch.Tensor, loss_mask: torch.Tensor):
        loss_mask = loss_mask.to(device=pred.device, dtype=pred.dtype)
        try:
            expanded_mask = loss_mask.expand_as(pred)
        except RuntimeError as exc:
            raise ValueError(
                "loss_mask must be broadcastable to prediction shape, got "
                f"{tuple(loss_mask.shape)} and {tuple(pred.shape)}"
            ) from exc
        squared = (pred - target).pow(2) * expanded_mask
        denom = expanded_mask.sum().clamp_min(1)
        return squared.sum() / denom

    @staticmethod
    def _resolve_loss_mode(loss_mode) -> LossMode:
        if isinstance(loss_mode, LossMode):
            return loss_mode
        try:
            return LossMode(loss_mode)
        except ValueError as exc:
            allowed = ", ".join(mode.value for mode in LossMode)
            raise ValueError(
                f"Unknown training.loss_mode: {loss_mode!r}. Allowed: {allowed}"
            ) from exc

    def _compute_loss(self, batch) -> tuple[torch.Tensor, Any]:
        x_0 = batch[0]
        eps = torch.randn_like(x_0)
        t = torch.rand(x_0.size(0), *([1] * (x_0.dim() - 1)), device=x_0.device)
        xt = self.flow.interpolate(x_0, eps, t)
        labels = self._labels_for_batch(batch)
        vt = self.flow.target(x_0, eps, t)
        loss_mode = self._resolve_loss_mode(
            getattr(self.training, "loss_mode", LossMode.mse)
        )

        if loss_mode is LossMode.mse:
            if self._has_active_training_masker():
                raise ValueError(
                    "training.loss_mode=mse cannot be used while the model has an "
                    "active masker in training mode"
                )
            v_pred = self._forward_model(xt, t, labels, return_aux=False)
            if isinstance(v_pred, dict):
                raise ValueError("training.loss_mode=mse requires tensor model output")
            return F.mse_loss(v_pred, vt), v_pred

        if loss_mode is LossMode.masked_mse:
            output = self._forward_model(xt, t, labels, return_aux=True)
            if not isinstance(output, dict):
                raise ValueError(
                    "training.loss_mode=masked_mse requires dict output with "
                    "pred and loss_mask"
                )
            if "pred" not in output or "loss_mask" not in output:
                raise ValueError(
                    "training.loss_mode=masked_mse requires output keys pred and loss_mask"
                )
            pred = output["pred"]
            loss = self._masked_mse(pred, vt, output["loss_mask"])
            return loss, pred

        raise AssertionError(f"Unhandled training.loss_mode: {loss_mode!r}")

    def fit(self, train_loader, val_loader, callbacks=None) -> None:
        callbacks = callbacks or []
        if self.lr_scheduler is None:
            self.lr_scheduler = self._build_scheduler(
                self.optimizer,
                self.training,
                steps_per_epoch=len(train_loader),
            )
        start_callbacks = sorted(
            callbacks, key=lambda cb: 0 if isinstance(cb, CheckpointCallback) else 1
        )
        for cb in start_callbacks:
            if hasattr(cb, "on_train_start"):
                cb.on_train_start(self)
        completed = False
        try:
            while self.epoch < self.training.epochs and not self._reached_max_steps():
                for cb in callbacks:
                    if hasattr(cb, "on_train_epoch_start"):
                        cb.on_train_epoch_start(self)
                reached_max_steps = self._train_epoch(train_loader, callbacks)
                self.epoch += 1
                for cb in callbacks:
                    if hasattr(cb, "on_train_epoch_end"):
                        cb.on_train_epoch_end(self)

                if (
                    self.training.eval_every > 0
                    and self.epoch % self.training.eval_every == 0
                ):
                    for cb in callbacks:
                        if hasattr(cb, "on_eval_epoch_start"):
                            cb.on_eval_epoch_start(self)
                    self._eval_epoch(val_loader)
                    for cb in callbacks:
                        if hasattr(cb, "on_eval_epoch_end"):
                            cb.on_eval_epoch_end(self)

                if reached_max_steps:
                    break
            completed = True
        finally:
            try:
                if completed:
                    for cb in callbacks:
                        if hasattr(cb, "on_train_end"):
                            cb.on_train_end(self)
            finally:
                for cb in callbacks:
                    if hasattr(cb, "on_train_cleanup"):
                        cb.on_train_cleanup(self)

    def _reached_max_steps(self) -> bool:
        max_steps = self.training.max_steps
        return max_steps is not None and self.step >= max_steps

    def _train_epoch(self, loader, callbacks) -> bool:
        self._reset_train_epoch_metrics()
        self.module.train()
        if isinstance(loader.sampler, DistributedSampler):
            loader.sampler.set_epoch(self.epoch)
        for batch in loader:
            if self._reached_max_steps():
                return True
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
            batch_size = int(batch[0].size(0))
            self.last_train_loss = loss.item()
            self.train_loss_sum += self.last_train_loss * batch_size
            self.train_loss_steps += 1
            self.train_loss_samples += batch_size
            self.step += 1
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            for cb in callbacks:
                if hasattr(cb, "on_train_step_end"):
                    cb.on_train_step_end(self)
            if self._reached_max_steps():
                return True
        return False

    @torch.no_grad()
    def _eval_epoch(self, loader) -> None:
        self._reset_val_epoch_metrics()
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
            batch_size = int(batch[0].size(0))
            self.val_loss_sum += loss.item() * batch_size
            self.val_loss_steps += 1
            self.val_loss_samples += batch_size

    def state_dict(self) -> dict:
        ckpt = {
            "model_state": self.raw_module.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "lr_scheduler_state": (
                self.lr_scheduler.state_dict()
                if self.lr_scheduler is not None
                else None
            ),
            "train_progress": {
                "num_epochs_completed": self.epoch,
                "num_steps_completed": self.step,
            },
            "losses": list(self.losses),
        }
        if self.ema_model is not None:
            ckpt["ema_state"] = self.ema_model.module.state_dict()
            ckpt["ema_n_averaged"] = int(self.ema_model.n_averaged.item())
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
            ema_n = int(ckpt.get("ema_n_averaged", 1))
            self.ema_model.n_averaged.fill_(ema_n)
        if ckpt.get("scaler_state") and self.scaler is not None:
            self.scaler.load_state_dict(ckpt["scaler_state"])

    def load_model_weights(self, ckpt: dict, weights: str = "raw") -> None:
        weights_value = weights.value if hasattr(weights, "value") else weights
        if weights_value == "raw":
            state = ckpt["model_state"]
        elif weights_value == "ema":
            state = ckpt.get("ema_state")
            if state is None:
                raise ValueError(
                    "init_from_weights=ema requested but checkpoint has no EMA state"
                )
        else:
            raise ValueError("training.init_from_weights must be 'raw' or 'ema'")
        self.raw_module.load_state_dict(state)
        self.epoch = 0
        self.step = 0
        self.losses = []
        if self.ema_model is not None:
            self.ema_model.module.load_state_dict(self.raw_module.state_dict())
            self.ema_model.n_averaged.zero_()


def _build_callbacks(cfg, run_dir_cb: RunDirCallback) -> list:
    writer = run_dir_cb.writer
    if cfg.training.resume == "auto" and not run_dir_cb.stable_run_dir:
        raise ValueError("training.resume=auto requires training.run_dir")
    ckpt_cb = CheckpointCallback(
        ckpt_dir=run_dir_cb.ckpt_dir,
        checkpoint_every=cfg.training.checkpoint_every,
        resume=cfg.training.resume,
        init_from=OmegaConf.select(cfg, "training.init_from", default=None),
        init_from_weights=OmegaConf.select(
            cfg, "training.init_from_weights", default="raw"
        ),
    )
    callbacks: list = [
        run_dir_cb,
        EpochSummaryCallback(
            writer=writer,
            total_epochs=cfg.training.epochs,
            batch_size=cfg.training.batch_size,
        ),
        ckpt_cb,
        StepLossCallback(writer=writer, log_every=cfg.training.log_every),
        LRMonitorCallback(writer=writer, log_every=cfg.training.log_every),
    ]
    if cfg.get("sample_logger") is not None:
        scfg = cfg.sample_logger
        callbacks.append(
            SampleLoggerCallback(
                writer=writer,
                every=cfg.training.eval_every,
                latent_shape=list(scfg.latent_shape),
                n_samples=scfg.n_samples,
                num_steps=scfg.num_steps,
                guidance_scale=OmegaConf.select(scfg, "guidance_scale", default=1.0),
                p_uncond=cfg.training.p_uncond,
                vae_cfg=OmegaConf.select(cfg, "vae", default=None),
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

        if cfg.training.resume == "auto" and cfg.training.run_dir is None:
            raise ValueError("training.resume=auto requires training.run_dir")

        run_dir_cb = RunDirCallback(
            runs_dir=cfg.runs_dir,
            run_prefix=cfg.training.run_prefix,
            cfg=cfg,
            run_dir=cfg.training.run_dir,
        )
        callbacks = _build_callbacks(cfg, run_dir_cb)

        def _handler(sig, frame):
            print("\nSIGTERM caught, preserving latest epoch checkpoint")
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
