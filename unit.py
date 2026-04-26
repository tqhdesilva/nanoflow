"""FlowMatchingUnit — TorchTNT AutoUnit subclass for flow matching training."""

import os
from typing import Any, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler
from torchtnt.framework import AutoPredictUnit, AutoUnit, State
from torchtnt.framework.auto_unit import SWAParams
from torchtnt.utils.prepare_module import DDPStrategy, FSDPStrategy

from flow import NoisePath


_STRATEGIES = {"ddp": DDPStrategy, "fsdp": FSDPStrategy}


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


@torch.no_grad()
def euler_sample(model, noise, num_steps):
    """Generate samples via Euler integration of the learned velocity field."""
    xt = noise
    dt = 1.0 / num_steps
    for t_val in torch.linspace(0, 1, num_steps, device=noise.device):
        vt = model(xt, t_val.expand(xt.size(0)))
        xt = xt + vt * dt
    return xt


@torch.no_grad()
def guided_euler_sample(model, noise, num_steps, cond, guidance_scale):
    """Euler sampling with classifier-free guidance."""
    null_cond = torch.full_like(cond, model.null_token)
    xt = noise
    dt = 1.0 / num_steps
    for t_val in torch.linspace(0, 1, num_steps, device=noise.device):
        t_batch = t_val.expand(xt.size(0))
        v_cond = model(xt, t_batch, cond)
        v_uncond = model(xt, t_batch, null_cond)
        vt = v_uncond + guidance_scale * (v_cond - v_uncond)
        xt = xt + vt * dt
    return xt


# ---------------------------------------------------------------------------
# DataLoader
# ---------------------------------------------------------------------------


def build_dataloader(dataset, batch_size, num_workers, train, **_):
    """Build a (possibly distributed) DataLoader.

    `dataset` is a Hydra partial — called here with `train=...`. When WORLD_SIZE>1
    (set by torchrun), wrap in a DistributedSampler so each rank sees a disjoint shard.
    """
    ds = dataset(train=train)
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    sampler = None
    shuffle = train
    if world_size > 1:
        sampler = DistributedSampler(
            ds, num_replicas=world_size, rank=rank, shuffle=train
        )
        shuffle = False
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=train,
    )


# ---------------------------------------------------------------------------
# FlowMatchingUnit
# ---------------------------------------------------------------------------


class FlowMatchingUnit(AutoUnit):
    """Flow matching training. Implements only the methods AutoUnit's docs call out:
    `compute_loss`, `configure_optimizers_and_lr_scheduler`, `on_train_step_end`,
    `on_eval_step_end`. `move_data_to_device` and `_prefetch_next_batch` are
    overridden only to work around the MPS prefetch bug in upstream AutoUnit.
    """

    def __init__(
        self,
        *,
        model: torch.nn.Module,
        flow: NoisePath,
        training,
        device: torch.device,
        distributed: Optional[str] = None,
    ):
        ema_decay = getattr(training, "ema_decay", 0) or 0
        swa_params = (
            SWAParams(
                warmup_steps_or_epochs=0,
                step_or_epoch_update_freq=1,
                averaging_method="ema",
                ema_decay=ema_decay,
            )
            if ema_decay > 0
            else None
        )

        if distributed is not None and distributed not in _STRATEGIES:
            raise ValueError(f"Unknown distributed strategy: {distributed!r}")
        strategy = _STRATEGIES[distributed]() if distributed else None

        super().__init__(
            module=model,
            device=device,
            strategy=strategy,
            precision=training.precision,
            clip_grad_norm=training.grad_clip if training.grad_clip > 0 else None,
            step_lr_interval="epoch",
            swa_params=swa_params,
        )

        self.path = flow
        self.tcfg = training

        if training.p_uncond is not None and not hasattr(self._raw_module, "null_token"):
            raise ValueError(
                "p_uncond is set but model has no null_token. "
                "Use a class-conditioned model (ClassCondMLP, ClassCondUNet)."
            )

        # Loss accumulators read by callbacks at epoch boundaries.
        self.train_loss_sum = 0.0
        self.train_loss_steps = 0
        self.val_loss_sum = 0.0
        self.val_loss_steps = 0

    @property
    def _raw_module(self) -> torch.nn.Module:
        m = self.module
        return m.module if hasattr(m, "module") else m  # unwrap DDP

    # --- AutoUnit interface ---

    def configure_optimizers_and_lr_scheduler(self, module):
        optimizer = torch.optim.Adam(module.parameters(), lr=self.tcfg.lr)
        warmup = getattr(self.tcfg, "warmup_epochs", 0)
        if warmup > 0:
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                [
                    torch.optim.lr_scheduler.LinearLR(
                        optimizer, start_factor=1e-3, total_iters=warmup
                    ),
                    torch.optim.lr_scheduler.CosineAnnealingLR(
                        optimizer, T_max=self.tcfg.epochs - warmup
                    ),
                ],
                milestones=[warmup],
            )
        else:
            scheduler = None
        return optimizer, scheduler

    def compute_loss(self, state: State, data) -> Tuple[torch.Tensor, Any]:
        x_0 = data[0] if isinstance(data, (tuple, list)) else data
        eps = torch.randn_like(x_0)
        t = torch.rand(x_0.size(0), *([1] * (x_0.dim() - 1)), device=x_0.device)
        xt = self.path.interpolate(x_0, eps, t)
        if self.tcfg.p_uncond is not None:
            cond = data[1]
            if self.tcfg.p_uncond > 0:
                mask = torch.rand(cond.size(0), device=cond.device) < self.tcfg.p_uncond
                cond = cond.clone()
                cond[mask] = self._raw_module.null_token
            v_pred = self.module(xt, t.view(-1), cond)
        else:
            v_pred = self.module(xt, t.view(-1))
        vt = self.path.target(x_0, eps, t)
        loss = F.mse_loss(v_pred, vt)
        return loss, v_pred

    def on_train_step_end(self, state, data, step, results) -> None:
        self.train_loss_sum += results.loss.item()
        self.train_loss_steps += 1

    def on_eval_step_end(self, state, data, step, loss, outputs) -> None:
        self.val_loss_sum += loss.item()
        self.val_loss_steps += 1

    # MPS workaround: AutoPredictUnit / AutoUnit prefetch via torch.cuda.stream,
    # which crashes on MPS/CPU. Fall back to synchronous prefetch on non-cuda devices.
    def _prefetch_next_batch(self, state, data_iter):
        if self.device.type == "cuda":
            return super()._prefetch_next_batch(state, data_iter)
        try:
            next_batch = next(data_iter)
        except StopIteration:
            self._phase_to_next_batch[state.active_phase] = None
            self._is_last_batch = True
            return
        self._phase_to_next_batch[state.active_phase] = self.move_data_to_device(
            state, next_batch, non_blocking=False
        )

    def move_data_to_device(self, state: State, data, non_blocking: bool):
        dev = self.device
        if isinstance(data, (tuple, list)):
            return tuple(d.to(dev) if isinstance(d, torch.Tensor) else d for d in data)
        if isinstance(data, torch.Tensor):
            return data.to(dev)
        return data


# ---------------------------------------------------------------------------
# InferenceUnit
# ---------------------------------------------------------------------------


class InferenceUnit(AutoPredictUnit):
    """Loads checkpoint (preferring EMA weights), runs Euler integration."""

    def __init__(
        self,
        model,
        checkpoint: Optional[str] = None,
        num_steps: int = 100,
        latent_shape: Optional[list] = None,
        device: str = "cpu",
        class_sampler=None,
    ):
        dev = torch.device(device)
        model = model.to(dev)

        if checkpoint:
            ckpt = torch.load(checkpoint, weights_only=True, map_location=dev)
            state = ckpt.get("ema_state") or ckpt["model_state"]
            model.load_state_dict(state)
            tag = "EMA" if "ema_state" in ckpt and ckpt["ema_state"] else "model"
            epoch = ckpt.get("train_progress", {}).get("num_epochs_completed", "?")
            print(f"Loaded {tag} weights from {checkpoint} (epoch {epoch})")

        model.eval()
        super().__init__(module=model, device=dev)

        self.num_steps = num_steps
        self.latent_shape = latent_shape
        self.class_sampler = class_sampler
        self.results = []

    def _prefetch_next_batch(self, state, data_iter):
        if self.device.type == "cuda":
            return super()._prefetch_next_batch(state, data_iter)
        try:
            next_batch = next(data_iter)
        except StopIteration:
            self._phase_to_next_batch[state.active_phase] = None
            self._is_last_batch = True
            return
        self._phase_to_next_batch[state.active_phase] = self.move_data_to_device(
            state, next_batch, non_blocking=False
        )

    def predict_step(self, state: State, data: Tuple) -> torch.Tensor:
        n_samples = data[0].shape[0]
        shape = tuple(self.latent_shape) if self.latent_shape else (2,)
        noise = torch.randn(n_samples, *shape, device=self.device)
        if self.class_sampler is not None and len(data) > 1:
            cond = data[1].to(self.device)
            samples = guided_euler_sample(
                self.module,
                noise,
                self.num_steps,
                cond,
                self.class_sampler.guidance_scale,
            )
        else:
            samples = euler_sample(self.module, noise, self.num_steps)
        self.results.append(samples)
        return samples
