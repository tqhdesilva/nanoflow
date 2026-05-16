"""SDE rollout for Flow-GRPO.

Converts the deterministic rectified-flow ODE into an SDE so each integration
step is a Gaussian policy. Records per-step samples, means, and log-probs so
the loss can be evaluated and log-probs recomputed under a new policy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from config import SamplerConfig


@dataclass
class RolloutBatch:
    xs: torch.Tensor          # [T+1, B, C, H, W] trajectory
    actions: torch.Tensor     # [T, B, C, H, W] iid N(0, I) samples used at each step
    log_probs_old: torch.Tensor  # [T, B]
    means: torch.Tensor       # [T, B, C, H, W] mu_k under theta_old
    stds: torch.Tensor        # [T] sigma_k_step (data-independent)
    prompts: torch.Tensor     # [B]
    ts: torch.Tensor          # [T+1] timestep grid


def _make_grid(t_min: float, t_max: float, T: int, device: torch.device) -> torch.Tensor:
    return torch.linspace(t_min, t_max, T + 1, device=device)


def _sigma_t(t: torch.Tensor, sigma_a: float) -> torch.Tensor:
    # sigma_t = a * sqrt(t / (1 - t))
    return sigma_a * torch.sqrt(t / (1.0 - t))


def cfg_velocity(
    model: nn.Module, x: torch.Tensor, t_scalar: torch.Tensor,
    cond: torch.Tensor, guidance_scale: float,
) -> torch.Tensor:
    """CFG-mixed velocity: v_uncond + g * (v_cond - v_uncond).

    t_scalar: scalar tensor; broadcast to batch.
    """
    t_batch = t_scalar.expand(x.size(0))
    null_cond = torch.full_like(cond, model.null_token)
    v_cond = model(x, t_batch, cond)
    v_uncond = model(x, t_batch, null_cond)
    return v_uncond + guidance_scale * (v_cond - v_uncond)


def _gaussian_logprob(
    x: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor,
) -> torch.Tensor:
    """Isotropic Gaussian log-prob, summed over non-batch dims.

    x, mu: [B, ...]; sigma: scalar tensor (data-independent per step).
    """
    B = x.size(0)
    diff = (x - mu).view(B, -1)
    D = diff.shape[1]
    var = sigma * sigma
    sq = (diff * diff).sum(dim=1)
    return -0.5 * sq / var - D * torch.log(sigma) - 0.5 * D * math.log(2 * math.pi)


def _drift(
    x: torch.Tensor, v: torch.Tensor, t: torch.Tensor, sigma_t: torch.Tensor
) -> torch.Tensor:
    """Drift term in the SDE: v + sigma_t^2 / (2 t) * (x + (1 - t) v).

    x_{k+1} = x_k + drift * dt + sigma_t * sqrt(dt) * eps.
    """
    coef = (sigma_t * sigma_t) / (2.0 * t)
    return v + coef * (x + (1.0 - t) * v)


@torch.no_grad()
def sde_rollout(
    model: nn.Module,
    prompts: torch.Tensor,
    sampler: SamplerConfig,
    latent_shape: tuple[int, ...],
    device: torch.device,
    generator: Optional[torch.Generator] = None,
) -> RolloutBatch:
    """Run a T-step SDE rollout under the current policy.

    Returns a RolloutBatch with xs, actions, log_probs_old, means, stds, prompts.
    """
    B = prompts.size(0)
    T = sampler.T_rollout
    ts = _make_grid(sampler.t_min, sampler.t_max, T, device)

    x = torch.randn(B, *latent_shape, device=device, generator=generator)
    xs = [x]
    actions = []
    means = []
    log_probs = []
    stds_list = []

    for k in range(T):
        t_k = ts[k]
        dt = ts[k + 1] - ts[k]
        sigma_t = _sigma_t(t_k, sampler.sigma_a)
        sigma_step = sigma_t * torch.sqrt(dt)
        stds_list.append(sigma_step)

        v = cfg_velocity(model, x, t_k, prompts, sampler.guidance_scale)
        mu = x + _drift(x, v, t_k, sigma_t) * dt

        eps = torch.randn(B, *latent_shape, device=device, generator=generator)
        x_next = mu + sigma_step * eps

        logp = _gaussian_logprob(x_next, mu, sigma_step)

        xs.append(x_next)
        actions.append(eps)
        means.append(mu)
        log_probs.append(logp)
        x = x_next

    return RolloutBatch(
        xs=torch.stack(xs, dim=0),
        actions=torch.stack(actions, dim=0),
        log_probs_old=torch.stack(log_probs, dim=0),
        means=torch.stack(means, dim=0),
        stds=torch.stack(stds_list, dim=0),
        prompts=prompts,
        ts=ts,
    )


def recompute_logprobs(
    model: nn.Module,
    traj: RolloutBatch,
    sampler: SamplerConfig,
    no_grad: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Recompute (log_probs, means) for the saved trajectory under `model`.

    sigma is data-independent so it stays fixed from the rollout. Only mu_k
    depends on theta, and we evaluate the Gaussian density of the saved
    x_{k+1} under the new mu_k.

    Returns:
        log_probs: [T, B]
        means:     [T, B, C, H, W]
    """
    T = traj.xs.shape[0] - 1
    log_probs = []
    means = []
    ctx = torch.no_grad() if no_grad else torch.enable_grad()
    with ctx:
        for k in range(T):
            x_k = traj.xs[k]
            x_next = traj.xs[k + 1]
            t_k = traj.ts[k]
            dt = traj.ts[k + 1] - traj.ts[k]
            sigma_t = _sigma_t(t_k, sampler.sigma_a)
            sigma_step = traj.stds[k]

            v = cfg_velocity(model, x_k, t_k, traj.prompts, sampler.guidance_scale)
            mu = x_k + _drift(x_k, v, t_k, sigma_t) * dt

            logp = _gaussian_logprob(x_next, mu, sigma_step)
            log_probs.append(logp)
            means.append(mu)

    return torch.stack(log_probs, dim=0), torch.stack(means, dim=0)
