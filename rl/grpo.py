"""Flow-GRPO loss: group-relative advantage + clipped IS surrogate + Gaussian KL."""

from __future__ import annotations

import torch


def compute_group_advantage(rewards: torch.Tensor, G: int) -> torch.Tensor:
    """Per-prompt z-scored advantage. rewards: [B*G]; returns [B*G].

    Assumes prompts are flattened as [c1] * G + [c2] * G + ... (repeat_interleave).
    """
    r = rewards.view(-1, G)
    adv = (r - r.mean(dim=1, keepdim=True)) / (r.std(dim=1, keepdim=True) + 1e-8)
    return adv.view(-1)


def gaussian_kl_mu(
    mu_new: torch.Tensor, mu_ref: torch.Tensor, sigma_step: torch.Tensor
) -> torch.Tensor:
    """KL(N(mu_new, sigma^2 I) || N(mu_ref, sigma^2 I)) = ||diff||^2 / (2 sigma^2).

    mu_*: [T, B, ...]; sigma_step: [T]. Returns scalar mean over T and B.
    """
    T, B = mu_new.shape[:2]
    diff = (mu_new - mu_ref).view(T, B, -1)
    sq = (diff * diff).sum(dim=-1)            # [T, B]
    var = (sigma_step * sigma_step).view(T, 1)
    return (sq / (2.0 * var)).mean()


def grpo_loss(
    new_logprobs: torch.Tensor,    # [T, B]
    old_logprobs: torch.Tensor,    # [T, B]
    new_mus: torch.Tensor,         # [T, B, ...]
    ref_mus: torch.Tensor,         # [T, B, ...]
    sigma_step: torch.Tensor,      # [T]
    advantage: torch.Tensor,       # [B]
    clip_eps: float,
    kl_beta: float,
) -> tuple[torch.Tensor, dict]:
    """Per-step clipped IS surrogate + Gaussian KL to reference policy."""
    log_ratio = new_logprobs - old_logprobs
    ratio = log_ratio.exp()
    A = advantage[None, :].expand_as(ratio)
    unclipped = ratio * A
    clipped = ratio.clamp(1 - clip_eps, 1 + clip_eps) * A
    pg_loss = -torch.min(unclipped, clipped).mean()

    kl = gaussian_kl_mu(new_mus, ref_mus, sigma_step)
    loss = pg_loss + kl_beta * kl

    with torch.no_grad():
        approx_kl_is = (ratio - 1 - log_ratio).mean()
        clip_frac = ((ratio - 1).abs() > clip_eps).float().mean()

    info = {
        "loss": loss.detach(),
        "pg_loss": pg_loss.detach(),
        "kl": kl.detach(),
        "approx_kl_is": approx_kl_is,
        "clip_frac": clip_frac,
        "ratio_mean": ratio.detach().mean(),
    }
    return loss, info
