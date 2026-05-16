"""Rollout client interface for the GRPO trainer.

Defines the `RolloutClient` Protocol (`rollout`, `update_weights`) and the
in-process implementation that runs the SDE sampler synchronously.
"""

from __future__ import annotations

from typing import Protocol

import torch
import torch.nn as nn

from config import SamplerConfig
from rl.sde_sampler import RolloutBatch, sde_rollout


class RolloutClient(Protocol):
    def rollout(self, prompts: torch.Tensor) -> RolloutBatch: ...
    def update_weights(self, state_dict: dict) -> None: ...


class InProcessRolloutClient:
    """Synchronous, same-process, same-GPU. Trivial wrapper around sde_rollout."""

    def __init__(
        self,
        model: nn.Module,
        sampler: SamplerConfig,
        device: str,
        latent_shape: list[int],
    ):
        self.model = model
        self.sampler = sampler
        self.device = torch.device(device)
        self.latent_shape = tuple(latent_shape)

    def rollout(self, prompts: torch.Tensor) -> RolloutBatch:
        was_training = self.model.training
        self.model.eval()
        try:
            return sde_rollout(
                self.model, prompts.to(self.device),
                self.sampler, self.latent_shape, self.device,
            )
        finally:
            if was_training:
                self.model.train()

    def update_weights(self, state_dict: dict) -> None:
        # No-op: trainer and rollout share the same nn.Module here.
        return
