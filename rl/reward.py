"""Reward functions for Flow-GRPO."""

from __future__ import annotations

from typing import Protocol

import torch
import torch.nn.functional as F

from rl.classifier import load_classifier


class RewardFn(Protocol):
    def __call__(
        self, x_final: torch.Tensor, prompts: torch.Tensor
    ) -> torch.Tensor:
        """Return [B] reward."""
        ...


class TargetClassReward:
    """Reward = log p(target_class = prompt | x_final) under a frozen classifier."""

    def __init__(self, classifier_checkpoint: str, device: str = "mps"):
        self.device = torch.device(device)
        self.classifier = load_classifier(classifier_checkpoint, self.device)

    @torch.no_grad()
    def __call__(
        self, x_final: torch.Tensor, prompts: torch.Tensor
    ) -> torch.Tensor:
        logits = self.classifier(x_final.to(self.device))
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs.gather(1, prompts.to(self.device).view(-1, 1)).squeeze(1)
