"""Reward functions for Flow-GRPO."""

from __future__ import annotations

from typing import Protocol

import torch
import torch.nn.functional as F

from rl.classifier import load_classifier
from rl.compression import jpeg_bpp_for_sample


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


class FixedClassReward:
    """Reward = log p(target_class | x_final), ignoring the conditioning prompt."""

    def __init__(
        self,
        classifier_checkpoint: str,
        target_class: int,
        device: str = "mps",
    ):
        self.device = torch.device(device)
        self.target_class = target_class
        self.classifier = load_classifier(classifier_checkpoint, self.device)

    @torch.no_grad()
    def __call__(
        self, x_final: torch.Tensor, prompts: torch.Tensor
    ) -> torch.Tensor:
        logits = self.classifier(x_final.to(self.device))
        log_probs = F.log_softmax(logits, dim=-1)
        targets = torch.full(
            (x_final.size(0), 1),
            self.target_class,
            dtype=torch.long,
            device=self.device,
        )
        return log_probs.gather(1, targets).squeeze(1)


class JpegCompressibilityReward:
    """Reward = negative JPEG bits-per-pixel for each final sample."""

    def __init__(
        self,
        quality: int = 75,
        optimize: bool = False,
        progressive: bool = False,
        subsampling: int | str | None = None,
    ):
        self.quality = quality
        self.optimize = optimize
        self.progressive = progressive
        self.subsampling = subsampling

    @torch.no_grad()
    def __call__(
        self, x_final: torch.Tensor, prompts: torch.Tensor
    ) -> torch.Tensor:
        rewards = [
            -jpeg_bpp_for_sample(
                sample,
                quality=self.quality,
                optimize=self.optimize,
                progressive=self.progressive,
                subsampling=self.subsampling,
            )
            for sample in x_final.detach().cpu()
        ]
        return torch.tensor(rewards, dtype=x_final.dtype, device=x_final.device)
