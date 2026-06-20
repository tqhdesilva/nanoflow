"""Generic ODE solvers for flow-matching sampling."""

from __future__ import annotations

from typing import Callable, Protocol

import torch

VelocityFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


class LatentODESolver(Protocol):
    """One-step ODE solver interface for latent flow sampling."""

    name: str

    def step(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        dt: float,
        velocity_fn: VelocityFn,
    ) -> torch.Tensor:
        """Advance one ODE step."""
        ...


class EulerSolver:
    """First-order Euler solver for $dx/dt = v(x, t)$."""

    name = "euler"

    def step(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        dt: float,
        velocity_fn: VelocityFn,
    ) -> torch.Tensor:
        return x + velocity_fn(x, t) * dt


class HeunSolver:
    """Second-order predictor-corrector solver for $dx/dt = v(x, t)$."""

    name = "heun"

    def step(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        dt: float,
        velocity_fn: VelocityFn,
    ) -> torch.Tensor:
        v0 = velocity_fn(x, t)
        x_predict = x + v0 * dt
        v1 = velocity_fn(x_predict, t + dt)
        return x + 0.5 * (v0 + v1) * dt
