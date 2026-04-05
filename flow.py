"""Flow matching core — interpolation and velocity target."""

import torch


def interpolate(x_0, eps, t):
    """
    Compute x_t along the CondOT path.

    x_0: (B, *shape) — data samples
    eps: (B, *shape) — noise ~ N(0, I)
    t:   (B, 1, ...) — time in [0, 1], broadcast-ready

    At t=0: x_t = eps (pure noise)
    At t=1: x_t = x_0 (clean data)
    """
    return (1 - t) * eps + t * x_0


def target_velocity(x_0, eps):
    """
    Ground-truth velocity for flow matching.

    d/dt[(1-t)*eps + t*x_0] = x_0 - eps
    """
    return x_0 - eps
