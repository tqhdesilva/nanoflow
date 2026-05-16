"""Noise paths: interpolation and training target.

Swappable interface: CondOT (flow matching) is the default.
Future paths (DDPM, VP-SDE) implement the same interface.
"""

from torch import Tensor


class NoisePath:
    """Base interface for noise paths."""

    def interpolate(self, x_0: Tensor, eps: Tensor, t: Tensor) -> Tensor:
        """Compute x_t given data x_0, noise eps, and time t."""
        raise NotImplementedError

    def target(self, x_0: Tensor, eps: Tensor, t: Tensor) -> Tensor:
        """Ground-truth prediction target for the model."""
        raise NotImplementedError


class CondOT(NoisePath):
    """Conditional Optimal Transport path (flow matching).

    x_t = (1-t)*eps + t*x_0      (linear interpolation)
    target = x_0 - eps           (constant velocity)
    """

    def interpolate(self, x_0, eps, t):
        return (1 - t) * eps + t * x_0

    def target(self, x_0, eps, t):
        return x_0 - eps
