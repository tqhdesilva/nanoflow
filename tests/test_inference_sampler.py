import unittest

import torch

from inference import FlowSampler
from ode_solvers import EulerSolver


class RecordingEulerSolver(EulerSolver):
    def __init__(self):
        self.times = []

    def step(self, x, t, dt, velocity_fn):
        self.times.append(float(t.item()))
        return super().step(x, t, dt, velocity_fn)


class ZeroVelocityModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))

    def forward(self, x, t, labels=None):
        return torch.zeros_like(x) + self.anchor * 0


class FlowSamplerTest(unittest.TestCase):
    def test_generate_impl_uses_endpoint_excluded_time_grid(self):
        solver = RecordingEulerSolver()
        sampler = FlowSampler(
            ZeroVelocityModel(),
            num_steps=4,
            latent_shape=[1],
            device="cpu",
            solver=solver,
        )

        sampler.generate_impl(torch.zeros(1, 1))

        self.assertEqual(solver.times, [0.0, 0.25, 0.5, 0.75])


if __name__ == "__main__":
    unittest.main()
