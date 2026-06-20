import unittest

import torch

from ode_solvers import EulerSolver, HeunSolver


class ODESolverTest(unittest.TestCase):
    def test_euler_and_heun_step_linear_velocity(self):
        x = torch.tensor([1.0])
        t = torch.tensor(0.0)
        dt = 1.0

        def velocity_fn(x_t, t_t):
            return x_t + t_t * 0

        torch.testing.assert_close(
            EulerSolver().step(x, t, dt, velocity_fn),
            torch.tensor([2.0]),
        )
        torch.testing.assert_close(
            HeunSolver().step(x, t, dt, velocity_fn),
            torch.tensor([2.5]),
        )

    def test_heun_uses_t_plus_dt_without_clamping(self):
        seen_times = []

        def velocity_fn(x_t, t_t):
            seen_times.append(float(t_t.item()))
            return torch.zeros_like(x_t)

        HeunSolver().step(
            torch.tensor([1.0]),
            torch.tensor(0.75),
            0.5,
            velocity_fn,
        )

        self.assertEqual(seen_times, [0.75, 1.25])


if __name__ == "__main__":
    unittest.main()
