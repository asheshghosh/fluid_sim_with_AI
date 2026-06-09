import unittest

import numpy as np

from fluid_ai_sim.data import generate_velocity_trajectories, make_transition_pairs
from fluid_ai_sim.incompressible import (
    SpectralIncompressibleNavierStokes2D,
    VelocitySolverConfig,
    random_divergence_free_velocity,
)


class IncompressibleSolverTests(unittest.TestCase):
    def test_random_velocity_is_divergence_free_and_normalized(self):
        solver = SpectralIncompressibleNavierStokes2D(VelocitySolverConfig(n=32))
        velocity = random_divergence_free_velocity(32, seed=123)

        self.assertEqual(velocity.shape, (2, 32, 32))
        self.assertAlmostEqual(float(np.sqrt(np.mean(np.sum(velocity * velocity, axis=0)))), 1.0, places=12)
        self.assertLess(float(np.max(np.abs(solver.divergence(velocity)))), 1.0e-10)

    def test_projection_step_returns_finite_divergence_free_velocity(self):
        solver = SpectralIncompressibleNavierStokes2D(VelocitySolverConfig(n=32, viscosity=1.0e-2, dt=5.0e-3))
        velocity = random_divergence_free_velocity(32, seed=2)

        next_velocity = solver.step(velocity)

        self.assertEqual(next_velocity.shape, velocity.shape)
        self.assertTrue(np.all(np.isfinite(next_velocity)))
        self.assertLess(float(np.max(np.abs(solver.divergence(next_velocity)))), 1.0e-10)

    def test_projection_removes_divergence_from_arbitrary_velocity(self):
        solver = SpectralIncompressibleNavierStokes2D(VelocitySolverConfig(n=32))
        raw = np.random.default_rng(4).normal(size=(2, 32, 32))

        projected = solver.project_velocity(raw)

        self.assertLess(float(np.max(np.abs(solver.divergence(projected)))), 1.0e-10)

    def test_viscosity_reduces_kinetic_energy_over_short_rollout(self):
        solver = SpectralIncompressibleNavierStokes2D(VelocitySolverConfig(n=32, viscosity=5.0e-2, dt=2.0e-3))
        velocity = random_divergence_free_velocity(32, seed=3)

        start = solver.diagnostics(velocity)["kinetic_energy"]
        end = solver.diagnostics(solver.rollout(velocity, steps=10)[-1])["kinetic_energy"]

        self.assertLess(end, start)

    def test_velocity_trajectory_pairs_keep_channel_dimension(self):
        config = VelocitySolverConfig(n=16, viscosity=1.0e-2, dt=2.0e-3)
        trajectories = generate_velocity_trajectories(config, trajectories=2, steps=4, keep_every=2)

        x, y = make_transition_pairs(trajectories)

        self.assertEqual(trajectories.shape, (2, 3, 2, 16, 16))
        self.assertEqual(x.shape, (4, 2, 16, 16))
        self.assertEqual(y.shape, (4, 2, 16, 16))


if __name__ == "__main__":
    unittest.main()
