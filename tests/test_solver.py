import unittest

import numpy as np

from fluid_ai_sim.solver import SolverConfig, SpectralNavierStokes2D, random_vorticity


class SolverTests(unittest.TestCase):
    def test_random_vorticity_is_normalized_and_zero_mean(self):
        omega = random_vorticity(32, seed=123)
        self.assertEqual(omega.shape, (32, 32))
        self.assertAlmostEqual(float(np.mean(omega)), 0.0, places=12)
        self.assertAlmostEqual(float(np.std(omega)), 1.0, places=12)

    def test_velocity_is_divergence_free(self):
        solver = SpectralNavierStokes2D(SolverConfig(n=32))
        omega = random_vorticity(32, seed=1)
        div = solver.divergence(omega)
        self.assertLess(float(np.max(np.abs(div))), 1.0e-10)

    def test_step_returns_finite_zero_mean_state(self):
        solver = SpectralNavierStokes2D(SolverConfig(n=32, viscosity=1.0e-2, dt=5.0e-3))
        omega = random_vorticity(32, seed=2)
        next_omega = solver.step(omega)
        self.assertEqual(next_omega.shape, omega.shape)
        self.assertTrue(np.all(np.isfinite(next_omega)))
        self.assertAlmostEqual(float(np.mean(next_omega)), 0.0, places=12)

    def test_viscosity_reduces_enstrophy_over_short_rollout(self):
        solver = SpectralNavierStokes2D(SolverConfig(n=32, viscosity=5.0e-2, dt=2.0e-3))
        omega = random_vorticity(32, seed=3)
        start = solver.diagnostics(omega)["enstrophy"]
        end = solver.diagnostics(solver.rollout(omega, steps=10)[-1])["enstrophy"]
        self.assertLess(end, start)


if __name__ == "__main__":
    unittest.main()
