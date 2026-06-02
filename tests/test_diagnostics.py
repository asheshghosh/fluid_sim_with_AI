import unittest

import numpy as np

from fluid_ai_sim.diagnostics import comparison_errors, diagnostics_to_table, trajectory_diagnostics
from fluid_ai_sim.solver import SolverConfig, SpectralNavierStokes2D, random_vorticity


class DiagnosticsTests(unittest.TestCase):
    def test_trajectory_diagnostics_are_tabular(self):
        config = SolverConfig(n=16, dt=1.0e-2)
        solver = SpectralNavierStokes2D(config)
        omega = random_vorticity(config.n, seed=4)
        trajectory = solver.rollout(omega, steps=3)

        diagnostics = trajectory_diagnostics(solver, trajectory, dt=config.dt)
        names, values = diagnostics_to_table(diagnostics)

        self.assertEqual(len(diagnostics), 4)
        self.assertEqual(values.shape[0], 4)
        self.assertEqual(values.shape[1], len(names))
        self.assertIn("kinetic_energy", names.tolist())
        self.assertIn("palinstrophy", names.tolist())
        self.assertTrue(np.all(np.isfinite(values)))

    def test_comparison_errors_are_zero_for_identical_trajectories(self):
        trajectory = np.ones((3, 8, 8), dtype=np.float64)
        errors = comparison_errors(trajectory, trajectory.copy())

        for values in errors.values():
            self.assertTrue(np.allclose(values, 0.0))


if __name__ == "__main__":
    unittest.main()
