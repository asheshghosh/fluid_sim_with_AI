import unittest

import numpy as np

from fluid_ai_sim.data import generate_heat_trajectories, make_transition_pairs
from fluid_ai_sim.heat import (
    HeatSolverConfig,
    SpectralHeatEquation2D,
    chip_heat_source_map,
    random_temperature_field,
)


class HeatSolverTests(unittest.TestCase):
    def test_chip_heat_source_is_nonnegative_and_normalized(self):
        source = chip_heat_source_map(32, source_count=3, width=0.25, amplitude=2.5, seed=7)

        self.assertEqual(source.shape, (32, 32))
        self.assertGreaterEqual(float(np.min(source)), 0.0)
        self.assertAlmostEqual(float(np.max(source)), 2.5, places=12)

    def test_step_returns_finite_temperature(self):
        config = HeatSolverConfig(n=32, diffusivity=1.0e-2, dt=1.0e-2, source_amplitude=0.5)
        solver = SpectralHeatEquation2D(config)
        temperature = random_temperature_field(32, seed=4, amplitude=0.1)

        next_temperature = solver.step(temperature)

        self.assertEqual(next_temperature.shape, temperature.shape)
        self.assertTrue(np.all(np.isfinite(next_temperature)))

    def test_diffusion_reduces_variance_without_source(self):
        config = HeatSolverConfig(
            n=32,
            diffusivity=5.0e-2,
            dt=2.0e-2,
            sink_rate=0.0,
            source_amplitude=0.0,
            source_count=0,
        )
        solver = SpectralHeatEquation2D(config)
        temperature = random_temperature_field(32, seed=5, amplitude=1.0)

        start = solver.diagnostics(temperature)["thermal_variance"]
        end = solver.diagnostics(solver.rollout(temperature, steps=10)[-1])["thermal_variance"]

        self.assertLess(end, start)

    def test_positive_source_increases_mean_temperature(self):
        config = HeatSolverConfig(n=32, diffusivity=1.0e-2, dt=1.0e-2, sink_rate=0.0, source_amplitude=1.0)
        solver = SpectralHeatEquation2D(config)
        temperature = np.zeros((32, 32), dtype=np.float64)

        next_temperature = solver.step(temperature)

        self.assertGreater(float(np.mean(next_temperature)), 0.0)

    def test_heat_trajectory_pairs_keep_temperature_and_source_channels(self):
        config = HeatSolverConfig(n=16, diffusivity=1.0e-2, dt=1.0e-2, source_count=2)
        trajectories = generate_heat_trajectories(config, trajectories=2, steps=4, keep_every=2)

        x, y = make_transition_pairs(trajectories)

        self.assertEqual(trajectories.shape, (2, 3, 2, 16, 16))
        self.assertEqual(x.shape, (4, 2, 16, 16))
        self.assertEqual(y.shape, (4, 2, 16, 16))
        np.testing.assert_allclose(x[:, 1], y[:, 1])


if __name__ == "__main__":
    unittest.main()
