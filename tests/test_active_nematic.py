import unittest

import numpy as np

from fluid_ai_sim.active_nematic import (
    ActiveNematicConfig,
    SpectralActiveNematic2D,
    random_active_nematic_state,
)
from fluid_ai_sim.data import generate_active_nematic_trajectories, make_transition_pairs


class ActiveNematicTests(unittest.TestCase):
    def test_random_state_has_expected_channels_and_projected_velocity(self):
        config = ActiveNematicConfig(n=16, dt=2.0e-3)
        solver = SpectralActiveNematic2D(config)

        state = random_active_nematic_state(config, seed=1)

        self.assertEqual(state.shape, (4, 16, 16))
        self.assertTrue(np.all(np.isfinite(state)))
        self.assertLess(float(np.max(np.abs(solver.divergence(state)))), 1.0e-10)

    def test_step_returns_finite_projected_state(self):
        config = ActiveNematicConfig(n=16, dt=1.0e-3, activity=0.2, elastic_constant=1.0e-2)
        solver = SpectralActiveNematic2D(config)
        state = random_active_nematic_state(config, seed=2)

        next_state = solver.step(state)

        self.assertEqual(next_state.shape, state.shape)
        self.assertTrue(np.all(np.isfinite(next_state)))
        self.assertLess(float(np.max(np.abs(solver.divergence(next_state)))), 1.0e-10)

    def test_uniform_director_has_no_defects(self):
        config = ActiveNematicConfig(n=16)
        solver = SpectralActiveNematic2D(config)
        state = np.zeros((4, 16, 16), dtype=np.float64)
        state[2] = config.preferred_order

        counts = solver.defect_counts(state)

        self.assertEqual(counts["defect_total"], 0.0)
        self.assertEqual(counts["net_topological_charge"], 0.0)

    def test_active_nematic_trajectory_pairs_keep_four_channels(self):
        config = ActiveNematicConfig(n=16, dt=1.0e-3)
        trajectories = generate_active_nematic_trajectories(config, trajectories=2, steps=4, keep_every=2)

        x, y = make_transition_pairs(trajectories)

        self.assertEqual(trajectories.shape, (2, 3, 4, 16, 16))
        self.assertEqual(x.shape, (4, 4, 16, 16))
        self.assertEqual(y.shape, (4, 4, 16, 16))


if __name__ == "__main__":
    unittest.main()
