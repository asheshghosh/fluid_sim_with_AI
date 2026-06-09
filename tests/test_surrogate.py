import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from fluid_ai_sim.solver import SolverConfig
from fluid_ai_sim.surrogate import (
    SurrogateConfig,
    build_surrogate,
    load_checkpoint,
    normalization_stats,
    save_checkpoint,
    tensorize,
)


class SurrogateTests(unittest.TestCase):
    def test_fno_forward_returns_finite_same_shape_batch(self):
        config = SurrogateConfig(model_type="fno", width=8, depth=2, modes=4)
        model = build_surrogate(config)
        x = torch.randn(2, 1, 16, 16)

        y = model(x)

        self.assertEqual(tuple(y.shape), tuple(x.shape))
        self.assertTrue(torch.all(torch.isfinite(y)))

    def test_fno_checkpoint_round_trips_model_type(self):
        fields = np.random.default_rng(0).normal(size=(2, 16, 16)).astype(np.float64)
        mean, std = normalization_stats(fields)
        config = SurrogateConfig(model_type="fno", width=8, depth=1, modes=4)
        model = build_surrogate(config)

        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "fno.pt"
            save_checkpoint(
                checkpoint,
                model,
                mean,
                std,
                SolverConfig(n=16).to_dict(),
                surrogate_step_size=4,
            )
            loaded, loaded_mean, loaded_std, solver_config = load_checkpoint(checkpoint, device=torch.device("cpu"))

        self.assertEqual(loaded.config.model_type, "fno")
        self.assertEqual(loaded.config.modes, 4)
        self.assertAlmostEqual(loaded_mean, mean)
        self.assertAlmostEqual(loaded_std, std)
        self.assertEqual(solver_config["n"], 16)

        x = tensorize(fields, loaded_mean, loaded_std, device=torch.device("cpu"))
        self.assertEqual(tuple(loaded(x).shape), tuple(x.shape))

    def test_two_channel_fno_checkpoint_round_trips(self):
        fields = np.random.default_rng(1).normal(size=(3, 2, 16, 16)).astype(np.float64)
        mean, std = normalization_stats(fields)
        config = SurrogateConfig(model_type="fno", channels=2, width=8, depth=1, modes=4)
        model = build_surrogate(config)

        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "velocity_fno.pt"
            save_checkpoint(
                checkpoint,
                model,
                mean,
                std,
                SolverConfig(n=16).to_dict(),
                surrogate_step_size=2,
            )
            loaded, loaded_mean, loaded_std, _ = load_checkpoint(checkpoint, device=torch.device("cpu"))

        self.assertEqual(loaded.config.channels, 2)
        x = tensorize(fields, loaded_mean, loaded_std, device=torch.device("cpu"))
        y = loaded(x)
        self.assertEqual(tuple(y.shape), tuple(x.shape))
        self.assertTrue(torch.all(torch.isfinite(y)))


if __name__ == "__main__":
    unittest.main()
