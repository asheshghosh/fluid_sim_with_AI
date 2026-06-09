from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np

from .incompressible import (
    SpectralIncompressibleNavierStokes2D,
    VelocitySolverConfig,
    random_divergence_free_velocity,
)
from .solver import SolverConfig, SpectralNavierStokes2D, random_vorticity


def generate_trajectories(
    config: SolverConfig,
    trajectories: int,
    steps: int,
    seed: int = 0,
    keep_every: int = 1,
    amplitude: float = 1.0,
) -> np.ndarray:
    if trajectories <= 0:
        raise ValueError("trajectories must be positive")
    if steps <= 0:
        raise ValueError("steps must be positive")

    solver = SpectralNavierStokes2D(config)
    samples = []
    for idx in range(trajectories):
        omega0 = random_vorticity(
            config.n,
            seed=seed + idx,
            length=config.length,
            low_pass=max(3, config.n // 8),
            amplitude=amplitude,
        )
        samples.append(solver.rollout(omega0, steps=steps, keep_every=keep_every))
    return np.stack(samples, axis=0)


def make_transition_pairs(trajectories: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if trajectories.ndim not in {4, 5}:
        raise ValueError("expected trajectories with shape [batch, time, n, n] or [batch, time, channels, n, n]")
    x = trajectories[:, :-1]
    y = trajectories[:, 1:]
    if trajectories.ndim == 4:
        return x.reshape(-1, *x.shape[-2:]), y.reshape(-1, *y.shape[-2:])
    return x.reshape(-1, *x.shape[-3:]), y.reshape(-1, *y.shape[-3:])


def generate_velocity_trajectories(
    config: VelocitySolverConfig,
    trajectories: int,
    steps: int,
    seed: int = 0,
    keep_every: int = 1,
    amplitude: float = 1.0,
) -> np.ndarray:
    if trajectories <= 0:
        raise ValueError("trajectories must be positive")
    if steps <= 0:
        raise ValueError("steps must be positive")

    solver = SpectralIncompressibleNavierStokes2D(config)
    samples = []
    for idx in range(trajectories):
        velocity0 = random_divergence_free_velocity(
            config.n,
            seed=seed + idx,
            length=config.length,
            low_pass=max(3, config.n // 8),
            amplitude=amplitude,
        )
        samples.append(solver.rollout(velocity0, steps=steps, keep_every=keep_every))
    return np.stack(samples, axis=0)


def save_dataset(path: str | Path, trajectories: np.ndarray, config: SolverConfig) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, trajectories=trajectories, config=config.to_dict())


def load_dataset(path: str | Path) -> Tuple[np.ndarray, dict]:
    data = np.load(Path(path), allow_pickle=True)
    config = data["config"].item() if "config" in data else {}
    return data["trajectories"], config
