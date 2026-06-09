"""Hybrid Navier-Stokes simulation with neural fast rollouts."""

from .incompressible import (
    SpectralIncompressibleNavierStokes2D,
    VelocitySolverConfig,
    random_divergence_free_velocity,
)
from .solver import SolverConfig, SpectralNavierStokes2D, random_vorticity

__all__ = [
    "SolverConfig",
    "SpectralNavierStokes2D",
    "VelocitySolverConfig",
    "SpectralIncompressibleNavierStokes2D",
    "random_vorticity",
    "random_divergence_free_velocity",
]
