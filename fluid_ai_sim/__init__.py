"""Hybrid Navier-Stokes simulation with neural fast rollouts."""

from .heat import HeatSolverConfig, SpectralHeatEquation2D, chip_heat_source_map, random_temperature_field
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
    "HeatSolverConfig",
    "SpectralHeatEquation2D",
    "random_vorticity",
    "random_divergence_free_velocity",
    "chip_heat_source_map",
    "random_temperature_field",
]
