"""Hybrid Navier-Stokes simulation with neural fast rollouts."""

from .solver import SolverConfig, SpectralNavierStokes2D, random_vorticity

__all__ = ["SolverConfig", "SpectralNavierStokes2D", "random_vorticity"]
