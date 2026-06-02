from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Tuple

import numpy as np


Array = np.ndarray


@dataclass(frozen=True)
class SolverConfig:
    """Configuration for a periodic 2D vorticity Navier-Stokes solver."""

    n: int = 64
    length: float = 2.0 * np.pi
    viscosity: float = 1.0e-3
    dt: float = 1.0e-2
    forcing_amplitude: float = 0.0
    forcing_wavenumber: int = 4
    dealias: bool = True

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


class SpectralNavierStokes2D:
    """Pseudo-spectral solver for 2D incompressible Navier-Stokes.

    The state is scalar vorticity, omega. Velocity is recovered from the
    streamfunction, which keeps the flow divergence-free on the periodic grid.
    """

    def __init__(self, config: SolverConfig):
        if config.n < 8:
            raise ValueError("n must be at least 8")
        if config.dt <= 0.0:
            raise ValueError("dt must be positive")
        if config.viscosity < 0.0:
            raise ValueError("viscosity must be non-negative")

        self.config = config
        self.n = config.n
        self.length = config.length
        self.dx = self.length / self.n

        x = np.linspace(0.0, self.length, self.n, endpoint=False)
        self.x, self.y = np.meshgrid(x, x, indexing="ij")

        k = 2.0 * np.pi * np.fft.fftfreq(self.n, d=self.dx)
        self.kx, self.ky = np.meshgrid(k, k, indexing="ij")
        self.k2 = self.kx * self.kx + self.ky * self.ky
        self.inv_k2 = np.zeros_like(self.k2)
        nonzero = self.k2 > 0.0
        self.inv_k2[nonzero] = 1.0 / self.k2[nonzero]

        if config.dealias:
            cutoff = (2.0 / 3.0) * np.max(np.abs(k))
            self.dealias_mask = (np.abs(self.kx) <= cutoff) & (np.abs(self.ky) <= cutoff)
        else:
            self.dealias_mask = np.ones((self.n, self.n), dtype=bool)

        self._forcing = self._build_forcing()
        self._forcing_hat = np.fft.fft2(self._forcing)

    def _build_forcing(self) -> Array:
        cfg = self.config
        if cfg.forcing_amplitude == 0.0:
            return np.zeros((self.n, self.n), dtype=np.float64)
        return cfg.forcing_amplitude * np.sin(cfg.forcing_wavenumber * self.y)

    def streamfunction_hat(self, omega_hat: Array) -> Array:
        psi_hat = omega_hat * self.inv_k2
        psi_hat[0, 0] = 0.0
        return psi_hat

    def velocity_from_hat(self, omega_hat: Array) -> Tuple[Array, Array]:
        psi_hat = self.streamfunction_hat(omega_hat)
        u_hat = 1j * self.ky * psi_hat
        v_hat = -1j * self.kx * psi_hat
        u = np.fft.ifft2(u_hat).real
        v = np.fft.ifft2(v_hat).real
        return u, v

    def velocity(self, omega: Array) -> Tuple[Array, Array]:
        return self.velocity_from_hat(np.fft.fft2(omega))

    def divergence(self, omega: Array) -> Array:
        omega_hat = np.fft.fft2(omega)
        u_hat = 1j * self.ky * self.streamfunction_hat(omega_hat)
        v_hat = -1j * self.kx * self.streamfunction_hat(omega_hat)
        div_hat = 1j * self.kx * u_hat + 1j * self.ky * v_hat
        return np.fft.ifft2(div_hat).real

    def nonlinear_advection_hat(self, omega_hat: Array) -> Array:
        omega_hat = omega_hat * self.dealias_mask
        u, v = self.velocity_from_hat(omega_hat)
        domega_dx = np.fft.ifft2(1j * self.kx * omega_hat).real
        domega_dy = np.fft.ifft2(1j * self.ky * omega_hat).real
        advection = u * domega_dx + v * domega_dy
        return np.fft.fft2(advection) * self.dealias_mask

    def step(self, omega: Array) -> Array:
        if omega.shape != (self.n, self.n):
            raise ValueError(f"expected omega shape {(self.n, self.n)}, got {omega.shape}")

        omega_hat = np.fft.fft2(omega)
        advection_hat = self.nonlinear_advection_hat(omega_hat)

        cfg = self.config
        rhs_hat = omega_hat + cfg.dt * (-advection_hat + self._forcing_hat)
        next_hat = rhs_hat / (1.0 + cfg.dt * cfg.viscosity * self.k2)
        next_hat *= self.dealias_mask
        next_hat[0, 0] = 0.0
        return np.fft.ifft2(next_hat).real

    def rollout(self, omega0: Array, steps: int, keep_every: int = 1) -> Array:
        if steps < 0:
            raise ValueError("steps must be non-negative")
        if keep_every <= 0:
            raise ValueError("keep_every must be positive")

        omega = np.array(omega0, dtype=np.float64, copy=True)
        frames = [omega.copy()]
        for step in range(1, steps + 1):
            omega = self.step(omega)
            if step % keep_every == 0:
                frames.append(omega.copy())
        return np.stack(frames, axis=0)

    def diagnostics(self, omega: Array) -> Dict[str, float]:
        u, v = self.velocity(omega)
        omega_hat = np.fft.fft2(omega)
        domega_dx = np.fft.ifft2(1j * self.kx * omega_hat).real
        domega_dy = np.fft.ifft2(1j * self.ky * omega_hat).real
        return {
            "kinetic_energy": float(0.5 * np.mean(u * u + v * v)),
            "enstrophy": float(0.5 * np.mean(omega * omega)),
            "palinstrophy": float(0.5 * np.mean(domega_dx * domega_dx + domega_dy * domega_dy)),
            "vorticity_mean": float(np.mean(omega)),
            "vorticity_std": float(np.std(omega)),
            "vorticity_min": float(np.min(omega)),
            "vorticity_max": float(np.max(omega)),
            "vorticity_linf": float(np.max(np.abs(omega))),
            "circulation": float(np.mean(omega) * self.length * self.length),
            "divergence_linf": float(np.max(np.abs(self.divergence(omega)))),
        }


def random_vorticity(
    n: int,
    seed: int = 0,
    length: float = 2.0 * np.pi,
    low_pass: int = 8,
    amplitude: float = 1.0,
) -> Array:
    """Create smooth random vorticity with zero mean and controlled variance."""

    rng = np.random.default_rng(seed)
    raw = rng.normal(size=(n, n))

    dx = length / n
    k = 2.0 * np.pi * np.fft.fftfreq(n, d=dx)
    kx, ky = np.meshgrid(k, k, indexing="ij")
    mask = (kx * kx + ky * ky) <= float(low_pass * low_pass)

    omega_hat = np.fft.fft2(raw) * mask
    omega = np.fft.ifft2(omega_hat).real
    omega -= np.mean(omega)
    scale = np.std(omega)
    if scale > 0.0:
        omega = amplitude * omega / scale
    return omega.astype(np.float64)
