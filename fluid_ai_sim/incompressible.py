from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict

import numpy as np


Array = np.ndarray


@dataclass(frozen=True)
class VelocitySolverConfig:
    """Configuration for a periodic 2D incompressible velocity solver."""

    n: int = 64
    length: float = 2.0 * np.pi
    viscosity: float = 1.0e-3
    dt: float = 1.0e-2
    forcing_amplitude: float = 0.0
    forcing_wavenumber: int = 4
    dealias: bool = True

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


class SpectralIncompressibleNavierStokes2D:
    """Pseudo-spectral velocity-pressure solver with periodic boundaries.

    The state is velocity with shape ``[2, n, n]``. Pressure is eliminated by a
    Fourier-space Leray projection, so every exact solver step returns a
    divergence-free velocity field up to floating-point roundoff.
    """

    def __init__(self, config: VelocitySolverConfig):
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
        self._forcing_hat = self.project_hat(np.fft.fft2(self._forcing, axes=(-2, -1)))

    def _validate_velocity(self, velocity: Array) -> Array:
        velocity = np.asarray(velocity, dtype=np.float64)
        if velocity.shape != (2, self.n, self.n):
            raise ValueError(f"expected velocity shape {(2, self.n, self.n)}, got {velocity.shape}")
        return velocity

    def _build_forcing(self) -> Array:
        cfg = self.config
        forcing = np.zeros((2, self.n, self.n), dtype=np.float64)
        if cfg.forcing_amplitude != 0.0:
            forcing[0] = cfg.forcing_amplitude * np.sin(cfg.forcing_wavenumber * self.y)
        return forcing

    def project_hat(self, velocity_hat: Array) -> Array:
        """Apply the Fourier-space incompressibility projection."""

        projected = np.array(velocity_hat, dtype=np.complex128, copy=True)
        dot = self.kx * projected[0] + self.ky * projected[1]
        projected[0] -= self.kx * dot * self.inv_k2
        projected[1] -= self.ky * dot * self.inv_k2
        projected *= self.dealias_mask
        projected[:, 0, 0] = 0.0
        return projected

    def project_velocity(self, velocity: Array) -> Array:
        velocity = self._validate_velocity(velocity)
        velocity_hat = np.fft.fft2(velocity, axes=(-2, -1))
        projected_hat = self.project_hat(velocity_hat)
        return np.fft.ifft2(projected_hat, axes=(-2, -1)).real

    def divergence(self, velocity: Array) -> Array:
        velocity = self._validate_velocity(velocity)
        velocity_hat = np.fft.fft2(velocity, axes=(-2, -1))
        div_hat = 1j * self.kx * velocity_hat[0] + 1j * self.ky * velocity_hat[1]
        return np.fft.ifft2(div_hat).real

    def vorticity(self, velocity: Array) -> Array:
        velocity = self._validate_velocity(velocity)
        velocity_hat = np.fft.fft2(velocity, axes=(-2, -1))
        omega_hat = 1j * self.kx * velocity_hat[1] - 1j * self.ky * velocity_hat[0]
        return np.fft.ifft2(omega_hat).real

    def nonlinear_advection_hat(self, velocity_hat: Array) -> Array:
        velocity_hat = self.project_hat(velocity_hat) * self.dealias_mask
        u = np.fft.ifft2(velocity_hat[0]).real
        v = np.fft.ifft2(velocity_hat[1]).real

        du_dx = np.fft.ifft2(1j * self.kx * velocity_hat[0]).real
        du_dy = np.fft.ifft2(1j * self.ky * velocity_hat[0]).real
        dv_dx = np.fft.ifft2(1j * self.kx * velocity_hat[1]).real
        dv_dy = np.fft.ifft2(1j * self.ky * velocity_hat[1]).real

        advection = np.empty((2, self.n, self.n), dtype=np.float64)
        advection[0] = u * du_dx + v * du_dy
        advection[1] = u * dv_dx + v * dv_dy
        return np.fft.fft2(advection, axes=(-2, -1)) * self.dealias_mask

    def step(self, velocity: Array) -> Array:
        velocity = self.project_velocity(velocity)
        velocity_hat = np.fft.fft2(velocity, axes=(-2, -1))
        advection_hat = self.project_hat(self.nonlinear_advection_hat(velocity_hat))

        cfg = self.config
        rhs_hat = velocity_hat + cfg.dt * (-advection_hat + self._forcing_hat)
        next_hat = rhs_hat / (1.0 + cfg.dt * cfg.viscosity * self.k2)
        next_hat *= self.dealias_mask
        next_hat = self.project_hat(next_hat)
        return np.fft.ifft2(next_hat, axes=(-2, -1)).real

    def rollout(self, velocity0: Array, steps: int, keep_every: int = 1) -> Array:
        if steps < 0:
            raise ValueError("steps must be non-negative")
        if keep_every <= 0:
            raise ValueError("keep_every must be positive")

        velocity = self.project_velocity(velocity0)
        frames = [velocity.copy()]
        for step in range(1, steps + 1):
            velocity = self.step(velocity)
            if step % keep_every == 0:
                frames.append(velocity.copy())
        return np.stack(frames, axis=0)

    def diagnostics(self, velocity: Array) -> Dict[str, float]:
        velocity = self._validate_velocity(velocity)
        omega = self.vorticity(velocity)
        speed = np.sqrt(velocity[0] * velocity[0] + velocity[1] * velocity[1])
        divergence = self.divergence(velocity)
        return {
            "kinetic_energy": float(0.5 * np.mean(speed * speed)),
            "enstrophy": float(0.5 * np.mean(omega * omega)),
            "velocity_rms": float(np.sqrt(np.mean(speed * speed))),
            "speed_mean": float(np.mean(speed)),
            "speed_max": float(np.max(speed)),
            "vorticity_mean": float(np.mean(omega)),
            "vorticity_std": float(np.std(omega)),
            "vorticity_min": float(np.min(omega)),
            "vorticity_max": float(np.max(omega)),
            "vorticity_linf": float(np.max(np.abs(omega))),
            "divergence_linf": float(np.max(np.abs(divergence))),
        }


def random_divergence_free_velocity(
    n: int,
    seed: int = 0,
    length: float = 2.0 * np.pi,
    low_pass: int = 8,
    amplitude: float = 1.0,
) -> Array:
    """Create a smooth, zero-mean, divergence-free random velocity field."""

    rng = np.random.default_rng(seed)
    raw = rng.normal(size=(2, n, n))

    dx = length / n
    k = 2.0 * np.pi * np.fft.fftfreq(n, d=dx)
    kx, ky = np.meshgrid(k, k, indexing="ij")
    k2 = kx * kx + ky * ky
    inv_k2 = np.zeros_like(k2)
    nonzero = k2 > 0.0
    inv_k2[nonzero] = 1.0 / k2[nonzero]
    mask = k2 <= float(low_pass * low_pass)

    velocity_hat = np.fft.fft2(raw, axes=(-2, -1)) * mask
    dot = kx * velocity_hat[0] + ky * velocity_hat[1]
    velocity_hat[0] -= kx * dot * inv_k2
    velocity_hat[1] -= ky * dot * inv_k2
    velocity_hat[:, 0, 0] = 0.0
    velocity = np.fft.ifft2(velocity_hat, axes=(-2, -1)).real

    rms_speed = np.sqrt(np.mean(velocity[0] * velocity[0] + velocity[1] * velocity[1]))
    if rms_speed > 0.0:
        velocity = amplitude * velocity / rms_speed
    return velocity.astype(np.float64)
