from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Tuple

import numpy as np

from .incompressible import (
    SpectralIncompressibleNavierStokes2D,
    VelocitySolverConfig,
    random_divergence_free_velocity,
)


Array = np.ndarray


@dataclass(frozen=True)
class ActiveNematicConfig:
    """Configuration for a periodic 2D Beris-Edwards active nematic solver."""

    n: int = 64
    length: float = 2.0 * np.pi
    viscosity: float = 1.0e-3
    dt: float = 1.0e-2
    activity: float = 0.25
    elastic_constant: float = 2.0e-2
    rotational_diffusivity: float = 0.4
    bulk_strength: float = 1.0
    preferred_order: float = 0.6
    flow_alignment: float = 0.7
    dealias: bool = True
    max_order: float = 1.5

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


def _wrap_angle(angle: Array) -> Array:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def _matmul2(left: Array, right: Array) -> Array:
    out = np.empty_like(left)
    out[0, 0] = left[0, 0] * right[0, 0] + left[0, 1] * right[1, 0]
    out[0, 1] = left[0, 0] * right[0, 1] + left[0, 1] * right[1, 1]
    out[1, 0] = left[1, 0] * right[0, 0] + left[1, 1] * right[1, 0]
    out[1, 1] = left[1, 0] * right[0, 1] + left[1, 1] * right[1, 1]
    return out


class SpectralActiveNematic2D:
    """Pseudo-spectral Beris-Edwards-style active nematic solver.

    The state has shape ``[4, n, n]``:

    ``state[0] = u``, ``state[1] = v``, ``state[2] = Qxx``,
    ``state[3] = Qxy``. The traceless 2D tensor is
    ``Q = [[Qxx, Qxy], [Qxy, -Qxx]]``.
    """

    def __init__(self, config: ActiveNematicConfig):
        if config.n < 8:
            raise ValueError("n must be at least 8")
        if config.dt <= 0.0:
            raise ValueError("dt must be positive")
        if config.viscosity < 0.0:
            raise ValueError("viscosity must be non-negative")
        if config.elastic_constant < 0.0:
            raise ValueError("elastic_constant must be non-negative")
        if config.rotational_diffusivity < 0.0:
            raise ValueError("rotational_diffusivity must be non-negative")
        if config.max_order <= 0.0:
            raise ValueError("max_order must be positive")

        self.config = config
        self.n = config.n
        self.length = config.length
        velocity_config = VelocitySolverConfig(
            n=config.n,
            length=config.length,
            viscosity=config.viscosity,
            dt=config.dt,
            forcing_amplitude=0.0,
            forcing_wavenumber=4,
            dealias=config.dealias,
        )
        self.hydro = SpectralIncompressibleNavierStokes2D(velocity_config)
        self.x = self.hydro.x
        self.y = self.hydro.y
        self.kx = self.hydro.kx
        self.ky = self.hydro.ky
        self.k2 = self.hydro.k2
        self.dealias_mask = self.hydro.dealias_mask

    def _validate_state(self, state: Array) -> Array:
        state = np.asarray(state, dtype=np.float64)
        if state.shape != (4, self.n, self.n):
            raise ValueError(f"expected state shape {(4, self.n, self.n)}, got {state.shape}")
        return state

    @staticmethod
    def split_state(state: Array) -> Tuple[Array, Array]:
        return state[:2], state[2:]

    @staticmethod
    def pack_state(velocity: Array, q: Array) -> Array:
        return np.concatenate([velocity, q], axis=0)

    @staticmethod
    def q_matrix(q: Array) -> Array:
        qmat = np.empty((2, 2, *q.shape[-2:]), dtype=np.float64)
        qmat[0, 0] = q[0]
        qmat[0, 1] = q[1]
        qmat[1, 0] = q[1]
        qmat[1, 1] = -q[0]
        return qmat

    @staticmethod
    def q_components(qmat: Array) -> Array:
        q = np.empty((2, *qmat.shape[-2:]), dtype=np.float64)
        q[0] = 0.5 * (qmat[0, 0] - qmat[1, 1])
        q[1] = 0.5 * (qmat[0, 1] + qmat[1, 0])
        return q

    def project_state(self, state: Array) -> Array:
        state = self._validate_state(state)
        velocity, q = self.split_state(state)
        velocity = self.hydro.project_velocity(velocity)
        q = np.array(q, dtype=np.float64, copy=True)
        magnitude = np.sqrt(q[0] * q[0] + q[1] * q[1])
        too_large = magnitude > self.config.max_order
        if np.any(too_large):
            scale = self.config.max_order / np.maximum(magnitude, 1.0e-12)
            q[:, too_large] *= scale[too_large]
        return self.pack_state(velocity, q)

    def divergence(self, state: Array) -> Array:
        velocity, _ = self.split_state(self._validate_state(state))
        return self.hydro.divergence(velocity)

    def vorticity(self, state: Array) -> Array:
        velocity, _ = self.split_state(self._validate_state(state))
        return self.hydro.vorticity(velocity)

    def director_angle(self, state: Array) -> Array:
        _, q = self.split_state(self._validate_state(state))
        return 0.5 * np.arctan2(q[1], q[0])

    def scalar_order(self, state: Array) -> Array:
        _, q = self.split_state(self._validate_state(state))
        return np.sqrt(q[0] * q[0] + q[1] * q[1])

    def active_force_hat(self, q: Array) -> Array:
        q_hat = np.fft.fft2(q, axes=(-2, -1)) * self.dealias_mask
        force = np.empty((2, self.n, self.n), dtype=np.float64)
        force[0] = np.fft.ifft2(1j * self.kx * q_hat[0] + 1j * self.ky * q_hat[1]).real
        force[1] = np.fft.ifft2(1j * self.kx * q_hat[1] - 1j * self.ky * q_hat[0]).real
        return self.config.activity * np.fft.fft2(force, axes=(-2, -1)) * self.dealias_mask

    def _q_rhs(self, velocity: Array, q: Array) -> Array:
        cfg = self.config
        velocity_hat = np.fft.fft2(velocity, axes=(-2, -1)) * self.dealias_mask
        q_hat = np.fft.fft2(q, axes=(-2, -1)) * self.dealias_mask

        u = np.fft.ifft2(velocity_hat[0]).real
        v = np.fft.ifft2(velocity_hat[1]).real
        du_dx = np.fft.ifft2(1j * self.kx * velocity_hat[0]).real
        du_dy = np.fft.ifft2(1j * self.ky * velocity_hat[0]).real
        dv_dx = np.fft.ifft2(1j * self.kx * velocity_hat[1]).real
        dv_dy = np.fft.ifft2(1j * self.ky * velocity_hat[1]).real

        dqxx_dx = np.fft.ifft2(1j * self.kx * q_hat[0]).real
        dqxx_dy = np.fft.ifft2(1j * self.ky * q_hat[0]).real
        dqxy_dx = np.fft.ifft2(1j * self.kx * q_hat[1]).real
        dqxy_dy = np.fft.ifft2(1j * self.ky * q_hat[1]).real
        advection = np.empty_like(q)
        advection[0] = u * dqxx_dx + v * dqxx_dy
        advection[1] = u * dqxy_dx + v * dqxy_dy

        qmat = self.q_matrix(q)
        identity_shift = np.zeros_like(qmat)
        identity_shift[0, 0] = 0.5
        identity_shift[1, 1] = 0.5
        qplus = qmat + identity_shift

        grad = np.empty_like(qmat)
        grad[0, 0] = du_dx
        grad[0, 1] = du_dy
        grad[1, 0] = dv_dx
        grad[1, 1] = dv_dy
        grad_t = np.empty_like(grad)
        grad_t[0, 0] = grad[0, 0]
        grad_t[0, 1] = grad[1, 0]
        grad_t[1, 0] = grad[0, 1]
        grad_t[1, 1] = grad[1, 1]
        strain = 0.5 * (grad + grad_t)
        rotation = 0.5 * (grad - grad_t)

        xi = cfg.flow_alignment
        left = xi * strain + rotation
        right = xi * strain - rotation
        q_contract_strain = (
            qmat[0, 0] * strain[0, 0]
            + qmat[0, 1] * strain[0, 1]
            + qmat[1, 0] * strain[1, 0]
            + qmat[1, 1] * strain[1, 1]
        )
        alignment = _matmul2(left, qplus) + _matmul2(qplus, right) - 2.0 * xi * qplus * q_contract_strain
        alignment_q = self.q_components(alignment)

        q_magnitude2 = q[0] * q[0] + q[1] * q[1]
        bulk = cfg.bulk_strength * (cfg.preferred_order * cfg.preferred_order - q_magnitude2) * q
        return -advection + alignment_q + cfg.rotational_diffusivity * bulk

    def step(self, state: Array) -> Array:
        state = self.project_state(state)
        velocity, q = self.split_state(state)
        cfg = self.config

        velocity_hat = np.fft.fft2(velocity, axes=(-2, -1))
        advection_hat = self.hydro.project_hat(self.hydro.nonlinear_advection_hat(velocity_hat))
        active_force_hat = self.hydro.project_hat(self.active_force_hat(q))
        rhs_hat = velocity_hat + cfg.dt * (-advection_hat + active_force_hat)
        next_velocity_hat = rhs_hat / (1.0 + cfg.dt * cfg.viscosity * self.k2)
        next_velocity = np.fft.ifft2(self.hydro.project_hat(next_velocity_hat), axes=(-2, -1)).real

        q_rhs = self._q_rhs(velocity, q)
        q_rhs_hat = np.fft.fft2(q + cfg.dt * q_rhs, axes=(-2, -1))
        next_q_hat = q_rhs_hat / (1.0 + cfg.dt * cfg.rotational_diffusivity * cfg.elastic_constant * self.k2)
        next_q_hat *= self.dealias_mask
        next_q = np.fft.ifft2(next_q_hat, axes=(-2, -1)).real
        return self.project_state(self.pack_state(next_velocity, next_q))

    def rollout(self, state0: Array, steps: int, keep_every: int = 1) -> Array:
        if steps < 0:
            raise ValueError("steps must be non-negative")
        if keep_every <= 0:
            raise ValueError("keep_every must be positive")

        state = self.project_state(state0)
        frames = [state.copy()]
        for step in range(1, steps + 1):
            state = self.step(state)
            if step % keep_every == 0:
                frames.append(state.copy())
        return np.stack(frames, axis=0)

    def defect_charge_grid(self, state: Array) -> Array:
        _, q = self.split_state(self._validate_state(state))
        phase = np.arctan2(q[1], q[0])
        d_x = _wrap_angle(np.roll(phase, -1, axis=0) - phase)
        d_y = _wrap_angle(np.roll(phase, -1, axis=1) - phase)
        winding = d_x + np.roll(d_y, -1, axis=0) - np.roll(d_x, -1, axis=1) - d_y
        return 0.5 * np.rint(winding / (2.0 * np.pi))

    def defect_counts(self, state: Array) -> Dict[str, float]:
        charges = self.defect_charge_grid(state)
        plus = int(np.count_nonzero(charges > 0.25))
        minus = int(np.count_nonzero(charges < -0.25))
        area = self.length * self.length
        return {
            "defects_plus_half": float(plus),
            "defects_minus_half": float(minus),
            "defect_total": float(plus + minus),
            "defect_density": float((plus + minus) / area),
            "net_topological_charge": float(np.sum(charges)),
        }

    def diagnostics(self, state: Array) -> Dict[str, float]:
        state = self.project_state(state)
        velocity, q = self.split_state(state)
        speed = np.sqrt(velocity[0] * velocity[0] + velocity[1] * velocity[1])
        omega = self.hydro.vorticity(velocity)
        divergence = self.hydro.divergence(velocity)
        q_order = np.sqrt(q[0] * q[0] + q[1] * q[1])

        q_hat = np.fft.fft2(q, axes=(-2, -1))
        qxx_dx = np.fft.ifft2(1j * self.kx * q_hat[0]).real
        qxx_dy = np.fft.ifft2(1j * self.ky * q_hat[0]).real
        qxy_dx = np.fft.ifft2(1j * self.kx * q_hat[1]).real
        qxy_dy = np.fft.ifft2(1j * self.ky * q_hat[1]).real
        elastic_density = 0.5 * self.config.elastic_constant * np.mean(
            qxx_dx * qxx_dx + qxx_dy * qxx_dy + qxy_dx * qxy_dx + qxy_dy * qxy_dy
        )
        bulk_density = 0.25 * self.config.bulk_strength * np.mean(
            (q_order * q_order - self.config.preferred_order * self.config.preferred_order) ** 2
        )
        defects = self.defect_counts(state)
        active_length = np.sqrt(
            self.config.elastic_constant / max(abs(self.config.activity), 1.0e-12)
        )
        return {
            "kinetic_energy": float(0.5 * np.mean(speed * speed)),
            "enstrophy": float(0.5 * np.mean(omega * omega)),
            "velocity_rms": float(np.sqrt(np.mean(speed * speed))),
            "speed_max": float(np.max(speed)),
            "q_order_mean": float(np.mean(q_order)),
            "q_order_std": float(np.std(q_order)),
            "q_order_min": float(np.min(q_order)),
            "q_order_max": float(np.max(q_order)),
            "free_energy_density": float(elastic_density + bulk_density),
            "elastic_energy_density": float(elastic_density),
            "bulk_energy_density": float(bulk_density),
            "vorticity_linf": float(np.max(np.abs(omega))),
            "divergence_linf": float(np.max(np.abs(divergence))),
            "active_length": float(active_length),
            **defects,
        }


def _smooth_random_scalar(n: int, length: float, seed: int, low_pass: int) -> Array:
    rng = np.random.default_rng(seed)
    raw = rng.normal(size=(n, n))
    dx = length / n
    k = 2.0 * np.pi * np.fft.fftfreq(n, d=dx)
    kx, ky = np.meshgrid(k, k, indexing="ij")
    mask = (kx * kx + ky * ky) <= float(low_pass * low_pass)
    field = np.fft.ifft2(np.fft.fft2(raw) * mask).real
    field -= np.mean(field)
    std = np.std(field)
    if std > 0.0:
        field /= std
    return field


def random_active_nematic_state(
    config: ActiveNematicConfig,
    seed: int = 0,
    velocity_amplitude: float = 0.15,
    angle_amplitude: float = np.pi,
) -> Array:
    """Create a smooth projected velocity plus smooth nematic Q field."""

    low_pass = max(3, config.n // 8)
    velocity = random_divergence_free_velocity(
        config.n,
        seed=seed,
        length=config.length,
        low_pass=low_pass,
        amplitude=velocity_amplitude,
    )
    theta_noise = _smooth_random_scalar(config.n, config.length, seed + 10_000, low_pass)
    order_noise = _smooth_random_scalar(config.n, config.length, seed + 20_000, low_pass)
    theta = angle_amplitude * theta_noise
    order = config.preferred_order * np.clip(1.0 + 0.08 * order_noise, 0.2, 1.4)
    q = np.empty((2, config.n, config.n), dtype=np.float64)
    q[0] = order * np.cos(2.0 * theta)
    q[1] = order * np.sin(2.0 * theta)
    solver = SpectralActiveNematic2D(config)
    return solver.project_state(SpectralActiveNematic2D.pack_state(velocity, q))
