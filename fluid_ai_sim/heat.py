from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict

import numpy as np


Array = np.ndarray


@dataclass(frozen=True)
class HeatSolverConfig:
    """Configuration for a periodic 2D heat-diffusion solver.

    The evolved state is the temperature rise above an ambient baseline. A
    positive source term models localized chip power dissipation.
    """

    n: int = 64
    length: float = 2.0 * np.pi
    diffusivity: float = 5.0e-3
    dt: float = 2.0e-2
    sink_rate: float = 2.0e-3
    source_amplitude: float = 1.0
    source_width: float = 0.22
    source_count: int = 4
    source_seed: int = 0
    ambient_temperature: float = 300.0
    dealias: bool = True

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


class SpectralHeatEquation2D:
    """Semi-implicit pseudo-spectral solver for periodic chip heat diffusion.

    The equation is

        dT/dt = kappa Laplacian(T) - gamma T + q(x, y),

    where ``T`` is the temperature rise above ambient, ``kappa`` is thermal
    diffusivity, ``gamma`` is a simple cooling/sink term, and ``q`` is a
    prescribed heat-source map. Periodicity is useful for clean spectral tests;
    later chip-focused branches can swap in package boundaries or substrate
    layers without changing the AI rollout interface.
    """

    def __init__(self, config: HeatSolverConfig):
        if config.n < 8:
            raise ValueError("n must be at least 8")
        if config.dt <= 0.0:
            raise ValueError("dt must be positive")
        if config.diffusivity < 0.0:
            raise ValueError("diffusivity must be non-negative")
        if config.sink_rate < 0.0:
            raise ValueError("sink_rate must be non-negative")
        if config.source_count < 0:
            raise ValueError("source_count must be non-negative")
        if config.source_width <= 0.0:
            raise ValueError("source_width must be positive")

        self.config = config
        self.n = config.n
        self.length = config.length
        self.dx = self.length / self.n

        x = np.linspace(0.0, self.length, self.n, endpoint=False)
        self.x, self.y = np.meshgrid(x, x, indexing="ij")

        k = 2.0 * np.pi * np.fft.fftfreq(self.n, d=self.dx)
        self.kx, self.ky = np.meshgrid(k, k, indexing="ij")
        self.k2 = self.kx * self.kx + self.ky * self.ky

        if config.dealias:
            cutoff = (2.0 / 3.0) * np.max(np.abs(k))
            self.dealias_mask = (np.abs(self.kx) <= cutoff) & (np.abs(self.ky) <= cutoff)
        else:
            self.dealias_mask = np.ones((self.n, self.n), dtype=bool)

        self.source = chip_heat_source_map(
            self.n,
            length=self.length,
            source_count=config.source_count,
            width=config.source_width,
            amplitude=config.source_amplitude,
            seed=config.source_seed,
        )
        self.source_hat = np.fft.fft2(self.source) * self.dealias_mask

    def _validate_temperature(self, temperature: Array) -> Array:
        temperature = np.asarray(temperature, dtype=np.float64)
        if temperature.shape != (self.n, self.n):
            raise ValueError(f"expected temperature shape {(self.n, self.n)}, got {temperature.shape}")
        return temperature

    def gradient(self, temperature: Array) -> tuple[Array, Array]:
        temperature = self._validate_temperature(temperature)
        temp_hat = np.fft.fft2(temperature)
        dtdx = np.fft.ifft2(1j * self.kx * temp_hat).real
        dtdy = np.fft.ifft2(1j * self.ky * temp_hat).real
        return dtdx, dtdy

    def laplacian(self, temperature: Array) -> Array:
        temperature = self._validate_temperature(temperature)
        temp_hat = np.fft.fft2(temperature)
        return np.fft.ifft2(-self.k2 * temp_hat).real

    def heat_flux(self, temperature: Array) -> tuple[Array, Array]:
        dtdx, dtdy = self.gradient(temperature)
        return -self.config.diffusivity * dtdx, -self.config.diffusivity * dtdy

    def step(self, temperature: Array) -> Array:
        temperature = self._validate_temperature(temperature)
        temp_hat = np.fft.fft2(temperature) * self.dealias_mask
        cfg = self.config
        rhs_hat = temp_hat + cfg.dt * self.source_hat
        denominator = 1.0 + cfg.dt * (cfg.diffusivity * self.k2 + cfg.sink_rate)
        next_hat = rhs_hat / denominator
        next_hat *= self.dealias_mask
        return np.fft.ifft2(next_hat).real

    def rollout(self, temperature0: Array, steps: int, keep_every: int = 1) -> Array:
        if steps < 0:
            raise ValueError("steps must be non-negative")
        if keep_every <= 0:
            raise ValueError("keep_every must be positive")

        temperature = self._validate_temperature(temperature0).copy()
        frames = [temperature.copy()]
        for step in range(1, steps + 1):
            temperature = self.step(temperature)
            if step % keep_every == 0:
                frames.append(temperature.copy())
        return np.stack(frames, axis=0)

    def diagnostics(self, temperature: Array) -> Dict[str, float]:
        temperature = self._validate_temperature(temperature)
        dtdx, dtdy = self.gradient(temperature)
        flux_x, flux_y = self.heat_flux(temperature)
        absolute = temperature + self.config.ambient_temperature
        source_power = float(np.mean(self.source))
        cooling_power = float(self.config.sink_rate * np.mean(temperature))
        return {
            "temperature_mean": float(np.mean(temperature)),
            "temperature_std": float(np.std(temperature)),
            "temperature_min": float(np.min(temperature)),
            "temperature_max": float(np.max(temperature)),
            "temperature_linf": float(np.max(np.abs(temperature))),
            "absolute_temperature_mean": float(np.mean(absolute)),
            "absolute_temperature_max": float(np.max(absolute)),
            "thermal_energy": float(np.mean(temperature)),
            "thermal_variance": float(np.mean((temperature - np.mean(temperature)) ** 2)),
            "gradient_rms": float(np.sqrt(np.mean(dtdx * dtdx + dtdy * dtdy))),
            "hotspot_area_fraction": float(np.mean(temperature >= 0.8 * np.max(temperature))) if np.max(temperature) > 0.0 else 0.0,
            "source_power_density": source_power,
            "cooling_power_density": cooling_power,
            "net_power_density": source_power - cooling_power,
            "flux_rms": float(np.sqrt(np.mean(flux_x * flux_x + flux_y * flux_y))),
            "source_temperature_correlation": _safe_correlation(self.source, temperature),
        }


def _periodic_delta(a: Array, b: float, length: float) -> Array:
    return (a - b + 0.5 * length) % length - 0.5 * length


def _safe_correlation(a: Array, b: Array) -> float:
    aa = np.asarray(a, dtype=np.float64) - float(np.mean(a))
    bb = np.asarray(b, dtype=np.float64) - float(np.mean(b))
    denom = float(np.sqrt(np.mean(aa * aa) * np.mean(bb * bb)))
    if denom <= 1.0e-14:
        return 0.0
    return float(np.mean(aa * bb) / denom)


def chip_heat_source_map(
    n: int,
    length: float = 2.0 * np.pi,
    source_count: int = 4,
    width: float = 0.22,
    amplitude: float = 1.0,
    seed: int = 0,
) -> Array:
    """Create smooth periodic heat-source bumps resembling chip hot spots."""

    if source_count < 0:
        raise ValueError("source_count must be non-negative")
    rng = np.random.default_rng(seed)
    x = np.linspace(0.0, length, n, endpoint=False)
    grid_x, grid_y = np.meshgrid(x, x, indexing="ij")
    source = np.zeros((n, n), dtype=np.float64)
    if source_count == 0 or amplitude == 0.0:
        return source

    centers = rng.uniform(0.0, length, size=(source_count, 2))
    weights = rng.uniform(0.55, 1.0, size=source_count)
    for (cx, cy), weight in zip(centers, weights):
        dx = _periodic_delta(grid_x, float(cx), length)
        dy = _periodic_delta(grid_y, float(cy), length)
        source += weight * np.exp(-(dx * dx + dy * dy) / (2.0 * width * width))

    peak = float(np.max(source))
    if peak > 0.0:
        source *= amplitude / peak
    return source


def random_temperature_field(
    n: int,
    seed: int = 0,
    length: float = 2.0 * np.pi,
    low_pass: int = 8,
    amplitude: float = 0.2,
    bias: float = 0.0,
) -> Array:
    """Create a smooth random initial temperature rise."""

    rng = np.random.default_rng(seed)
    raw = rng.normal(size=(n, n))
    dx = length / n
    k = 2.0 * np.pi * np.fft.fftfreq(n, d=dx)
    kx, ky = np.meshgrid(k, k, indexing="ij")
    mask = (kx * kx + ky * ky) <= float(low_pass * low_pass)
    temp_hat = np.fft.fft2(raw) * mask
    temperature = np.fft.ifft2(temp_hat).real
    temperature -= float(np.mean(temperature))
    scale = float(np.std(temperature))
    if scale > 0.0:
        temperature = amplitude * temperature / scale
    return (temperature + bias).astype(np.float64)
