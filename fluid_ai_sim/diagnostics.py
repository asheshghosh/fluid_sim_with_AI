from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

from .solver import SpectralNavierStokes2D


DIAGNOSTIC_KEYS = [
    "step",
    "time",
    "kinetic_energy",
    "enstrophy",
    "palinstrophy",
    "vorticity_mean",
    "vorticity_std",
    "vorticity_min",
    "vorticity_max",
    "vorticity_linf",
    "circulation",
    "divergence_linf",
]


def frame_diagnostics(solver: SpectralNavierStokes2D, omega: np.ndarray) -> Dict[str, float]:
    """Return scalar diagnostics for one vorticity frame."""

    return solver.diagnostics(np.asarray(omega, dtype=np.float64))


def trajectory_diagnostics(
    solver: SpectralNavierStokes2D,
    trajectory: np.ndarray,
    dt: float,
    keep_every: int = 1,
) -> List[Dict[str, float]]:
    """Return diagnostics for each stored frame in a trajectory."""

    if trajectory.ndim != 3:
        raise ValueError("expected trajectory with shape [time, n, n]")
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    if keep_every <= 0:
        raise ValueError("keep_every must be positive")

    rows = []
    for frame_index, omega in enumerate(trajectory):
        step = frame_index * keep_every
        row = frame_diagnostics(solver, omega)
        row["step"] = float(step)
        row["time"] = float(step * dt)
        rows.append(row)
    return rows


def diagnostics_to_table(
    diagnostics: Sequence[Mapping[str, float]],
    keys: Sequence[str] = DIAGNOSTIC_KEYS,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert diagnostic dicts into a stable [time, metric] matrix."""

    names = np.array(list(keys), dtype="U32")
    values = np.empty((len(diagnostics), len(keys)), dtype=np.float64)
    for row_index, row in enumerate(diagnostics):
        for key_index, key in enumerate(keys):
            values[row_index, key_index] = float(row.get(key, np.nan))
    return names, values


def write_diagnostics_json(path: str | Path, diagnostics: Sequence[Mapping[str, float]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = [{key: float(value) for key, value in row.items()} for row in diagnostics]
    path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")


def comparison_errors(reference: np.ndarray, candidate: np.ndarray) -> Dict[str, np.ndarray]:
    """Return frame-wise error curves between two trajectories."""

    if reference.shape != candidate.shape:
        raise ValueError(f"expected matching trajectory shapes, got {reference.shape} and {candidate.shape}")

    diff = np.asarray(candidate, dtype=np.float64) - np.asarray(reference, dtype=np.float64)
    reduction_axes = tuple(range(1, diff.ndim))
    mse = np.mean(diff * diff, axis=reduction_axes)
    rmse = np.sqrt(mse)
    max_abs = np.max(np.abs(diff), axis=reduction_axes)
    reference_norm = np.sqrt(np.mean(np.asarray(reference, dtype=np.float64) ** 2, axis=reduction_axes))
    relative_l2 = np.divide(rmse, reference_norm, out=np.zeros_like(rmse), where=reference_norm > 0.0)
    return {
        "mse": mse,
        "rmse": rmse,
        "relative_l2": relative_l2,
        "max_abs": max_abs,
    }


def energy_spectrum(solver: SpectralNavierStokes2D, omega: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return isotropic kinetic-energy spectrum binned by integer wavenumber."""

    omega_hat = np.fft.fft2(np.asarray(omega, dtype=np.float64))
    u_hat = 1j * solver.ky * solver.streamfunction_hat(omega_hat)
    v_hat = -1j * solver.kx * solver.streamfunction_hat(omega_hat)
    energy_density = 0.5 * (np.abs(u_hat) ** 2 + np.abs(v_hat) ** 2) / float(solver.n**4)

    base_wavenumber = 2.0 * np.pi / solver.length
    radial_index = np.rint(np.sqrt(solver.kx * solver.kx + solver.ky * solver.ky) / base_wavenumber).astype(int)
    spectrum = np.bincount(radial_index.ravel(), weights=energy_density.ravel())
    wavenumbers = np.arange(spectrum.shape[0], dtype=np.float64)
    return wavenumbers, spectrum


def finite_metric_range(values: Iterable[float]) -> Tuple[float, float]:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return 0.0, 1.0
    return float(np.min(array)), float(np.max(array))
