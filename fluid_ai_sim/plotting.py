from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .diagnostics import comparison_errors, diagnostics_to_table, energy_spectrum
from .solver import SpectralNavierStokes2D


def _pyplot():
    cache_root = Path(tempfile.gettempdir()) / "fluid_ai_sim_matplotlib"
    mpl_config = cache_root / "mpl"
    xdg_cache = cache_root / "xdg"
    mpl_config.mkdir(parents=True, exist_ok=True)
    xdg_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config))
    os.environ.setdefault("XDG_CACHE_HOME", str(xdg_cache))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _diagnostic_column(diagnostics: Sequence[Mapping[str, float]], key: str) -> np.ndarray:
    return np.array([float(row[key]) for row in diagnostics], dtype=np.float64)


def _time_axis(diagnostics: Sequence[Mapping[str, float]]) -> np.ndarray:
    if diagnostics and "time" in diagnostics[0]:
        return _diagnostic_column(diagnostics, "time")
    return np.arange(len(diagnostics), dtype=np.float64)


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    return path


def plot_vorticity_frames(
    trajectory: np.ndarray,
    path: str | Path,
    title: str = "Vorticity snapshots",
    max_frames: int = 6,
) -> Path:
    plt = _pyplot()
    path = Path(path)

    frame_count = min(max_frames, trajectory.shape[0])
    indices = np.linspace(0, trajectory.shape[0] - 1, num=frame_count, dtype=int)
    columns = min(3, frame_count)
    rows = int(np.ceil(frame_count / columns))
    scale = np.percentile(np.abs(trajectory), 98.0)
    if scale <= 1.0e-12:
        scale = 1.0

    fig, axes = plt.subplots(rows, columns, figsize=(4.0 * columns, 3.5 * rows), squeeze=False)
    for axis in axes.ravel():
        axis.axis("off")
    for axis, trajectory_index in zip(axes.ravel(), indices):
        image = axis.imshow(
            trajectory[trajectory_index],
            cmap="RdBu_r",
            vmin=-scale,
            vmax=scale,
            origin="lower",
            interpolation="nearest",
        )
        axis.set_title(f"step {int(trajectory_index)}")
    fig.suptitle(title)
    fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.72, label="vorticity")
    saved = _save(fig, path)
    plt.close(fig)
    return saved


def plot_energy_enstrophy(diagnostics: Sequence[Mapping[str, float]], path: str | Path) -> Path:
    plt = _pyplot()
    path = Path(path)
    time = _time_axis(diagnostics)

    fig, axes = plt.subplots(2, 1, figsize=(8.0, 6.0), sharex=True)
    axes[0].plot(time, _diagnostic_column(diagnostics, "kinetic_energy"), color="#2563eb")
    axes[0].set_ylabel("kinetic energy")
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(time, _diagnostic_column(diagnostics, "enstrophy"), color="#dc2626")
    axes[1].set_ylabel("enstrophy")
    axes[1].set_xlabel("time")
    axes[1].grid(True, alpha=0.3)
    fig.suptitle("Energy and enstrophy")
    saved = _save(fig, path)
    plt.close(fig)
    return saved


def plot_vorticity_range(diagnostics: Sequence[Mapping[str, float]], path: str | Path) -> Path:
    plt = _pyplot()
    path = Path(path)
    time = _time_axis(diagnostics)

    fig, axis = plt.subplots(figsize=(8.0, 4.0))
    axis.plot(time, _diagnostic_column(diagnostics, "vorticity_min"), label="min", color="#1d4ed8")
    axis.plot(time, _diagnostic_column(diagnostics, "vorticity_max"), label="max", color="#b91c1c")
    axis.plot(time, _diagnostic_column(diagnostics, "vorticity_linf"), label="max abs", color="#111827")
    axis.set_xlabel("time")
    axis.set_ylabel("vorticity")
    axis.set_title("Vorticity range")
    axis.grid(True, alpha=0.3)
    axis.legend()
    saved = _save(fig, path)
    plt.close(fig)
    return saved


def plot_divergence(diagnostics: Sequence[Mapping[str, float]], path: str | Path) -> Path:
    plt = _pyplot()
    path = Path(path)
    time = _time_axis(diagnostics)
    divergence = np.maximum(_diagnostic_column(diagnostics, "divergence_linf"), 1.0e-18)

    fig, axis = plt.subplots(figsize=(8.0, 4.0))
    axis.semilogy(time, divergence, color="#0f766e")
    axis.set_xlabel("time")
    axis.set_ylabel("max |div u|")
    axis.set_title("Divergence check")
    axis.grid(True, which="both", alpha=0.3)
    saved = _save(fig, path)
    plt.close(fig)
    return saved


def plot_energy_spectrum(
    solver: SpectralNavierStokes2D,
    trajectory: np.ndarray,
    path: str | Path,
) -> Path:
    plt = _pyplot()
    path = Path(path)
    initial_k, initial_spectrum = energy_spectrum(solver, trajectory[0])
    final_k, final_spectrum = energy_spectrum(solver, trajectory[-1])

    fig, axis = plt.subplots(figsize=(8.0, 4.5))
    axis.loglog(initial_k[1:], np.maximum(initial_spectrum[1:], 1.0e-30), label="initial", color="#2563eb")
    axis.loglog(final_k[1:], np.maximum(final_spectrum[1:], 1.0e-30), label="final", color="#dc2626")
    axis.set_xlabel("wavenumber")
    axis.set_ylabel("kinetic energy")
    axis.set_title("Energy spectrum")
    axis.grid(True, which="both", alpha=0.3)
    axis.legend()
    saved = _save(fig, path)
    plt.close(fig)
    return saved


def plot_run_summary(
    solver: SpectralNavierStokes2D,
    trajectory: np.ndarray,
    diagnostics: Sequence[Mapping[str, float]],
    out_dir: str | Path,
    max_frames: int = 6,
) -> list[Path]:
    out_dir = Path(out_dir)
    return [
        plot_vorticity_frames(trajectory, out_dir / "vorticity_frames.png", max_frames=max_frames),
        plot_energy_enstrophy(diagnostics, out_dir / "energy_enstrophy.png"),
        plot_vorticity_range(diagnostics, out_dir / "vorticity_range.png"),
        plot_divergence(diagnostics, out_dir / "divergence.png"),
        plot_energy_spectrum(solver, trajectory, out_dir / "energy_spectrum.png"),
    ]


def plot_comparison_errors(
    reference: np.ndarray,
    trajectories: Mapping[str, np.ndarray],
    path: str | Path,
    dt: float,
    correction_interval: int | None = None,
) -> Path:
    plt = _pyplot()
    path = Path(path)
    time = np.arange(reference.shape[0], dtype=np.float64) * dt

    fig, axes = plt.subplots(2, 1, figsize=(8.0, 6.0), sharex=True)
    for label, trajectory in trajectories.items():
        if label == "solver":
            continue
        errors = comparison_errors(reference, trajectory)
        axes[0].plot(time, errors["rmse"], label=label)
        axes[1].plot(time, errors["relative_l2"], label=label)

    if correction_interval:
        for axis in axes:
            for step in range(correction_interval, reference.shape[0], correction_interval):
                axis.axvline(step * dt, color="#9ca3af", alpha=0.25, linewidth=0.8)

    axes[0].set_ylabel("RMSE")
    axes[0].grid(True, alpha=0.3)
    axes[1].set_ylabel("relative L2")
    axes[1].set_xlabel("time")
    axes[1].grid(True, alpha=0.3)
    axes[0].legend()
    fig.suptitle("Error against exact solver")
    saved = _save(fig, path)
    plt.close(fig)
    return saved


def plot_comparison_diagnostics(
    diagnostics_by_label: Mapping[str, Sequence[Mapping[str, float]]],
    path: str | Path,
) -> Path:
    plt = _pyplot()
    path = Path(path)

    fig, axes = plt.subplots(2, 1, figsize=(8.0, 6.0), sharex=True)
    for label, diagnostics in diagnostics_by_label.items():
        time = _time_axis(diagnostics)
        axes[0].plot(time, _diagnostic_column(diagnostics, "kinetic_energy"), label=label)
        axes[1].plot(time, _diagnostic_column(diagnostics, "enstrophy"), label=label)
    axes[0].set_ylabel("kinetic energy")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    axes[1].set_ylabel("enstrophy")
    axes[1].set_xlabel("time")
    axes[1].grid(True, alpha=0.3)
    fig.suptitle("Mode diagnostics")
    saved = _save(fig, path)
    plt.close(fig)
    return saved


def plot_final_vorticity_comparison(
    trajectories: Mapping[str, np.ndarray],
    path: str | Path,
) -> Path:
    plt = _pyplot()
    path = Path(path)
    labels = list(trajectories.keys())
    scale = np.percentile(np.abs(np.stack([trajectories[label][-1] for label in labels])), 98.0)
    if scale <= 1.0e-12:
        scale = 1.0

    fig, axes = plt.subplots(1, len(labels), figsize=(4.0 * len(labels), 3.5), squeeze=False)
    for axis, label in zip(axes.ravel(), labels):
        image = axis.imshow(
            trajectories[label][-1],
            cmap="RdBu_r",
            vmin=-scale,
            vmax=scale,
            origin="lower",
            interpolation="nearest",
        )
        axis.set_title(label)
        axis.axis("off")
    fig.suptitle("Final vorticity by mode")
    fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.75, label="vorticity")
    saved = _save(fig, path)
    plt.close(fig)
    return saved


def plot_speed_comparison(metadata_by_label: Mapping[str, Mapping[str, object]], path: str | Path) -> Path:
    plt = _pyplot()
    path = Path(path)
    labels = list(metadata_by_label.keys())
    speeds = [float(metadata_by_label[label].get("steps_per_second") or 0.0) for label in labels]

    fig, axis = plt.subplots(figsize=(7.0, 4.0))
    axis.bar(labels, speeds, color=["#2563eb", "#dc2626", "#0f766e", "#7c3aed"][: len(labels)])
    axis.set_ylabel("steps per second")
    axis.set_title("Runtime speed")
    axis.grid(True, axis="y", alpha=0.3)
    saved = _save(fig, path)
    plt.close(fig)
    return saved


def plot_comparison_summary(
    solver_reference: np.ndarray,
    trajectories: Mapping[str, np.ndarray],
    diagnostics_by_label: Mapping[str, Sequence[Mapping[str, float]]],
    metadata_by_label: Mapping[str, Mapping[str, object]],
    out_dir: str | Path,
    dt: float,
    correction_interval: int | None = None,
) -> list[Path]:
    out_dir = Path(out_dir)
    return [
        plot_comparison_errors(
            solver_reference,
            trajectories,
            out_dir / "error_to_solver.png",
            dt=dt,
            correction_interval=correction_interval,
        ),
        plot_comparison_diagnostics(diagnostics_by_label, out_dir / "mode_energy_enstrophy.png"),
        plot_final_vorticity_comparison(trajectories, out_dir / "final_vorticity.png"),
        plot_speed_comparison(metadata_by_label, out_dir / "speed_comparison.png"),
    ]


def save_diagnostics_csv(
    diagnostics: Sequence[Mapping[str, float]],
    path: str | Path,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    names, values = diagnostics_to_table(diagnostics)
    header = ",".join(names.tolist())
    np.savetxt(path, values, delimiter=",", header=header, comments="")
    return path
