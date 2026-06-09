from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Dict, Mapping, Sequence

import numpy as np
import torch

from .diagnostics import comparison_errors, diagnostics_to_table, write_diagnostics_json
from .incompressible import (
    SpectralIncompressibleNavierStokes2D,
    VelocitySolverConfig,
    random_divergence_free_velocity,
)
from .surrogate import detensorize, load_checkpoint, load_checkpoint_metadata, tensorize


DIAGNOSTIC_KEYS = [
    "step",
    "time",
    "kinetic_energy",
    "enstrophy",
    "velocity_rms",
    "speed_mean",
    "speed_max",
    "vorticity_mean",
    "vorticity_std",
    "vorticity_min",
    "vorticity_max",
    "vorticity_linf",
    "divergence_linf",
]


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


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


def _solver_config_from_dict(data: dict, base: VelocitySolverConfig) -> VelocitySolverConfig:
    allowed = set(base.to_dict().keys())
    return VelocitySolverConfig(**{**base.to_dict(), **{key: value for key, value in data.items() if key in allowed}})


def _trajectory_diagnostics(
    solver: SpectralIncompressibleNavierStokes2D,
    trajectory: np.ndarray,
    dt: float,
    keep_every: int,
) -> list[dict]:
    rows = []
    for frame_index, velocity in enumerate(trajectory):
        step = frame_index * keep_every
        row = solver.diagnostics(velocity)
        row["step"] = float(step)
        row["time"] = float(step * dt)
        rows.append(row)
    return rows


def _save_diagnostics_csv(diagnostics: Sequence[Mapping[str, float]], path: Path) -> None:
    names, values = diagnostics_to_table(diagnostics, keys=DIAGNOSTIC_KEYS)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, values, delimiter=",", header=",".join(names.tolist()), comments="")


def _predict_velocity(
    model,
    solver: SpectralIncompressibleNavierStokes2D,
    velocity: np.ndarray,
    mean: float,
    std: float,
    device: torch.device,
) -> np.ndarray:
    current = tensorize(velocity[None, :, :, :], mean, std, device)
    predicted = detensorize(model(current), mean, std)[0].astype(np.float64)
    return solver.project_velocity(predicted)


def _run_ai(
    model,
    solver: SpectralIncompressibleNavierStokes2D,
    velocity0: np.ndarray,
    model_steps: int,
    mean: float,
    std: float,
    device: torch.device,
) -> np.ndarray:
    velocity = solver.project_velocity(velocity0)
    frames = [velocity.copy()]
    model.eval()
    with torch.no_grad():
        for _ in range(model_steps):
            velocity = _predict_velocity(model, solver, velocity, mean, std, device)
            frames.append(velocity.copy())
    return np.stack(frames, axis=0)


def _run_hybrid(
    model,
    solver: SpectralIncompressibleNavierStokes2D,
    velocity0: np.ndarray,
    model_steps: int,
    surrogate_step_size: int,
    mean: float,
    std: float,
    device: torch.device,
    correction_interval: int,
) -> np.ndarray:
    if correction_interval <= 0:
        raise ValueError("correction_interval must be positive")

    velocity = solver.project_velocity(velocity0)
    frames = [velocity.copy()]
    model.eval()
    with torch.no_grad():
        for step in range(1, model_steps + 1):
            if step % correction_interval == 0:
                velocity = solver.rollout(velocity, steps=surrogate_step_size)[-1]
            else:
                velocity = _predict_velocity(model, solver, velocity, mean, std, device)
            frames.append(velocity.copy())
    return np.stack(frames, axis=0)


def _save_comparison_metrics(out: Path, reference: np.ndarray, trajectories: Dict[str, np.ndarray], sample_dt: float) -> None:
    arrays = {
        "step": np.arange(reference.shape[0], dtype=np.int64),
        "time": np.arange(reference.shape[0], dtype=np.float64) * sample_dt,
    }
    summary = {}
    for label, trajectory in trajectories.items():
        if label == "solver":
            continue
        errors = comparison_errors(reference, trajectory)
        for metric, values in errors.items():
            arrays[f"{label}_{metric}"] = values
        summary[label] = {f"final_{metric}": float(values[-1]) for metric, values in errors.items()}
    np.savez_compressed(out / "comparison_metrics.npz", **arrays)
    (out / "comparison_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def _save_run(
    out: Path,
    label: str,
    trajectory: np.ndarray,
    solver: SpectralIncompressibleNavierStokes2D,
    seconds: float,
    solver_equivalent_steps: int,
    keep_every: int,
    args: argparse.Namespace,
) -> tuple[list[dict], dict]:
    run_dir = out / label
    run_dir.mkdir(parents=True, exist_ok=True)

    diagnostics = _trajectory_diagnostics(solver, trajectory, dt=solver.config.dt, keep_every=keep_every)
    diagnostic_names, diagnostic_values = diagnostics_to_table(diagnostics, keys=DIAGNOSTIC_KEYS)
    vorticity = np.stack([solver.vorticity(frame) for frame in trajectory], axis=0)
    np.savez_compressed(
        run_dir / "trajectory.npz",
        velocity=trajectory,
        vorticity=vorticity,
        diagnostics=diagnostics,
        diagnostic_names=diagnostic_names,
        diagnostic_values=diagnostic_values,
    )
    write_diagnostics_json(run_dir / "diagnostics.json", diagnostics)
    _save_diagnostics_csv(diagnostics, run_dir / "diagnostics.csv")

    metadata = {
        "mode": label,
        "steps": solver_equivalent_steps,
        "stored_frames": int(trajectory.shape[0]),
        "surrogate_step_size": keep_every,
        "seconds": seconds,
        "steps_per_second": solver_equivalent_steps / seconds if seconds > 0.0 else None,
        "solver_config": solver.config.to_dict(),
        "checkpoint": args.checkpoint,
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return diagnostics, metadata


def _plot_comparison_summary(
    solver: SpectralIncompressibleNavierStokes2D,
    trajectories: Mapping[str, np.ndarray],
    diagnostics_by_label: Mapping[str, Sequence[Mapping[str, float]]],
    metadata_by_label: Mapping[str, Mapping[str, object]],
    out_dir: Path,
    dt: float,
) -> list[Path]:
    plt = _pyplot()
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    time_axis = np.arange(trajectories["solver"].shape[0], dtype=np.float64) * dt
    fig, axes = plt.subplots(2, 1, figsize=(8.0, 6.0), sharex=True)
    for label, trajectory in trajectories.items():
        if label == "solver":
            continue
        errors = comparison_errors(trajectories["solver"], trajectory)
        axes[0].plot(time_axis, errors["rmse"], label=label)
        axes[1].plot(time_axis, errors["relative_l2"], label=label)
    axes[0].set_ylabel("velocity RMSE")
    axes[1].set_ylabel("relative L2")
    axes[1].set_xlabel("time")
    for axis in axes:
        axis.grid(True, alpha=0.3)
        axis.legend()
    fig.suptitle("Velocity error against projected spectral solver")
    path = out_dir / "velocity_error_to_solver.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)

    fig, axes = plt.subplots(2, 1, figsize=(8.0, 6.0), sharex=True)
    for label, diagnostics in diagnostics_by_label.items():
        time = np.array([row["time"] for row in diagnostics], dtype=np.float64)
        axes[0].plot(time, [row["kinetic_energy"] for row in diagnostics], label=label)
        axes[1].semilogy(time, np.maximum([row["divergence_linf"] for row in diagnostics], 1.0e-18), label=label)
    axes[0].set_ylabel("kinetic energy")
    axes[1].set_ylabel("max |div u|")
    axes[1].set_xlabel("time")
    for axis in axes:
        axis.grid(True, which="both", alpha=0.3)
        axis.legend()
    fig.suptitle("Energy and incompressibility")
    path = out_dir / "energy_divergence.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)

    labels = list(trajectories.keys())
    final_vorticity = {label: solver.vorticity(trajectory[-1]) for label, trajectory in trajectories.items()}
    scale = np.percentile(np.abs(np.stack([final_vorticity[label] for label in labels])), 98.0)
    if scale <= 1.0e-12:
        scale = 1.0
    fig, axes = plt.subplots(1, len(labels), figsize=(4.0 * len(labels), 3.5), squeeze=False)
    for axis, label in zip(axes.ravel(), labels):
        image = axis.imshow(
            final_vorticity[label],
            cmap="RdBu_r",
            vmin=-scale,
            vmax=scale,
            origin="lower",
            interpolation="nearest",
        )
        axis.set_title(label)
        axis.axis("off")
    fig.suptitle("Final vorticity from velocity state")
    fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.75, label="vorticity")
    path = out_dir / "final_vorticity.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)

    fig, axis = plt.subplots(figsize=(7.0, 4.0))
    labels = list(metadata_by_label.keys())
    speeds = [float(metadata_by_label[label].get("steps_per_second") or 0.0) for label in labels]
    axis.bar(labels, speeds, color=["#2563eb", "#dc2626", "#0f766e"][: len(labels)])
    axis.set_ylabel("solver-equivalent steps/sec")
    axis.set_title("Runtime speed")
    axis.grid(True, axis="y", alpha=0.3)
    path = out_dir / "speed_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)

    return paths


def compare(args: argparse.Namespace) -> None:
    device = _device()
    model, mean, std, checkpoint_solver_config = load_checkpoint(args.checkpoint, device=device)
    checkpoint_metadata = load_checkpoint_metadata(args.checkpoint)
    surrogate_step_size = args.surrogate_step_size or int(checkpoint_metadata.get("surrogate_step_size", 1))
    if surrogate_step_size <= 0:
        raise ValueError("surrogate_step_size must be positive")
    if args.steps % surrogate_step_size != 0:
        raise ValueError("--steps must be divisible by the surrogate step size for aligned comparison")
    model_steps = args.steps // surrogate_step_size

    base_config = VelocitySolverConfig(
        n=args.n,
        viscosity=args.viscosity,
        dt=args.dt,
        forcing_amplitude=args.forcing_amplitude,
        forcing_wavenumber=args.forcing_wavenumber,
    )
    config = _solver_config_from_dict(checkpoint_solver_config, base_config) if args.use_checkpoint_config else base_config
    solver = SpectralIncompressibleNavierStokes2D(config)

    velocity0 = random_divergence_free_velocity(
        config.n,
        seed=args.seed,
        length=config.length,
        low_pass=max(3, config.n // 8),
        amplitude=args.amplitude,
    )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    trajectories = {}
    timings = {}

    start = time.perf_counter()
    trajectories["solver"] = solver.rollout(velocity0, steps=args.steps, keep_every=surrogate_step_size)
    timings["solver"] = time.perf_counter() - start

    start = time.perf_counter()
    trajectories["ai"] = _run_ai(model, solver, velocity0, model_steps, mean, std, device)
    timings["ai"] = time.perf_counter() - start

    start = time.perf_counter()
    trajectories["hybrid"] = _run_hybrid(
        model,
        solver,
        velocity0,
        model_steps,
        surrogate_step_size,
        mean,
        std,
        device,
        correction_interval=args.correction_interval,
    )
    timings["hybrid"] = time.perf_counter() - start

    diagnostics_by_label = {}
    metadata_by_label = {}
    for label, trajectory in trajectories.items():
        diagnostics, metadata = _save_run(
            out,
            label,
            trajectory,
            solver,
            timings[label],
            solver_equivalent_steps=args.steps,
            keep_every=surrogate_step_size,
            args=args,
        )
        diagnostics_by_label[label] = diagnostics
        metadata_by_label[label] = metadata

    _save_comparison_metrics(out, trajectories["solver"], trajectories, config.dt * surrogate_step_size)
    if not args.no_plots:
        plot_paths = _plot_comparison_summary(
            solver,
            trajectories,
            diagnostics_by_label,
            metadata_by_label,
            out / "plots",
            dt=config.dt * surrogate_step_size,
        )
        print("comparison plots:")
        for plot_path in plot_paths:
            print(f"  {plot_path}")

    (out / "metadata.json").write_text(
        json.dumps(
            {
                "equation": "velocity_incompressible_2d_periodic",
                "steps": args.steps,
                "model_steps": model_steps,
                "surrogate_step_size": surrogate_step_size,
                "device": device.type,
                "solver_config": config.to_dict(),
                "checkpoint": args.checkpoint,
                "correction_interval": args.correction_interval,
                "modes": list(trajectories.keys()),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"wrote incompressible comparison: {out}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare exact, AI, and hybrid incompressible velocity rollouts.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--use-checkpoint-config", action="store_true")
    parser.add_argument("--out", default="runs/incompressible_comparison")
    parser.add_argument("--n", type=int, default=32)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument(
        "--surrogate-step-size",
        type=int,
        default=None,
        help="Solver steps represented by each AI inference; defaults to checkpoint metadata.",
    )
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--amplitude", type=float, default=1.0)
    parser.add_argument("--dt", type=float, default=1.0e-2)
    parser.add_argument("--viscosity", type=float, default=1.0e-3)
    parser.add_argument("--forcing-amplitude", type=float, default=0.0)
    parser.add_argument("--forcing-wavenumber", type=int, default=4)
    parser.add_argument("--correction-interval", type=int, default=10)
    parser.add_argument("--no-plots", action="store_true")
    return parser


def main() -> None:
    compare(build_parser().parse_args())


if __name__ == "__main__":
    main()
