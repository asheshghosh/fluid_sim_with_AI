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
from .heat import HeatSolverConfig, SpectralHeatEquation2D, random_temperature_field
from .surrogate import detensorize, load_checkpoint, load_checkpoint_metadata, tensorize


DIAGNOSTIC_KEYS = [
    "step",
    "time",
    "temperature_mean",
    "temperature_std",
    "temperature_min",
    "temperature_max",
    "temperature_linf",
    "absolute_temperature_mean",
    "absolute_temperature_max",
    "thermal_energy",
    "thermal_variance",
    "gradient_rms",
    "hotspot_area_fraction",
    "source_power_density",
    "cooling_power_density",
    "net_power_density",
    "flux_rms",
    "source_temperature_correlation",
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


def _solver_config_from_dict(data: dict, base: HeatSolverConfig) -> HeatSolverConfig:
    allowed = set(base.to_dict().keys())
    return HeatSolverConfig(**{**base.to_dict(), **{key: value for key, value in data.items() if key in allowed}})


def _state_from_temperature(solver: SpectralHeatEquation2D, trajectory: np.ndarray) -> np.ndarray:
    source = np.broadcast_to(solver.source, trajectory.shape)
    return np.stack([trajectory, source], axis=1)


def _trajectory_diagnostics(
    solver: SpectralHeatEquation2D,
    trajectory: np.ndarray,
    dt: float,
    keep_every: int,
) -> list[dict]:
    rows = []
    for frame_index, temperature in enumerate(trajectory):
        step = frame_index * keep_every
        row = solver.diagnostics(temperature)
        row["step"] = float(step)
        row["time"] = float(step * dt)
        rows.append(row)
    return rows


def _save_diagnostics_csv(diagnostics: Sequence[Mapping[str, float]], path: Path) -> None:
    names, values = diagnostics_to_table(diagnostics, keys=DIAGNOSTIC_KEYS)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, values, delimiter=",", header=",".join(names.tolist()), comments="")


def _predict_state(
    model,
    solver: SpectralHeatEquation2D,
    state: np.ndarray,
    mean: float,
    std: float,
    device: torch.device,
) -> np.ndarray:
    current = tensorize(state[None, :, :, :], mean, std, device)
    predicted = detensorize(model(current), mean, std)[0].astype(np.float64)
    predicted[1] = solver.source
    return predicted


def _run_ai(
    model,
    solver: SpectralHeatEquation2D,
    temperature0: np.ndarray,
    model_steps: int,
    mean: float,
    std: float,
    device: torch.device,
) -> np.ndarray:
    state = np.stack([temperature0, solver.source], axis=0).astype(np.float64)
    frames = [state[0].copy()]
    model.eval()
    with torch.no_grad():
        for _ in range(model_steps):
            state = _predict_state(model, solver, state, mean, std, device)
            frames.append(state[0].copy())
    return np.stack(frames, axis=0)


def _run_hybrid(
    model,
    solver: SpectralHeatEquation2D,
    temperature0: np.ndarray,
    model_steps: int,
    surrogate_step_size: int,
    mean: float,
    std: float,
    device: torch.device,
    correction_interval: int,
) -> np.ndarray:
    if correction_interval <= 0:
        raise ValueError("correction_interval must be positive")

    state = np.stack([temperature0, solver.source], axis=0).astype(np.float64)
    frames = [state[0].copy()]
    model.eval()
    with torch.no_grad():
        for step in range(1, model_steps + 1):
            if step % correction_interval == 0:
                state[0] = solver.rollout(state[0], steps=surrogate_step_size)[-1]
                state[1] = solver.source
            else:
                state = _predict_state(model, solver, state, mean, std, device)
            frames.append(state[0].copy())
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
    solver: SpectralHeatEquation2D,
    seconds: float,
    solver_equivalent_steps: int,
    keep_every: int,
    args: argparse.Namespace,
) -> tuple[list[dict], dict]:
    run_dir = out / label
    run_dir.mkdir(parents=True, exist_ok=True)

    diagnostics = _trajectory_diagnostics(solver, trajectory, dt=solver.config.dt, keep_every=keep_every)
    diagnostic_names, diagnostic_values = diagnostics_to_table(diagnostics, keys=DIAGNOSTIC_KEYS)
    gradients = np.stack([solver.gradient(frame) for frame in trajectory], axis=0)
    flux = np.stack([solver.heat_flux(frame) for frame in trajectory], axis=0)
    np.savez_compressed(
        run_dir / "trajectory.npz",
        temperature=trajectory,
        absolute_temperature=trajectory + solver.config.ambient_temperature,
        source=solver.source,
        gradients=gradients,
        flux=flux,
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
    solver: SpectralHeatEquation2D,
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
    axes[0].set_ylabel("temperature RMSE")
    axes[1].set_ylabel("relative L2")
    axes[1].set_xlabel("time")
    for axis in axes:
        axis.grid(True, alpha=0.3)
        axis.legend()
    fig.suptitle("Temperature error against spectral heat solver")
    path = out_dir / "temperature_error_to_solver.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)

    fig, axes = plt.subplots(3, 1, figsize=(8.0, 8.0), sharex=True)
    for label, diagnostics in diagnostics_by_label.items():
        time = np.array([row["time"] for row in diagnostics], dtype=np.float64)
        axes[0].plot(time, [row["temperature_mean"] for row in diagnostics], label=label)
        axes[1].plot(time, [row["temperature_max"] for row in diagnostics], label=label)
        axes[2].plot(time, [row["gradient_rms"] for row in diagnostics], label=label)
    axes[0].set_ylabel("mean rise")
    axes[1].set_ylabel("max rise")
    axes[2].set_ylabel("gradient RMS")
    axes[2].set_xlabel("time")
    for axis in axes:
        axis.grid(True, alpha=0.3)
        axis.legend()
    fig.suptitle("Thermal diagnostics")
    path = out_dir / "thermal_diagnostics.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)

    labels = list(trajectories.keys())
    fields = [solver.source] + [trajectories[label][-1] for label in labels]
    temp_scale = np.percentile(np.abs(np.stack([field for field in fields[1:]])), 98.0)
    if temp_scale <= 1.0e-12:
        temp_scale = 1.0
    temp_vmax = max(temp_scale, max(float(np.max(field)) for field in fields[1:]))
    fig, axes = plt.subplots(1, len(fields), figsize=(4.0 * len(fields), 3.6), squeeze=False)
    image = None
    for axis, title, field in zip(axes.ravel(), ["source", *labels], fields):
        if title == "source":
            image = axis.imshow(field, cmap="magma", origin="lower", interpolation="nearest")
        else:
            image = axis.imshow(
                field,
                cmap="inferno",
                vmin=0.0,
                vmax=temp_vmax,
                origin="lower",
                interpolation="nearest",
            )
        axis.set_title(title)
        axis.axis("off")
    fig.suptitle("Heat source and final temperature rise")
    fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.75)
    path = out_dir / "final_temperature_source.png"
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

    base_config = HeatSolverConfig(
        n=args.n,
        diffusivity=args.diffusivity,
        dt=args.dt,
        sink_rate=args.sink_rate,
        source_amplitude=args.source_amplitude,
        source_width=args.source_width,
        source_count=args.source_count,
        source_seed=args.source_seed,
        ambient_temperature=args.ambient_temperature,
    )
    config = _solver_config_from_dict(checkpoint_solver_config, base_config) if args.use_checkpoint_config else base_config
    solver = SpectralHeatEquation2D(config)

    temperature0 = random_temperature_field(
        config.n,
        seed=args.seed,
        length=config.length,
        low_pass=max(3, config.n // 8),
        amplitude=args.amplitude,
        bias=args.bias,
    )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    trajectories = {}
    timings = {}

    start = time.perf_counter()
    trajectories["solver"] = solver.rollout(temperature0, steps=args.steps, keep_every=surrogate_step_size)
    timings["solver"] = time.perf_counter() - start

    start = time.perf_counter()
    trajectories["ai"] = _run_ai(model, solver, temperature0, model_steps, mean, std, device)
    timings["ai"] = time.perf_counter() - start

    start = time.perf_counter()
    trajectories["hybrid"] = _run_hybrid(
        model,
        solver,
        temperature0,
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

    states = {label: _state_from_temperature(solver, trajectory) for label, trajectory in trajectories.items()}
    np.savez_compressed(out / "state_trajectories.npz", **states)
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
                "equation": "heat_2d_periodic_source_conditioned",
                "steps": args.steps,
                "model_steps": model_steps,
                "surrogate_step_size": surrogate_step_size,
                "device": device.type,
                "solver_config": config.to_dict(),
                "checkpoint": args.checkpoint,
                "correction_interval": args.correction_interval,
                "state_channels": ["temperature", "source"],
                "modes": list(trajectories.keys()),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"wrote heat comparison: {out}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare exact, AI, and hybrid source-conditioned heat rollouts.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--use-checkpoint-config", action="store_true")
    parser.add_argument("--out", default="runs/heat_comparison")
    parser.add_argument("--n", type=int, default=32)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument(
        "--surrogate-step-size",
        type=int,
        default=None,
        help="Solver steps represented by each AI inference; defaults to checkpoint metadata.",
    )
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--amplitude", type=float, default=0.15)
    parser.add_argument("--bias", type=float, default=0.0)
    parser.add_argument("--dt", type=float, default=2.0e-2)
    parser.add_argument("--diffusivity", type=float, default=5.0e-3)
    parser.add_argument("--sink-rate", type=float, default=2.0e-3)
    parser.add_argument("--source-amplitude", type=float, default=1.0)
    parser.add_argument("--source-width", type=float, default=0.22)
    parser.add_argument("--source-count", type=int, default=4)
    parser.add_argument("--source-seed", type=int, default=0)
    parser.add_argument("--ambient-temperature", type=float, default=300.0)
    parser.add_argument("--correction-interval", type=int, default=5)
    parser.add_argument("--no-plots", action="store_true")
    return parser


def main() -> None:
    compare(build_parser().parse_args())


if __name__ == "__main__":
    main()
