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

from .active_nematic import ActiveNematicConfig, SpectralActiveNematic2D, random_active_nematic_state
from .diagnostics import comparison_errors, diagnostics_to_table, write_diagnostics_json
from .surrogate import detensorize, load_checkpoint, load_checkpoint_metadata, tensorize


DIAGNOSTIC_KEYS = [
    "step",
    "time",
    "kinetic_energy",
    "enstrophy",
    "velocity_rms",
    "speed_max",
    "q_order_mean",
    "q_order_std",
    "q_order_min",
    "q_order_max",
    "free_energy_density",
    "elastic_energy_density",
    "bulk_energy_density",
    "vorticity_linf",
    "divergence_linf",
    "active_length",
    "defects_plus_half",
    "defects_minus_half",
    "defect_total",
    "defect_density",
    "net_topological_charge",
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


def _config_from_args(args: argparse.Namespace) -> ActiveNematicConfig:
    return ActiveNematicConfig(
        n=args.n,
        viscosity=args.viscosity,
        dt=args.dt,
        activity=args.activity,
        elastic_constant=args.elastic_constant,
        rotational_diffusivity=args.rotational_diffusivity,
        bulk_strength=args.bulk_strength,
        preferred_order=args.preferred_order,
        flow_alignment=args.flow_alignment,
        max_order=args.max_order,
    )


def _config_from_dict(data: dict, base: ActiveNematicConfig) -> ActiveNematicConfig:
    allowed = set(base.to_dict().keys())
    return ActiveNematicConfig(**{**base.to_dict(), **{key: value for key, value in data.items() if key in allowed}})


def _trajectory_diagnostics(
    solver: SpectralActiveNematic2D,
    trajectory: np.ndarray,
    dt: float,
    keep_every: int,
) -> list[dict]:
    rows = []
    for frame_index, state in enumerate(trajectory):
        step = frame_index * keep_every
        row = solver.diagnostics(state)
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
    solver: SpectralActiveNematic2D,
    state: np.ndarray,
    mean: float,
    std: float,
    device: torch.device,
) -> np.ndarray:
    current = tensorize(state[None, :, :, :], mean, std, device)
    predicted = detensorize(model(current), mean, std)[0].astype(np.float64)
    return solver.project_state(predicted)


def _run_ai(
    model,
    solver: SpectralActiveNematic2D,
    state0: np.ndarray,
    model_steps: int,
    mean: float,
    std: float,
    device: torch.device,
) -> np.ndarray:
    state = solver.project_state(state0)
    frames = [state.copy()]
    model.eval()
    with torch.no_grad():
        for _ in range(model_steps):
            state = _predict_state(model, solver, state, mean, std, device)
            frames.append(state.copy())
    return np.stack(frames, axis=0)


def _run_hybrid(
    model,
    solver: SpectralActiveNematic2D,
    state0: np.ndarray,
    model_steps: int,
    surrogate_step_size: int,
    mean: float,
    std: float,
    device: torch.device,
    correction_interval: int,
) -> np.ndarray:
    if correction_interval <= 0:
        raise ValueError("correction_interval must be positive")

    state = solver.project_state(state0)
    frames = [state.copy()]
    model.eval()
    with torch.no_grad():
        for step in range(1, model_steps + 1):
            if step % correction_interval == 0:
                state = solver.rollout(state, steps=surrogate_step_size)[-1]
            else:
                state = _predict_state(model, solver, state, mean, std, device)
            frames.append(state.copy())
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
    solver: SpectralActiveNematic2D,
    seconds: float,
    solver_equivalent_steps: int,
    keep_every: int,
    args: argparse.Namespace,
) -> tuple[list[dict], dict]:
    run_dir = out / label
    run_dir.mkdir(parents=True, exist_ok=True)

    diagnostics = _trajectory_diagnostics(solver, trajectory, dt=solver.config.dt, keep_every=keep_every)
    diagnostic_names, diagnostic_values = diagnostics_to_table(diagnostics, keys=DIAGNOSTIC_KEYS)
    velocity = trajectory[:, :2]
    q = trajectory[:, 2:]
    vorticity = np.stack([solver.vorticity(frame) for frame in trajectory], axis=0)
    q_order = np.stack([solver.scalar_order(frame) for frame in trajectory], axis=0)
    director_angle = np.stack([solver.director_angle(frame) for frame in trajectory], axis=0)
    np.savez_compressed(
        run_dir / "trajectory.npz",
        state=trajectory,
        velocity=velocity,
        q=q,
        vorticity=vorticity,
        q_order=q_order,
        director_angle=director_angle,
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


def _diagnostic_column(diagnostics: Sequence[Mapping[str, float]], key: str) -> np.ndarray:
    return np.array([float(row[key]) for row in diagnostics], dtype=np.float64)


def _plot_comparison_summary(
    solver: SpectralActiveNematic2D,
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
    axes[0].set_ylabel("state RMSE")
    axes[1].set_ylabel("relative L2")
    axes[1].set_xlabel("time")
    for axis in axes:
        axis.grid(True, alpha=0.3)
        axis.legend()
    fig.suptitle("Active nematic state error against solver")
    path = out_dir / "state_error_to_solver.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)

    fig, axes = plt.subplots(3, 1, figsize=(8.0, 8.0), sharex=True)
    for label, diagnostics in diagnostics_by_label.items():
        time = _diagnostic_column(diagnostics, "time")
        axes[0].plot(time, _diagnostic_column(diagnostics, "kinetic_energy"), label=label)
        axes[1].plot(time, _diagnostic_column(diagnostics, "free_energy_density"), label=label)
        axes[2].semilogy(time, np.maximum(_diagnostic_column(diagnostics, "divergence_linf"), 1.0e-18), label=label)
    axes[0].set_ylabel("kinetic energy")
    axes[1].set_ylabel("free energy density")
    axes[2].set_ylabel("max |div u|")
    axes[2].set_xlabel("time")
    for axis in axes:
        axis.grid(True, which="both", alpha=0.3)
        axis.legend()
    fig.suptitle("Energy and incompressibility")
    path = out_dir / "energy_free_energy_divergence.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)

    fig, axes = plt.subplots(2, 1, figsize=(8.0, 6.0), sharex=True)
    for label, diagnostics in diagnostics_by_label.items():
        time = _diagnostic_column(diagnostics, "time")
        axes[0].plot(time, _diagnostic_column(diagnostics, "q_order_mean"), label=label)
        axes[1].plot(time, _diagnostic_column(diagnostics, "defect_total"), label=label)
    axes[0].set_ylabel("mean |Q|")
    axes[1].set_ylabel("defect count")
    axes[1].set_xlabel("time")
    for axis in axes:
        axis.grid(True, alpha=0.3)
        axis.legend()
    fig.suptitle("Nematic order and defects")
    path = out_dir / "order_defects.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)

    labels = list(trajectories.keys())
    final_order = {label: solver.scalar_order(trajectory[-1]) for label, trajectory in trajectories.items()}
    final_vorticity = {label: solver.vorticity(trajectory[-1]) for label, trajectory in trajectories.items()}
    order_scale = max(np.percentile(np.stack(list(final_order.values())), 98.0), 1.0e-6)
    vort_scale = max(np.percentile(np.abs(np.stack(list(final_vorticity.values()))), 98.0), 1.0e-6)
    fig, axes = plt.subplots(2, len(labels), figsize=(4.0 * len(labels), 6.5), squeeze=False)
    for col, label in enumerate(labels):
        order_image = axes[0, col].imshow(
            final_order[label],
            cmap="viridis",
            vmin=0.0,
            vmax=order_scale,
            origin="lower",
            interpolation="nearest",
        )
        axes[0, col].set_title(f"{label} |Q|")
        axes[0, col].axis("off")
        vort_image = axes[1, col].imshow(
            final_vorticity[label],
            cmap="RdBu_r",
            vmin=-vort_scale,
            vmax=vort_scale,
            origin="lower",
            interpolation="nearest",
        )
        axes[1, col].set_title(f"{label} vorticity")
        axes[1, col].axis("off")
    fig.colorbar(order_image, ax=axes[0].ravel().tolist(), shrink=0.75, label="|Q|")
    fig.colorbar(vort_image, ax=axes[1].ravel().tolist(), shrink=0.75, label="vorticity")
    fig.suptitle("Final active nematic fields")
    path = out_dir / "final_order_vorticity.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)

    fig, axis = plt.subplots(figsize=(7.0, 4.0))
    labels = list(metadata_by_label.keys())
    speeds = [float(metadata_by_label[label].get("steps_per_second") or 0.0) for label in labels]
    axis.bar(labels, speeds, color=["#2563eb", "#7c3aed", "#0f766e"][: len(labels)])
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

    base_config = _config_from_args(args)
    config = _config_from_dict(checkpoint_solver_config, base_config) if args.use_checkpoint_config else base_config
    solver = SpectralActiveNematic2D(config)
    state0 = random_active_nematic_state(
        config,
        seed=args.seed,
        velocity_amplitude=args.velocity_amplitude,
    )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    trajectories = {}
    timings = {}

    start = time.perf_counter()
    trajectories["solver"] = solver.rollout(state0, steps=args.steps, keep_every=surrogate_step_size)
    timings["solver"] = time.perf_counter() - start

    start = time.perf_counter()
    trajectories["ai"] = _run_ai(model, solver, state0, model_steps, mean, std, device)
    timings["ai"] = time.perf_counter() - start

    start = time.perf_counter()
    trajectories["hybrid"] = _run_hybrid(
        model,
        solver,
        state0,
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
                "equation": "active_nematic_beris_edwards_2d_periodic",
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
    print(f"wrote active nematic comparison: {out}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare solver, AI, and hybrid active nematic rollouts.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--use-checkpoint-config", action="store_true")
    parser.add_argument("--out", default="runs/active_nematic_compare")
    parser.add_argument("--n", type=int, default=32)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--surrogate-step-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--velocity-amplitude", type=float, default=0.15)
    parser.add_argument("--dt", type=float, default=5.0e-3)
    parser.add_argument("--viscosity", type=float, default=2.0e-3)
    parser.add_argument("--activity", type=float, default=0.25)
    parser.add_argument("--elastic-constant", type=float, default=2.0e-2)
    parser.add_argument("--rotational-diffusivity", type=float, default=0.4)
    parser.add_argument("--bulk-strength", type=float, default=1.0)
    parser.add_argument("--preferred-order", type=float, default=0.6)
    parser.add_argument("--flow-alignment", type=float, default=0.7)
    parser.add_argument("--max-order", type=float, default=1.5)
    parser.add_argument("--correction-interval", type=int, default=5)
    parser.add_argument("--no-plots", action="store_true")
    return parser


def main() -> None:
    compare(build_parser().parse_args())


if __name__ == "__main__":
    main()
