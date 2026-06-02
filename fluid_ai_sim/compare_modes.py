from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict

import numpy as np
import torch

from .diagnostics import (
    comparison_errors,
    diagnostics_to_table,
    trajectory_diagnostics,
    write_diagnostics_json,
)
from .plotting import plot_comparison_summary, plot_run_summary, save_diagnostics_csv
from .render import save_frame_strip
from .simulate import _device, _run_ai, _run_hybrid
from .solver import SolverConfig, SpectralNavierStokes2D, random_vorticity
from .surrogate import detensorize, load_checkpoint, load_checkpoint_metadata, tensorize


def _run_hybrid_strided(
    model,
    solver: SpectralNavierStokes2D,
    omega0: np.ndarray,
    model_steps: int,
    surrogate_step_size: int,
    mean: float,
    std: float,
    device,
    correction_interval: int,
) -> np.ndarray:
    if correction_interval <= 0:
        raise ValueError("correction_interval must be positive")
    if surrogate_step_size <= 0:
        raise ValueError("surrogate_step_size must be positive")

    omega = omega0.copy()
    frames = [omega.copy()]
    model.eval()
    with torch.no_grad():
        for step in range(1, model_steps + 1):
            if step % correction_interval == 0:
                omega = solver.rollout(omega, steps=surrogate_step_size)[-1]
            else:
                current = tensorize(omega[None, :, :], mean, std, device)
                omega = detensorize(model(current), mean, std)[0].astype(np.float64)
            omega -= np.mean(omega)
            frames.append(omega.copy())
    return np.stack(frames, axis=0)


def _save_run(
    out: Path,
    label: str,
    trajectory: np.ndarray,
    solver: SpectralNavierStokes2D,
    seconds: float,
    solver_equivalent_steps: int,
    keep_every: int,
    args: argparse.Namespace,
) -> tuple[list[dict], dict]:
    run_dir = out / label
    run_dir.mkdir(parents=True, exist_ok=True)

    diagnostics = trajectory_diagnostics(solver, trajectory, dt=solver.config.dt, keep_every=keep_every)
    diagnostic_names, diagnostic_values = diagnostics_to_table(diagnostics)
    np.savez_compressed(
        run_dir / "trajectory.npz",
        vorticity=trajectory,
        diagnostics=diagnostics,
        diagnostic_names=diagnostic_names,
        diagnostic_values=diagnostic_values,
    )
    write_diagnostics_json(run_dir / "diagnostics.json", diagnostics)
    save_diagnostics_csv(diagnostics, run_dir / "diagnostics.csv")

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

    if not args.no_render:
        save_frame_strip(trajectory, run_dir / "frames", max_frames=args.max_frames)
    if not args.no_plots:
        plot_run_summary(solver, trajectory, diagnostics, run_dir / "plots", max_frames=args.plot_frames)

    return diagnostics, metadata


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

    base_config = {
        "n": args.n,
        "viscosity": args.viscosity,
        "dt": args.dt,
        "forcing_amplitude": args.forcing_amplitude,
        "forcing_wavenumber": args.forcing_wavenumber,
    }
    if args.use_checkpoint_config:
        base_config.update(checkpoint_solver_config)
    config = SolverConfig(**base_config)
    solver = SpectralNavierStokes2D(config)

    omega0 = random_vorticity(
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
    trajectories["solver"] = solver.rollout(omega0, steps=args.steps, keep_every=surrogate_step_size)
    timings["solver"] = time.perf_counter() - start

    start = time.perf_counter()
    trajectories["ai"] = _run_ai(model, omega0, model_steps, mean, std, device)
    timings["ai"] = time.perf_counter() - start

    start = time.perf_counter()
    if surrogate_step_size == 1:
        trajectories["hybrid"] = _run_hybrid(
            model,
            solver,
            omega0,
            args.steps,
            mean,
            std,
            device,
            correction_interval=args.correction_interval,
        )
    else:
        trajectories["hybrid"] = _run_hybrid_strided(
            model,
            solver,
            omega0,
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
        plot_paths = plot_comparison_summary(
            trajectories["solver"],
            trajectories,
            diagnostics_by_label,
            metadata_by_label,
            out / "plots",
            dt=config.dt * surrogate_step_size,
            correction_interval=args.correction_interval,
        )
        print("comparison plots:")
        for plot_path in plot_paths:
            print(f"  {plot_path}")

    (out / "metadata.json").write_text(
        json.dumps(
            {
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
    print(f"wrote comparison: {out}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare solver, AI, and hybrid fluid rollouts.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--use-checkpoint-config", action="store_true")
    parser.add_argument("--out", default="runs/comparison")
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
    parser.add_argument("--max-frames", type=int, default=24)
    parser.add_argument("--plot-frames", type=int, default=6)
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    return parser


def main() -> None:
    compare(build_parser().parse_args())


if __name__ == "__main__":
    main()
