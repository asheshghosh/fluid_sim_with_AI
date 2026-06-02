from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from .diagnostics import diagnostics_to_table, trajectory_diagnostics, write_diagnostics_json
from .plotting import plot_run_summary, save_diagnostics_csv
from .render import save_frame_strip
from .solver import SolverConfig, SpectralNavierStokes2D, random_vorticity
from .surrogate import detensorize, load_checkpoint, tensorize


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _run_ai(model, omega0, steps, mean, std, device):
    current = tensorize(omega0[None, :, :], mean, std, device)
    frames = [omega0.copy()]
    model.eval()
    with torch.no_grad():
        for _ in range(steps):
            current = model(current)
            frames.append(detensorize(current, mean, std)[0].astype(np.float64))
    return np.stack(frames, axis=0)


def _run_hybrid(model, solver, omega0, steps, mean, std, device, correction_interval):
    if correction_interval <= 0:
        raise ValueError("correction_interval must be positive")
    omega = omega0.copy()
    frames = [omega.copy()]
    model.eval()
    with torch.no_grad():
        for step in range(1, steps + 1):
            if step % correction_interval == 0:
                omega = solver.step(omega)
            else:
                current = tensorize(omega[None, :, :], mean, std, device)
                omega = detensorize(model(current), mean, std)[0].astype(np.float64)
            omega -= np.mean(omega)
            frames.append(omega.copy())
    return np.stack(frames, axis=0)


def simulate(args: argparse.Namespace) -> None:
    device = _device()
    checkpoint_solver_config = {}
    model = None
    mean = 0.0
    std = 1.0

    if args.mode in {"ai", "hybrid"}:
        if not args.checkpoint:
            raise SystemExit("--checkpoint is required for ai and hybrid modes")
        model, mean, std, checkpoint_solver_config = load_checkpoint(args.checkpoint, device=device)

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

    start = time.perf_counter()
    if args.mode == "solver":
        trajectory = solver.rollout(omega0, steps=args.steps)
    elif args.mode == "ai":
        trajectory = _run_ai(model, omega0, args.steps, mean, std, device)
    elif args.mode == "hybrid":
        trajectory = _run_hybrid(
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
        raise ValueError(f"unknown mode: {args.mode}")
    elapsed = time.perf_counter() - start

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    diagnostics = trajectory_diagnostics(solver, trajectory, dt=config.dt)
    diagnostic_names, diagnostic_values = diagnostics_to_table(diagnostics)
    np.savez_compressed(
        out / "trajectory.npz",
        vorticity=trajectory,
        diagnostics=diagnostics,
        diagnostic_names=diagnostic_names,
        diagnostic_values=diagnostic_values,
    )
    write_diagnostics_json(out / "diagnostics.json", diagnostics)
    save_diagnostics_csv(diagnostics, out / "diagnostics.csv")

    metadata = {
        "mode": args.mode,
        "steps": args.steps,
        "seconds": elapsed,
        "steps_per_second": args.steps / elapsed if elapsed > 0.0 else None,
        "device": device.type,
        "solver_config": config.to_dict(),
        "checkpoint": args.checkpoint,
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    if not args.no_render:
        save_frame_strip(trajectory, out / "frames", max_frames=args.max_frames)
    if not args.no_plots:
        plot_paths = plot_run_summary(solver, trajectory, diagnostics, out / "plots", max_frames=args.plot_frames)
        print("plots:")
        for plot_path in plot_paths:
            print(f"  {plot_path}")

    print(f"mode={args.mode}")
    print(f"wrote: {out}")
    print(f"steps/sec={metadata['steps_per_second']:.2f}")
    print(f"final diagnostics={diagnostics[-1]}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the hybrid fluid simulator.")
    parser.add_argument("--mode", choices=["solver", "ai", "hybrid"], default="solver")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--use-checkpoint-config", action="store_true")
    parser.add_argument("--out", default="runs/demo")
    parser.add_argument("--n", type=int, default=64)
    parser.add_argument("--steps", type=int, default=200)
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
    simulate(build_parser().parse_args())


if __name__ == "__main__":
    main()
