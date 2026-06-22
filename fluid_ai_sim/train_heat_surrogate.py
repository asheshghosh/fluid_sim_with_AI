from __future__ import annotations

import argparse
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .data import generate_heat_trajectories, load_dataset, make_transition_pairs, save_dataset
from .heat import HeatSolverConfig
from .surrogate import (
    SurrogateConfig,
    build_surrogate,
    normalization_stats,
    save_checkpoint,
    tensorize,
)


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _solver_config_from_dict(data: dict, base: HeatSolverConfig) -> HeatSolverConfig:
    allowed = set(base.to_dict().keys())
    return HeatSolverConfig(**{**base.to_dict(), **{key: value for key, value in data.items() if key in allowed}})


def train(args: argparse.Namespace) -> None:
    if args.target_stride <= 0:
        raise ValueError("--target-stride must be positive")
    if args.steps < args.target_stride:
        raise ValueError("--steps must be at least --target-stride")

    config = HeatSolverConfig(
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

    if args.dataset:
        trajectories, loaded_config = load_dataset(args.dataset)
        if loaded_config:
            config = _solver_config_from_dict(loaded_config, config)
    else:
        trajectories = generate_heat_trajectories(
            config,
            trajectories=args.trajectories,
            steps=args.steps,
            keep_every=args.target_stride,
            seed=args.seed,
            amplitude=args.amplitude,
            bias=args.bias,
            vary_sources=not args.fixed_source,
        )
        if args.save_dataset:
            save_dataset(args.save_dataset, trajectories, config)

    x_np, y_np = make_transition_pairs(trajectories)
    mean, std = normalization_stats(np.concatenate([x_np, y_np], axis=0))

    device = _device()
    x = tensorize(x_np, mean, std, device=torch.device("cpu"))
    y = tensorize(y_np, mean, std, device=torch.device("cpu"))
    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)

    model = build_surrogate(
        SurrogateConfig(
            model_type=args.model,
            channels=2,
            width=args.width,
            depth=args.depth,
            kernel_size=args.kernel_size,
            residual_scale=args.residual_scale,
            modes=args.modes,
        )
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))

    start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            temperature_loss = torch.mean((pred[:, :1] - yb[:, :1]) ** 2)
            source_loss = torch.mean((pred[:, 1:] - yb[:, 1:]) ** 2)
            smooth_x = torch.mean((pred[:, :1, :, 1:] - pred[:, :1, :, :-1]) ** 2)
            smooth_y = torch.mean((pred[:, :1, 1:, :] - pred[:, :1, :-1, :]) ** 2)
            loss = temperature_loss + 0.05 * source_loss + 1.0e-5 * (smooth_x + smooth_y)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        scheduler.step()

        if epoch == 1 or epoch == args.epochs or epoch % max(1, args.epochs // 5) == 0:
            print(f"epoch={epoch:03d} loss={np.mean(losses):.6f}")

    solver_config = {**config.to_dict(), "equation": "heat_2d_periodic_source_conditioned"}
    save_checkpoint(args.checkpoint, model, mean, std, solver_config, surrogate_step_size=args.target_stride)
    elapsed = time.perf_counter() - start
    print(f"saved checkpoint: {args.checkpoint}")
    print(f"model type: {args.model}")
    print(f"surrogate predicts every {args.target_stride} solver step(s)")
    print(f"training time: {elapsed:.2f}s on {device.type}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a source-conditioned surrogate for periodic chip heat diffusion.")
    parser.add_argument("--dataset", default=None, help="Optional existing .npz dataset.")
    parser.add_argument("--save-dataset", default=None, help="Optional path to save generated trajectories.")
    parser.add_argument("--checkpoint", default="runs/heat_surrogate.pt", help="Output checkpoint path.")
    parser.add_argument("--n", type=int, default=32)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument(
        "--target-stride",
        type=int,
        default=4,
        help="Train state_t -> state_{t+stride}; values above 1 trade accuracy for solver-equivalent speed.",
    )
    parser.add_argument("--trajectories", type=int, default=16)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--amplitude", type=float, default=0.15, help="Initial temperature perturbation scale.")
    parser.add_argument("--bias", type=float, default=0.0, help="Uniform initial temperature rise.")
    parser.add_argument("--dt", type=float, default=2.0e-2)
    parser.add_argument("--diffusivity", type=float, default=5.0e-3)
    parser.add_argument("--sink-rate", type=float, default=2.0e-3)
    parser.add_argument("--source-amplitude", type=float, default=1.0)
    parser.add_argument("--source-width", type=float, default=0.22)
    parser.add_argument("--source-count", type=int, default=4)
    parser.add_argument("--source-seed", type=int, default=0)
    parser.add_argument("--fixed-source", action="store_true", help="Use one heat-source layout for every trajectory.")
    parser.add_argument("--ambient-temperature", type=float, default=300.0)
    parser.add_argument("--model", choices=["cnn", "fno"], default="fno")
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--kernel-size", type=int, default=5)
    parser.add_argument("--modes", type=int, default=12, help="Fourier modes per spatial axis for --model fno.")
    parser.add_argument("--residual-scale", type=float, default=0.25)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    return parser


def main() -> None:
    train(build_parser().parse_args())


if __name__ == "__main__":
    main()
