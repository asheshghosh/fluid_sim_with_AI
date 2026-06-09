from __future__ import annotations

import argparse
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .data import generate_velocity_trajectories, load_dataset, make_transition_pairs, save_dataset
from .incompressible import VelocitySolverConfig
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


def _solver_config_from_dict(data: dict, base: VelocitySolverConfig) -> VelocitySolverConfig:
    allowed = set(base.to_dict().keys())
    return VelocitySolverConfig(**{**base.to_dict(), **{key: value for key, value in data.items() if key in allowed}})


def train(args: argparse.Namespace) -> None:
    if args.target_stride <= 0:
        raise ValueError("--target-stride must be positive")
    if args.steps < args.target_stride:
        raise ValueError("--steps must be at least --target-stride")

    config = VelocitySolverConfig(
        n=args.n,
        viscosity=args.viscosity,
        dt=args.dt,
        forcing_amplitude=args.forcing_amplitude,
        forcing_wavenumber=args.forcing_wavenumber,
    )

    if args.dataset:
        trajectories, loaded_config = load_dataset(args.dataset)
        if loaded_config:
            config = _solver_config_from_dict(loaded_config, config)
    else:
        trajectories = generate_velocity_trajectories(
            config,
            trajectories=args.trajectories,
            steps=args.steps,
            keep_every=args.target_stride,
            seed=args.seed,
            amplitude=args.amplitude,
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
            mse = torch.mean((pred - yb) ** 2)
            smooth_x = torch.mean((pred[:, :, :, 1:] - pred[:, :, :, :-1]) ** 2)
            smooth_y = torch.mean((pred[:, :, 1:, :] - pred[:, :, :-1, :]) ** 2)
            loss = mse + 1.0e-4 * (smooth_x + smooth_y)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        scheduler.step()

        if epoch == 1 or epoch == args.epochs or epoch % max(1, args.epochs // 5) == 0:
            print(f"epoch={epoch:03d} loss={np.mean(losses):.6f}")

    solver_config = {**config.to_dict(), "equation": "velocity_incompressible_2d_periodic"}
    save_checkpoint(args.checkpoint, model, mean, std, solver_config, surrogate_step_size=args.target_stride)
    elapsed = time.perf_counter() - start
    print(f"saved checkpoint: {args.checkpoint}")
    print(f"model type: {args.model}")
    print(f"surrogate predicts every {args.target_stride} solver step(s)")
    print(f"training time: {elapsed:.2f}s on {device.type}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a surrogate for periodic incompressible velocity rollouts.")
    parser.add_argument("--dataset", default=None, help="Optional existing .npz dataset.")
    parser.add_argument("--save-dataset", default=None, help="Optional path to save generated trajectories.")
    parser.add_argument("--checkpoint", default="runs/incompressible_surrogate.pt", help="Output checkpoint path.")
    parser.add_argument("--n", type=int, default=32)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument(
        "--target-stride",
        type=int,
        default=1,
        help="Train velocity_t -> velocity_{t+stride}; values above 1 trade accuracy for solver-equivalent speed.",
    )
    parser.add_argument("--trajectories", type=int, default=12)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--amplitude", type=float, default=1.0)
    parser.add_argument("--dt", type=float, default=1.0e-2)
    parser.add_argument("--viscosity", type=float, default=2.0e-3)
    parser.add_argument("--forcing-amplitude", type=float, default=0.0)
    parser.add_argument("--forcing-wavenumber", type=int, default=4)
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
