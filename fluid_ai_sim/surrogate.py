from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class SurrogateConfig:
    width: int = 48
    depth: int = 5
    kernel_size: int = 5
    residual_scale: float = 0.25

    def to_dict(self) -> Dict[str, float]:
        return {
            "width": self.width,
            "depth": self.depth,
            "kernel_size": self.kernel_size,
            "residual_scale": self.residual_scale,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> "SurrogateConfig":
        return cls(
            width=int(data.get("width", cls.width)),
            depth=int(data.get("depth", cls.depth)),
            kernel_size=int(data.get("kernel_size", cls.kernel_size)),
            residual_scale=float(data.get("residual_scale", cls.residual_scale)),
        )


class PeriodicConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__()
        if kernel_size % 2 != 1:
            raise ValueError("kernel_size must be odd")
        self.pad = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (self.pad, self.pad, self.pad, self.pad), mode="circular")
        return self.conv(x)


class ConvBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int):
        super().__init__()
        self.net = nn.Sequential(
            PeriodicConv2d(channels, channels, kernel_size),
            nn.GELU(),
            PeriodicConv2d(channels, channels, kernel_size),
        )
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.net(x))


class FastFluidSurrogate(nn.Module):
    """Small residual CNN for fast vorticity rollouts."""

    def __init__(self, config: SurrogateConfig = SurrogateConfig()):
        super().__init__()
        self.config = config
        self.lift = nn.Sequential(
            PeriodicConv2d(1, config.width, config.kernel_size),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(
            *[ConvBlock(config.width, config.kernel_size) for _ in range(config.depth)]
        )
        self.project = PeriodicConv2d(config.width, 1, config.kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.blocks(self.lift(x))
        delta = self.project(features)
        return x + self.config.residual_scale * delta


def tensorize(fields: np.ndarray, mean: float, std: float, device: torch.device) -> torch.Tensor:
    normalized = (fields.astype(np.float32) - mean) / std
    return torch.from_numpy(normalized[:, None, :, :]).to(device)


def detensorize(batch: torch.Tensor, mean: float, std: float) -> np.ndarray:
    fields = batch.detach().cpu().numpy()[:, 0]
    return fields * std + mean


def normalization_stats(fields: np.ndarray) -> Tuple[float, float]:
    mean = float(np.mean(fields))
    std = float(np.std(fields))
    if std <= 1.0e-8:
        std = 1.0
    return mean, std


def save_checkpoint(
    path: str | Path,
    model: FastFluidSurrogate,
    mean: float,
    std: float,
    solver_config: Dict[str, float],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "surrogate_config": model.config.to_dict(),
            "mean": mean,
            "std": std,
            "solver_config": solver_config,
        },
        path,
    )


def load_checkpoint(path: str | Path, device: torch.device | None = None) -> Tuple[FastFluidSurrogate, float, float, dict]:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(Path(path), map_location=device)
    model = FastFluidSurrogate(SurrogateConfig.from_dict(checkpoint["surrogate_config"]))
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model, float(checkpoint["mean"]), float(checkpoint["std"]), dict(checkpoint.get("solver_config", {}))


@torch.no_grad()
def rollout_surrogate(
    model: FastFluidSurrogate,
    omega0: np.ndarray,
    steps: int,
    mean: float,
    std: float,
    device: torch.device | None = None,
) -> np.ndarray:
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    current = tensorize(omega0[None, :, :], mean, std, device)
    frames = [omega0.astype(np.float64)]
    for _ in range(steps):
        current = model(current)
        frames.append(detensorize(current, mean, std)[0].astype(np.float64))
    return np.stack(frames, axis=0)
