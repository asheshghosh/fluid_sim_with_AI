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
    model_type: str = "cnn"
    width: int = 48
    depth: int = 5
    kernel_size: int = 5
    residual_scale: float = 0.25
    modes: int = 12

    def to_dict(self) -> Dict[str, object]:
        return {
            "model_type": self.model_type,
            "width": self.width,
            "depth": self.depth,
            "kernel_size": self.kernel_size,
            "residual_scale": self.residual_scale,
            "modes": self.modes,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> "SurrogateConfig":
        return cls(
            model_type=str(data.get("model_type", cls.model_type)),
            width=int(data.get("width", cls.width)),
            depth=int(data.get("depth", cls.depth)),
            kernel_size=int(data.get("kernel_size", cls.kernel_size)),
            residual_scale=float(data.get("residual_scale", cls.residual_scale)),
            modes=int(data.get("modes", cls.modes)),
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


class SpectralConv2d(nn.Module):
    """2D Fourier layer that learns low-frequency spectral mixing."""

    def __init__(self, in_channels: int, out_channels: int, modes: int):
        super().__init__()
        if modes <= 0:
            raise ValueError("modes must be positive")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        scale = 1.0 / float(in_channels * out_channels)
        self.weights_pos = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes, modes, dtype=torch.cfloat)
        )
        self.weights_neg = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes, modes, dtype=torch.cfloat)
        )

    @staticmethod
    def _compl_mul2d(inputs: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bixy,ioxy->boxy", inputs, weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = x.shape
        x_ft = torch.fft.rfft2(x)
        out_ft = torch.zeros(
            batch,
            self.out_channels,
            height,
            width // 2 + 1,
            dtype=x_ft.dtype,
            device=x.device,
        )

        modes_x = min(self.modes, max(1, height // 2))
        modes_y = min(self.modes, width // 2 + 1)
        out_ft[:, :, :modes_x, :modes_y] = self._compl_mul2d(
            x_ft[:, :, :modes_x, :modes_y],
            self.weights_pos[:, :, :modes_x, :modes_y],
        )
        out_ft[:, :, -modes_x:, :modes_y] = self._compl_mul2d(
            x_ft[:, :, -modes_x:, :modes_y],
            self.weights_neg[:, :, :modes_x, :modes_y],
        )
        return torch.fft.irfft2(out_ft, s=(height, width))


class FNOBlock(nn.Module):
    def __init__(self, channels: int, modes: int):
        super().__init__()
        self.spectral = SpectralConv2d(channels, channels, modes)
        self.local = nn.Conv2d(channels, channels, kernel_size=1)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.spectral(x) + self.local(x))


class FourierFluidSurrogate(nn.Module):
    """Small Fourier Neural Operator for periodic vorticity rollouts."""

    def __init__(self, config: SurrogateConfig = SurrogateConfig(model_type="fno")):
        super().__init__()
        if config.depth <= 0:
            raise ValueError("depth must be positive")
        self.config = config
        self.lift = nn.Conv2d(1, config.width, kernel_size=1)
        self.blocks = nn.Sequential(*[FNOBlock(config.width, config.modes) for _ in range(config.depth)])
        self.project = nn.Sequential(
            nn.Conv2d(config.width, config.width, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(config.width, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.blocks(self.lift(x))
        delta = self.project(features)
        return x + self.config.residual_scale * delta


def build_surrogate(config: SurrogateConfig) -> nn.Module:
    if config.model_type == "cnn":
        return FastFluidSurrogate(config)
    if config.model_type == "fno":
        return FourierFluidSurrogate(config)
    raise ValueError(f"unknown surrogate model_type: {config.model_type}")


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
    model: nn.Module,
    mean: float,
    std: float,
    solver_config: Dict[str, float],
    surrogate_step_size: int = 1,
) -> None:
    if surrogate_step_size <= 0:
        raise ValueError("surrogate_step_size must be positive")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "surrogate_config": model.config.to_dict(),
            "mean": mean,
            "std": std,
            "solver_config": solver_config,
            "surrogate_step_size": int(surrogate_step_size),
        },
        path,
    )


def load_checkpoint(path: str | Path, device: torch.device | None = None) -> Tuple[nn.Module, float, float, dict]:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(Path(path), map_location=device)
    model = build_surrogate(SurrogateConfig.from_dict(checkpoint["surrogate_config"]))
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model, float(checkpoint["mean"]), float(checkpoint["std"]), dict(checkpoint.get("solver_config", {}))


def load_checkpoint_metadata(path: str | Path) -> dict:
    checkpoint = torch.load(Path(path), map_location=torch.device("cpu"))
    return {
        "solver_config": dict(checkpoint.get("solver_config", {})),
        "surrogate_config": dict(checkpoint.get("surrogate_config", {})),
        "surrogate_step_size": int(checkpoint.get("surrogate_step_size", 1)),
        "mean": float(checkpoint["mean"]),
        "std": float(checkpoint["std"]),
    }


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
