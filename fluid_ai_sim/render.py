from __future__ import annotations

from pathlib import Path

import numpy as np


def _colorize(field: np.ndarray) -> np.ndarray:
    field = np.asarray(field, dtype=np.float64)
    centered = field - np.median(field)
    scale = np.percentile(np.abs(centered), 98.0)
    if scale <= 1.0e-12:
        scale = 1.0
    t = np.clip(0.5 + 0.5 * centered / scale, 0.0, 1.0)

    # Compact blue-white-red diverging palette.
    r = np.clip(2.0 * t, 0.0, 1.0)
    g = np.clip(2.0 - 2.0 * np.abs(t - 0.5), 0.0, 1.0)
    b = np.clip(2.0 * (1.0 - t), 0.0, 1.0)
    return (255.0 * np.stack([r, g, b], axis=-1)).astype(np.uint8)


def save_ppm(field: np.ndarray, path: str | Path, scale: int = 4) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image = _colorize(field)
    if scale > 1:
        image = np.repeat(np.repeat(image, scale, axis=0), scale, axis=1)
    height, width, _ = image.shape
    header = f"P6\n{width} {height}\n255\n".encode("ascii")
    path.write_bytes(header + image.tobytes())


def save_frame_strip(trajectory: np.ndarray, out_dir: str | Path, max_frames: int = 24) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    total = trajectory.shape[0]
    indices = np.linspace(0, total - 1, num=min(max_frames, total), dtype=int)
    for frame_index, trajectory_index in enumerate(indices):
        save_ppm(trajectory[trajectory_index], out_dir / f"frame_{frame_index:03d}.ppm")
