from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

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
import numpy as np


BENCHMARKS = [
    {
        "grid": "64x64",
        "solver": 3242.46,
        "ai": 1553.06,
        "hybrid": 1811.33,
    },
    {
        "grid": "128x128",
        "solver": 994.11,
        "ai": 461.83,
        "hybrid": 486.13,
    },
    {
        "grid": "256x256",
        "solver": 186.68,
        "ai": 98.39,
        "hybrid": 86.45,
    },
    {
        "grid": "512x512",
        "solver": 35.91,
        "ai": 24.53,
        "hybrid": 25.73,
    },
]


def make_plot(out: Path) -> None:
    labels = [item["grid"] for item in BENCHMARKS]
    solver = [item["solver"] for item in BENCHMARKS]
    ai = [item["ai"] for item in BENCHMARKS]
    hybrid = [item["hybrid"] for item in BENCHMARKS]

    x = np.arange(len(labels), dtype=np.float64)
    width = 0.24

    fig, axis = plt.subplots(figsize=(10.2, 5.2))
    fig.patch.set_facecolor("white")
    axis.set_facecolor("white")

    bars = [
        axis.bar(x - width, solver, width, label="Naive spectral solver", color="#2563eb"),
        axis.bar(x, ai, width, label="AI surrogate", color="#dc2626"),
        axis.bar(x + width, hybrid, width, label="Hybrid AI + solver", color="#0f766e"),
    ]

    for group in bars:
        for bar in group:
            height = bar.get_height()
            axis.annotate(
                f"{height:,.1f}" if height < 100.0 else f"{height:,.0f}",
                xy=(bar.get_x() + bar.get_width() / 2.0, height),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
                color="#111827",
            )

    axis.set_title("Simulation Throughput: Exact Solver vs AI Rollout", fontsize=15, pad=14)
    axis.set_ylabel("Steps per second, log scale (higher is better)")
    axis.set_xlabel("Grid resolution")
    axis.set_xticks(x)
    axis.set_xticklabels(labels)
    axis.set_yscale("log")
    axis.set_ylim(10.0, max(solver) * 1.7)
    axis.grid(True, axis="y", which="both", alpha=0.25)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.legend(loc="upper right", frameon=False)

    note = "Measured locally with runs/smoke_surrogate.pt, 100 rollout steps, rendering disabled."
    fig.text(0.01, 0.01, note, fontsize=9, color="#4b5563")

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, format="svg", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the README speed comparison SVG.")
    parser.add_argument("--out", default="docs/ai_solver_speed.svg")
    args = parser.parse_args()
    make_plot(Path(args.out))


if __name__ == "__main__":
    main()
