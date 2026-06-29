from __future__ import annotations

import argparse
import json
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


MODES = ["solver", "ai", "hybrid"]
COLORS = {
    "solver": "#2563eb",
    "ai": "#dc2626",
    "hybrid": "#0f766e",
}
LABELS = {
    "solver": "Exact heat solver",
    "ai": "Stride-8 FNO AI",
    "hybrid": "Stride-8 hybrid",
}


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def make_plot(run_dir: Path, out: Path) -> None:
    metadata = {mode: _load_json(run_dir / mode / "metadata.json") for mode in MODES}
    comparison = _load_json(run_dir / "comparison_metrics.json")

    speeds = [float(metadata[mode]["steps_per_second"]) for mode in MODES]
    solver_speed = speeds[0]
    speedups = [speed / solver_speed if solver_speed > 0.0 else 0.0 for speed in speeds]
    errors = [
        float("nan"),
        float(comparison["ai"]["final_relative_l2"]),
        float(comparison["hybrid"]["final_relative_l2"]),
    ]

    grid_size = int(metadata["solver"]["solver_config"]["n"])
    steps = int(metadata["solver"]["steps"])
    stride = int(metadata["ai"].get("surrogate_step_size", 1))
    checkpoint = str(metadata["solver"].get("checkpoint", "unknown"))

    fig, (speed_axis, error_axis) = plt.subplots(
        2,
        1,
        figsize=(9.2, 7.0),
        gridspec_kw={"height_ratios": [1.35, 1.0], "hspace": 0.42},
    )
    fig.patch.set_facecolor("white")

    x = list(range(len(MODES)))
    bars = speed_axis.bar(x, speeds, color=[COLORS[mode] for mode in MODES], width=0.58)
    for bar, speed, speedup in zip(bars, speeds, speedups):
        speed_axis.annotate(
            f"{speed:,.0f}\n{speedup:.2f}x",
            xy=(bar.get_x() + bar.get_width() / 2.0, bar.get_height()),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
            color="#111827",
        )
    speed_axis.axhline(solver_speed, color="#6b7280", linestyle="--", linewidth=1.0, alpha=0.75)
    speed_axis.set_title("N=128 Chip Heat FNO Rollout Benchmark", fontsize=15, pad=14)
    speed_axis.set_ylabel("Solver-equivalent steps/sec")
    speed_axis.set_xticks(x)
    speed_axis.set_xticklabels([LABELS[mode] for mode in MODES])
    speed_axis.set_ylim(0.0, max(speeds) * 1.28)
    speed_axis.grid(True, axis="y", alpha=0.25)
    speed_axis.spines["top"].set_visible(False)
    speed_axis.spines["right"].set_visible(False)

    error_modes = ["ai", "hybrid"]
    error_x = list(range(len(error_modes)))
    error_values = [errors[MODES.index(mode)] for mode in error_modes]
    error_bars = error_axis.bar(
        error_x,
        error_values,
        color=[COLORS[mode] for mode in error_modes],
        width=0.58,
    )
    for bar, error in zip(error_bars, error_values):
        error_axis.annotate(
            f"{error:.3f}",
            xy=(bar.get_x() + bar.get_width() / 2.0, bar.get_height()),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
            color="#111827",
        )
    error_axis.set_ylabel("Final relative L2 error")
    error_axis.set_xticks(error_x)
    error_axis.set_xticklabels([LABELS[mode] for mode in error_modes])
    error_axis.set_ylim(0.0, max(error_values) * 1.35)
    error_axis.grid(True, axis="y", alpha=0.25)
    error_axis.spines["top"].set_visible(False)
    error_axis.spines["right"].set_visible(False)

    note = (
        f"n={grid_size}, {steps} solver-equivalent steps, stride={stride}; "
        f"checkpoint={checkpoint}. Source-conditioned state channels are [temperature, heat_source]."
    )
    fig.text(0.01, 0.012, note, fontsize=8.5, color="#4b5563")

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, format="svg", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate an SVG summary for the heat FNO benchmark.")
    parser.add_argument("--run-dir", default="runs/heat_fno_stride8_n128_compare_ci8")
    parser.add_argument("--out", default="docs/heat_fno_stride8_n128.svg")
    args = parser.parse_args()
    make_plot(Path(args.run_dir), Path(args.out))


if __name__ == "__main__":
    main()
