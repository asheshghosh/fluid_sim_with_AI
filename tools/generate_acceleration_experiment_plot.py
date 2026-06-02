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
    "solver": "Exact solver",
    "ai": "Stride-8 AI",
    "hybrid": "Stride-8 hybrid",
}


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def make_plot(run_dir: Path, out: Path) -> None:
    metadata = {mode: _load_json(run_dir / mode / "metadata.json") for mode in MODES}
    comparison_path = run_dir / "comparison_metrics.json"
    comparison = _load_json(comparison_path) if comparison_path.exists() else {}

    speeds = [float(metadata[mode]["steps_per_second"]) for mode in MODES]
    baseline = speeds[0]
    speedups = [speed / baseline if baseline > 0.0 else 0.0 for speed in speeds]
    step_size = int(metadata["ai"].get("surrogate_step_size", 1))
    steps = int(metadata["solver"]["steps"])

    fig, axis = plt.subplots(figsize=(8.8, 4.8))
    fig.patch.set_facecolor("white")
    axis.set_facecolor("white")

    x = range(len(MODES))
    bars = axis.bar(
        x,
        speeds,
        color=[COLORS[mode] for mode in MODES],
        width=0.58,
    )

    for bar, speedup in zip(bars, speedups):
        height = bar.get_height()
        axis.annotate(
            f"{height:,.0f}\n{speedup:.2f}x",
            xy=(bar.get_x() + bar.get_width() / 2.0, height),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
            color="#111827",
        )

    axis.axhline(baseline, color="#6b7280", linestyle="--", linewidth=1.0, alpha=0.7)
    axis.text(
        2.42,
        baseline,
        "solver baseline",
        va="center",
        ha="left",
        fontsize=9,
        color="#4b5563",
    )

    axis.set_title("AI Acceleration Experiment: Strided Surrogate Rollout", fontsize=15, pad=14)
    axis.set_ylabel("Solver-equivalent steps per second")
    axis.set_xticks(list(x))
    axis.set_xticklabels([LABELS[mode] for mode in MODES])
    axis.set_ylim(0.0, max(speeds) * 1.28)
    axis.grid(True, axis="y", alpha=0.25)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)

    error_bits = []
    for mode in ["ai", "hybrid"]:
        if mode in comparison and "final_relative_l2" in comparison[mode]:
            error_bits.append(f"{LABELS[mode]} final relative L2: {comparison[mode]['final_relative_l2']:.2f}")
    error_note = "; ".join(error_bits) if error_bits else "accuracy metrics unavailable"
    note = (
        f"n=64, {steps} solver-equivalent steps, stride={step_size}; "
        f"checkpoint=runs/accelerated_stride8.pt. {error_note}."
    )
    fig.text(0.01, 0.01, note, fontsize=8.5, color="#4b5563")

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, format="svg", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the strided AI acceleration SVG.")
    parser.add_argument("--run-dir", default="runs/accelerated_stride8_compare_ci16")
    parser.add_argument("--out", default="docs/ai_acceleration_stride8.svg")
    args = parser.parse_args()
    make_plot(Path(args.run_dir), Path(args.out))


if __name__ == "__main__":
    main()
