# HeatLab AI Web App

Run from the repository root:

```bash
python3 -m http.server 8765 --bind 127.0.0.1 --directory web/heat-lab
```

Then open:

```text
http://127.0.0.1:8765
```

The app is a local browser workbench for heat-equation simulations over
well-defined chip-like geometries. It includes:

- exact finite-difference heat-equation updates,
- an AI-style macro surrogate that advances several solver-equivalent steps per call,
- hybrid correction that periodically swaps in exact physics,
- geometry presets for chiplets, hotspots, microchannels, vias, rings, and L-shaped accelerators,
- live temperature, source, cooling, and error fields,
- thermal diagnostics, CSV export, and PNG snapshots.

The in-browser surrogate is intentionally transparent: it is a neural-operator
style macro predictor, not a loaded PyTorch checkpoint. Use it to explore the
speed/accuracy contract, then use the Python FNO training path for checkpointed
experiments.
