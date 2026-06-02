# Fluid AI Sim

This is a starter project for an AI-enabled fluid dynamics simulator. It combines:

- A real 2D incompressible Navier-Stokes solver in vorticity-streamfunction form.
- A Torch CNN surrogate that learns fast one-step rollouts from solver-generated trajectories.
- CLI tools for exact simulation, dataset generation, training, AI rollout, and hybrid rollout.

The first target is periodic 2D flow:

$\frac{d\omega}{dt} + u \frac{d\omega}{dx} + v \frac{d\omega}{dy} = \nu \nabla^2 \omega + \text{forcing}$

$u = \frac{d \psi}{dy}$

$v = -\frac{d \psi}{dx}$ 

$\nabla^2 \psi = -\omega$


This formulation avoids a separate pressure solve and keeps the velocity field divergence-free.

## Quick Start

Run a classical simulation:

```bash
python3 -m fluid_ai_sim.simulate --mode solver --n 64 --steps 200 --out runs/solver_demo
```

Train a tiny surrogate on generated solver data:

```bash
python3 -m fluid_ai_sim.train_surrogate --n 32 --trajectories 12 --steps 40 --epochs 8 --checkpoint runs/surrogate.pt
```

Run the AI surrogate:

```bash
python3 -m fluid_ai_sim.simulate --mode ai --checkpoint runs/surrogate.pt --n 32 --steps 120 --out runs/ai_demo
```

Run a hybrid simulation where the exact solver periodically corrects the AI rollout:

```bash
python3 -m fluid_ai_sim.simulate --mode hybrid --checkpoint runs/surrogate.pt --n 32 --steps 120 --correction-interval 10 --out runs/hybrid_demo
```

Each simulation writes:

- `trajectory.npz`: vorticity fields and diagnostics.
- `diagnostics.json`: human-readable per-step diagnostic values.
- `diagnostics.csv`: tabular diagnostics for quick plotting or spreadsheet import.
- `metadata.json`: run settings and timing.
- `frames/*.ppm`: rendered vorticity snapshots viewable by most image tools.
- `plots/*.png`: vorticity snapshots, energy/enstrophy curves, vorticity range,
  divergence check, and energy spectrum.

Compare exact solver, AI, and hybrid rollouts from the same initial condition:

```bash
python3 -m fluid_ai_sim.compare_modes --checkpoint runs/surrogate.pt --n 32 --steps 120 --out runs/comparison
```

The comparison writes separate `solver`, `ai`, and `hybrid` run folders plus:

- `comparison_metrics.npz` and `comparison_metrics.json`: AI/hybrid error against
  the exact solver.
- `plots/error_to_solver.png`: RMSE and relative L2 drift over time.
- `plots/mode_energy_enstrophy.png`: energy/enstrophy by mode.
- `plots/final_vorticity.png`: final vorticity side-by-side.
- `plots/speed_comparison.png`: solver, AI, and hybrid throughput.

## Architecture

The simulator is deliberately split into three layers:

1. **Truth solver**: `fluid_ai_sim.solver.SpectralNavierStokes2D`
   - FFT-based spectral derivatives.
   - Semi-implicit viscosity.
   - Optional Kolmogorov-style vorticity forcing.
   - Periodic domain.

2. **Learning system**: `fluid_ai_sim.surrogate.FastFluidSurrogate`
   - Small residual CNN with circular padding.
   - Learns normalized `omega_t -> omega_{t+1}` transitions.
   - Saves normalization stats in the checkpoint.

3. **Rollout modes**: `fluid_ai_sim.simulate`
   - `solver`: exact solver every step.
   - `ai`: learned model every step.
   - `hybrid`: learned model with periodic exact correction.

## Why This Is Useful

Classical CFD is reliable but expensive. Neural rollouts can be much faster once trained, but they drift. The hybrid mode is a practical middle ground: use AI for most steps and periodically re-anchor with the physics solver.

For production or research-grade work, the next upgrades would be:

- Fourier Neural Operator or U-Net surrogate.
- Boundary conditions beyond periodic domains.
- Pressure-velocity formulation for obstacles and walls.
- Physics-informed loss terms for energy, enstrophy, and residual consistency.
- GPU batching for many parameterized flow scenarios.
- Validation against known benchmarks such as Taylor-Green vortex, lid-driven cavity, or cylinder wake.

## Tests

```bash
python3 -m unittest discover -s tests
```
