# Periodic Chip Heat Equation With AI Acceleration

This branch starts a chip-thermal simulation track. The first model is deliberately
simple: a two-dimensional periodic heat equation with fixed heat-source maps that
represent local power dissipation. The goal is to build the same workflow as the
fluid solvers:

1. generate trusted trajectories with a spectral solver,
2. train an FNO or CNN surrogate on solver transitions,
3. compare exact, AI-only, and hybrid-corrected rollouts,
4. measure speed and physical diagnostics.

The model is not yet a full electronic-package thermal stack. It is the smallest
clean step toward that: a controlled diffusion problem with power maps, cooling,
hotspot diagnostics, and source-conditioned neural rollouts.

## State Variables

The physical state is the temperature rise above ambient,

$\theta(x,y,t) = T(x, y, t) - T_\text{ambient}$

For the neural surrogate the state has two channels:

```text
state = [theta, q]
```

where `q(x, y)` is a fixed heat-source or power-density map. Keeping `q` in the
state lets the FNO learn source-conditioned dynamics rather than memorizing one
layout. During AI rollout, the code pins the source channel back to the known
map after each prediction.

## Governing Equation

The solver evolves

```text
d theta / dt = kappa Laplacian(theta) - gamma theta + q(x, y).
```

The terms are:

- `kappa Laplacian(theta)`: lateral heat diffusion.
- `-gamma theta`: a simple linear cooling path to ambient.
- `q(x, y)`: local heat generation from chip activity.

The absolute temperature is recovered as

```text
T = T_ambient + theta.
```

## Periodic Spectral Discretization

On a square periodic domain of side length `L`, the temperature is expanded in
Fourier modes:

```text
theta(x, y, t) = sum_k theta_hat_k(t) exp(i k . x).
```

For each Fourier mode,

```text
Laplacian(theta)_hat_k = -|k|^2 theta_hat_k.
```

The solver uses an exact Fourier derivative and a semi-implicit update for the
linear diffusion/cooling part:

```text
theta_hat^{n+1}_k =
    (theta_hat^n_k + dt q_hat_k)
    / (1 + dt (kappa |k|^2 + gamma)).
```

This is stable for the diffusion term and makes the exact solver a reliable
teacher for surrogate training.

## Chip-Like Heat Sources

The current source generator creates a sum of smooth periodic Gaussian bumps:

```text
q(x, y) = sum_j a_j exp(-d_periodic((x, y), c_j)^2 / (2 sigma^2)).
```

This mimics localized power blocks or hotspots. The generator can vary the
source seed per trajectory, so the surrogate can see many layouts.

Future chip realism can add:

- non-periodic package boundaries,
- layer-dependent thermal conductivity,
- anisotropic materials,
- transient workload traces,
- temperature-dependent conductivity,
- advection from liquid cooling channels,
- calibrated power maps from chip floorplans.

## Diagnostics

The compare runner writes per-frame diagnostics for each mode:

- mean, max, min, and standard deviation of temperature rise,
- max absolute temperature rise,
- absolute mean and max temperature,
- thermal variance,
- RMS temperature gradient,
- RMS heat flux,
- hotspot area fraction,
- source/cooling/net power density,
- correlation between heat source and temperature.

These are useful because a surrogate can look visually close but still miss
hotspot amplitude, thermal spreading rate, or the source-temperature alignment.

## AI Surrogate

The FNO surrogate learns a strided transition:

```text
[theta_t, q] -> [theta_{t + s}, q]
```

where `s` is `--target-stride`. A larger stride means one neural inference
represents more exact solver steps, which is how the AI path can become faster
in solver-equivalent steps/sec.

The FNO is a natural first model because heat diffusion is nonlocal in time and
diagonal in Fourier space for this periodic problem. Low-frequency Fourier modes
carry much of the thermal field evolution, especially after diffusion smooths
initial transients.

## Hybrid Correction

AI-only rollouts can drift. Hybrid mode alternates cheap neural steps with exact
solver corrections:

```text
for each surrogate step:
    if correction step:
        run exact heat solver for s small steps
    else:
        run FNO for one strided prediction
```

This keeps the experiment honest: the AI path is tested for speed, while the
hybrid path measures how much exact physics is needed to control error.

## Commands

Train a small FNO heat surrogate:

```bash
python -m fluid_ai_sim.train_heat_surrogate \
  --model fno \
  --n 32 \
  --trajectories 16 \
  --steps 64 \
  --target-stride 4 \
  --epochs 8 \
  --width 32 \
  --depth 4 \
  --modes 12 \
  --checkpoint runs/heat_fno.pt
```

Compare exact, AI, and hybrid rollouts:

```bash
python -m fluid_ai_sim.compare_heat_modes \
  --checkpoint runs/heat_fno.pt \
  --use-checkpoint-config \
  --steps 120 \
  --correction-interval 5 \
  --out runs/heat_fno_compare
```

The comparison writes:

- `solver/`, `ai/`, and `hybrid/` run folders,
- `comparison_metrics.json` and `.npz`,
- `plots/temperature_error_to_solver.png`,
- `plots/thermal_diagnostics.png`,
- `plots/final_temperature_source.png`,
- `plots/speed_comparison.png`.

## Why This Matters For Chips

Modern chips are power-density limited. Better thermal prediction helps with:

- floorplanning,
- hotspot mitigation,
- workload scheduling,
- package/cooling design,
- digital twins for runtime thermal management.

The long-term target is not just “make heat diffusion faster.” It is to learn
surrogates that preserve physically important quantities while mapping design
choices and workloads to thermal risk quickly enough for optimization loops.
