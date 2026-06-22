# Incompressible 2D Navier-Stokes With AI Acceleration

This note explains the physics and numerical steps behind the velocity-based
2D periodic incompressible Navier-Stokes simulator in this repository, plus the
AI acceleration layer built on top of it.

The existing solver in `fluid_ai_sim.solver` uses the equivalent
vorticity-streamfunction formulation. The new solver in
`fluid_ai_sim.incompressible` evolves the velocity field directly:

```text
state q(x, y, t) = [u(x, y, t), v(x, y, t)]
```

The velocity form makes incompressibility explicit and gives a direct place to
apply the projection method used by many spectral incompressible solvers.

## 1. Continuous Equations

The incompressible Navier-Stokes equations in two spatial dimensions are

$\frac{du}{dt} + (u \cdot\nabla)u = -\nabla p + \nu \nabla^2 u + f $

$\nabla u = 0$

where:

- `u = (u, v)` is velocity.
- `p` is pressure divided by constant density.
- `nu` is kinematic viscosity.
- `f` is an optional body force.
- `div u = du/dx + dv/dy = 0` enforces incompressibility.

The domain is periodic:

```text
(x, y) in [0, L) x [0, L)
u(x + L, y, t) = u(x, y, t)
u(x, y + L, t) = u(x, y, t)
```

Periodic boundaries are a natural fit for FFT-based spectral methods because
the Fourier basis already satisfies the boundary condition.

## 2. Why Pressure Enforces Incompressibility

Pressure is not advanced as an independent thermodynamic state in this
incompressible model. Instead, pressure acts as a constraint force that removes
compressible motion.

Take the divergence of the momentum equation:

```text
div(du/dt) + div((u . grad)u) = -Laplacian p + nu div(Laplacian u) + div f
```

Because `div u = 0`, the terms `div(du/dt)` and `div(Laplacian u)` vanish.
That gives a Poisson equation for pressure:

```text
Laplacian p = div(f - (u . grad)u)
```

Solving this Poisson equation is one way to enforce incompressibility. In a
periodic spectral solver, an equivalent and simpler route is the Leray
projection.

## 3. Fourier Representation

On a periodic grid, represent velocity by Fourier modes:

```text
u(x, y) = sum_k u_hat(k) exp(i k . x)
```

For a Fourier mode with wavevector `k = (kx, ky)`, the divergence constraint is

```text
k . u_hat(k) = 0
```

So each Fourier velocity coefficient must lie perpendicular to its wavevector.

## 4. Leray Projection

For each nonzero Fourier mode, the incompressible projection is

```text
P_k a = a - k (k . a) / |k|^2
```

where `a` is a two-component Fourier vector. This removes the part of `a`
parallel to `k`, leaving only the divergence-free component.

The zero mode `k = 0` represents the spatial mean velocity. In this project the
solver sets that mode to zero to avoid bulk drift:

```text
u_hat(0, 0) = 0
```

The code path is:

```text
velocity -> FFT -> project every mode -> inverse FFT
```

This is implemented by:

```text
SpectralIncompressibleNavierStokes2D.project_hat
SpectralIncompressibleNavierStokes2D.project_velocity
```

## 5. Pseudo-Spectral Spatial Discretization

The solver stores velocity on an `n x n` grid. Derivatives are computed in
Fourier space:

```text
d/dx  <->  i kx
d/dy  <->  i ky
Laplacian <-> -|k|^2
```

The nonlinear advection term is easier to evaluate in physical space:

```text
(u . grad)u = [
  u du/dx + v du/dy,
  u dv/dx + v dv/dy
]
```

The pseudo-spectral loop is:

1. FFT velocity.
2. Project velocity to remove numerical divergence.
3. Compute spectral derivatives.
4. Transform derivatives back to physical space.
5. Form the nonlinear product in physical space.
6. FFT the nonlinear product.
7. Apply dealiasing.
8. Project the nonlinear term so pressure is eliminated.

The code uses the standard two-thirds dealiasing mask when `dealias=True`.
That removes high-frequency modes where quadratic nonlinear products would
alias back into lower frequencies.

## 6. Time Stepping

The velocity equation after pressure projection is

```text
du/dt = P[-(u . grad)u + f] + nu Laplacian u
```

The implemented step treats nonlinear advection explicitly and viscosity
semi-implicitly:

```text
u_hat_next =
  (u_hat + dt * P[-advection_hat + forcing_hat])
  / (1 + dt * nu * |k|^2)
```

Then the solver applies the projection again:

```text
u_next = inverse_fft(P(u_hat_next))
```

This final projection is cheap and important. It prevents accumulated roundoff
or model corrections from leaving the incompressible subspace.

## 7. Diagnostics

For each velocity frame the simulator records:

- Kinetic energy:

```text
E = 1/2 mean(u^2 + v^2)
```

- Vorticity:

```text
omega = dv/dx - du/dy
```

- Enstrophy:

```text
Z = 1/2 mean(omega^2)
```

- Incompressibility error:

```text
max |div u|
```

- Speed statistics and vorticity range.

The divergence diagnostic is the quick health check. For exact projected solver
steps it should stay near floating-point roundoff. For AI predictions, this
project applies `project_velocity` after every model inference, so the stored
AI and hybrid velocity states are also projected back to `div u = 0`.

## 8. AI Surrogate Problem

The AI surrogate learns a map from one projected velocity state to a future
projected velocity state:

```text
G_theta: [u_t, v_t] -> [u_{t+s}, v_{t+s}]
```

where `s` is the target stride in exact solver steps. For example:

```text
--target-stride 8
```

means one neural network call tries to replace eight exact solver steps.

The training data is generated by the exact solver:

```text
velocity trajectory:
q_0, q_s, q_2s, ...

training pairs:
(q_0 -> q_s), (q_s -> q_2s), ...
```

The tensor shape is:

```text
[batch, channels, n, n] = [batch, 2, n, n]
```

The existing surrogate code now supports both:

```text
channels = 1  # scalar vorticity
channels = 2  # velocity components u and v
```

## 9. CNN vs FNO Surrogates

Two surrogate families are available:

- CNN: local periodic convolutions with circular padding.
- FNO: Fourier Neural Operator layers that learn low-frequency spectral mixing.

The FNO is a natural candidate here because the exact solver is spectral and
periodic. It can learn global low-frequency couplings more directly than a
small local CNN. The tradeoff is that FNO inference is often more expensive per
model call because it uses FFTs inside the neural network.

## 10. AI Rollout Modes

The new comparison command has the same three rollout modes as the vorticity
pipeline:

### Exact solver

```text
q_{m+1} = exact_solver^s(q_m)
```

### Pure AI

```text
q_{m+1} = P(G_theta(q_m))
```

The projection `P` is applied after every neural prediction.

### Hybrid AI plus exact correction

Most steps use AI:

```text
q_{m+1} = P(G_theta(q_m))
```

Every `correction_interval` model steps, the exact solver re-anchors the state:

```text
q_{m+1} = exact_solver^s(q_m)
```

This hybrid mode trades some speed for stability and physical consistency.

## 11. Commands

Train a small FNO surrogate for the explicit incompressible velocity solver:

```bash
python -m fluid_ai_sim.train_incompressible_surrogate \
  --model fno \
  --n 32 \
  --trajectories 8 \
  --steps 48 \
  --target-stride 4 \
  --epochs 4 \
  --width 16 \
  --depth 2 \
  --modes 8 \
  --checkpoint runs/incompressible_fno.pt
```

Compare exact solver, AI, and hybrid rollouts:

```bash
python -m fluid_ai_sim.compare_incompressible_modes \
  --checkpoint runs/incompressible_fno.pt \
  --use-checkpoint-config \
  --steps 80 \
  --correction-interval 5 \
  --out runs/incompressible_fno_compare
```

The comparison writes:

- `solver/`, `ai/`, and `hybrid/` run folders.
- `trajectory.npz` containing velocity, vorticity, and diagnostics.
- `diagnostics.json` and `diagnostics.csv`.
- `comparison_metrics.json` and `comparison_metrics.npz`.
- `plots/velocity_error_to_solver.png`.
- `plots/energy_divergence.png`.
- `plots/final_vorticity.png`.
- `plots/speed_comparison.png`.

## 12. What Accuracy Means Here

The AI model is judged against the exact projected spectral solver. Important
checks are:

- Velocity RMSE and relative L2 error.
- Drift in kinetic energy.
- Drift in enstrophy.
- Divergence error after projection.
- Visual vorticity structure.
- Throughput in solver-equivalent steps per second.

Projection can enforce incompressibility, but it cannot guarantee the AI model
gets the correct energy cascade, phase, or vortex dynamics. A projected bad
prediction is still divergence-free, but it may be physically wrong. That is
why the hybrid mode and diagnostics matter.

## 13. Practical Next Steps

The branch is structured for iterative improvement:

- Train larger FNO models at `n=64` and `n=128`.
- Add physics losses for energy, enstrophy, divergence before projection, and
  Navier-Stokes residual consistency.
- Predict vorticity and velocity together, with a consistency loss between
  `omega` and `curl u`.
- Batch many initial conditions to better use GPU/MPS throughput.
- Compare stride choices: `s = 2, 4, 8, 16`.
- Validate against known periodic flows such as Taylor-Green vortex decay.

The immediate win is conceptual clarity: the solver now exposes the velocity
field, the pressure projection, and incompressibility directly while keeping
the AI acceleration interface familiar.
