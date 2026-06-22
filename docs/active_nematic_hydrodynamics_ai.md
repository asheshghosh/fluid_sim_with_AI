# Active Nematic Hydrodynamics With AI Acceleration

This note documents the active-nematic extension in
`fluid_ai_sim.active_nematic`. It builds on the periodic incompressible solver
and adds a Beris-Edwards-style Q-tensor field coupled to flow through active
stress.

The simulated state is

```text
state = [u, v, Qxx, Qxy]
```

where `(u, v)` is the incompressible velocity and the 2D traceless nematic
tensor is

```text
Q = [[ Qxx,  Qxy],
     [ Qxy, -Qxx]]
```

This representation avoids the director ambiguity `n = -n`. If
`Qxx = S cos(2 theta)` and `Qxy = S sin(2 theta)`, then `theta` and
`theta + pi` represent the same nematic orientation.

## 1. Governing Fields

The active nematic model has two coupled fields:

- Velocity `u_i`.
- Nematic order tensor `Q_ij`.

The velocity remains incompressible:

```text
partial_i u_i = 0
```

The Q tensor describes local orientational order. Its scalar magnitude is

```text
|Q| = sqrt(Qxx^2 + Qxy^2)
```

and its director angle is

```text
theta = 1/2 atan2(Qxy, Qxx)
```

## 2. Velocity Equation

The velocity equation is a projected Navier-Stokes equation with active stress:

```text
partial_t u + (u . grad)u =
  -grad p + nu Laplacian u + P div(sigma_active)

div u = 0
```

The active stress is represented as

```text
sigma_active = alpha Q
```

where `alpha` is the activity. In this sign convention, larger `|alpha|`
injects stronger active forcing into the flow.

For the 2D traceless Q tensor,

```text
div(sigma_active)_x = alpha (partial_x Qxx + partial_y Qxy)
div(sigma_active)_y = alpha (partial_x Qxy - partial_y Qxx)
```

Pressure is eliminated with the same Fourier-space incompressibility projection
used by the velocity-pressure solver:

```text
P_k a = a - k (k . a) / |k|^2
```

Every exact step and every AI-predicted state is projected back into
`div u = 0`.

## 3. Q-Tensor Evolution

The Beris-Edwards-style Q equation is

```text
(partial_t + u . grad)Q - S(W, Q) = Gamma H
```

where:

- `Gamma` is the rotational diffusivity.
- `H` is the molecular field from the liquid-crystal free energy.
- `S(W, Q)` is the flow-alignment and co-rotation term.
- `W_ij = partial_j u_i` is the velocity gradient.

The velocity gradient is decomposed into strain and rotation:

```text
D = (W + W^T) / 2
Omega = (W - W^T) / 2
```

The solver uses the common Beris-Edwards alignment structure:

```text
S = (xi D + Omega)(Q + I/2)
  + (Q + I/2)(xi D - Omega)
  - 2 xi (Q + I/2)(Q : D)
```

where `xi` is the flow-alignment parameter.

## 4. Molecular Field

The implementation uses a compact one-constant Landau-de Gennes-like molecular
field:

```text
H = K Laplacian Q + B (S0^2 - |Q|^2) Q
```

where:

- `K` is `elastic_constant`.
- `B` is `bulk_strength`.
- `S0` is `preferred_order`.

The bulk term relaxes `|Q|` toward `S0`. The elastic term smooths spatial
gradients in Q.

## 5. Spectral Time Step

The solver is pseudo-spectral:

```text
partial_x <-> i kx
partial_y <-> i ky
Laplacian <-> -|k|^2
```

The velocity step treats advection and active forcing explicitly and viscosity
semi-implicitly:

```text
u_hat_next =
  (u_hat + dt P[-advection_hat + active_force_hat])
  / (1 + dt nu |k|^2)
```

The Q step treats advection, flow alignment, and bulk relaxation explicitly,
and Q elasticity semi-implicitly:

```text
Q_hat_next =
  fft(Q + dt[-u.grad Q + S + Gamma B(S0^2 - |Q|^2)Q])
  / (1 + dt Gamma K |k|^2)
```

Both velocity and Q updates use a two-thirds dealiasing mask by default.

## 6. Defect Diagnostics

Active nematics are organized by topological defects. The code detects defects
from the phase

```text
phi = atan2(Qxy, Qxx) = 2 theta
```

For each periodic plaquette, it computes the wrapped winding of `phi`. A
`+2 pi` winding in `phi` corresponds to a `+1/2` nematic defect, and a
`-2 pi` winding corresponds to a `-1/2` defect.

Diagnostics include:

- `defects_plus_half`.
- `defects_minus_half`.
- `defect_total`.
- `defect_density`.
- `net_topological_charge`.
- `q_order_mean`.
- `free_energy_density`.
- `active_length = sqrt(K / |alpha|)`.
- `divergence_linf`.

The defect count is intentionally simple and grid-based. It is useful for
simulator diagnostics and model comparison, but high-quality experimental
defect tracking usually needs additional filtering and sub-pixel localization.

## 7. AI Acceleration

The FNO surrogate learns the full coupled state:

```text
G_theta: [u_t, v_t, Qxx_t, Qxy_t]
      -> [u_{t+s}, v_{t+s}, Qxx_{t+s}, Qxy_{t+s}]
```

where `s` is `--target-stride`. For example, `--target-stride 4` means one
network inference replaces four exact Beris-Edwards solver steps.

The surrogate uses four input channels:

```text
[batch, 4, n, n]
```

The FNO is a natural fit because the physical solver is periodic and spectral:
both the exact solver and the model operate well on global Fourier information.

## 8. Rollout Modes

The comparison runner supports three modes:

### Exact Solver

```text
state_{m+1} = exact_solver^s(state_m)
```

### Pure AI

```text
state_{m+1} = project_state(G_theta(state_m))
```

The velocity part is projected to `div u = 0`, and Q magnitude is clipped to a
configured maximum to avoid runaway predictions during long rollouts.

### Hybrid

Most model steps use AI:

```text
state_{m+1} = project_state(G_theta(state_m))
```

Every `correction_interval` model steps, the exact solver is used instead:

```text
state_{m+1} = exact_solver^s(state_m)
```

This follows the same idea as the Navier-Stokes acceleration work: neural
rollouts are fast but drift; exact corrections re-anchor the trajectory.

## 9. Commands

Train a small active-nematic FNO:

```bash
python -m fluid_ai_sim.train_active_nematic_surrogate \
  --model fno \
  --n 32 \
  --trajectories 8 \
  --steps 48 \
  --target-stride 4 \
  --epochs 6 \
  --width 16 \
  --depth 2 \
  --modes 8 \
  --checkpoint runs/active_nematic_fno.pt
```

Compare exact, AI, and hybrid rollouts:

```bash
python -m fluid_ai_sim.compare_active_nematic_modes \
  --checkpoint runs/active_nematic_fno.pt \
  --use-checkpoint-config \
  --steps 80 \
  --correction-interval 5 \
  --out runs/active_nematic_compare
```

The comparison writes:

- `comparison_metrics.json` and `.npz`.
- Per-mode `trajectory.npz` with `state`, `velocity`, `q`, `vorticity`,
  `q_order`, and `director_angle`.
- `diagnostics.json` and `.csv`.
- `plots/state_error_to_solver.png`.
- `plots/energy_free_energy_divergence.png`.
- `plots/order_defects.png`.
- `plots/final_order_vorticity.png`.
- `plots/speed_comparison.png`.

## 10. What We Borrow From Active-Nematic ML Papers

The active-nematic ML literature suggests three practical design choices:

1. Use smooth Q-tensor components instead of raw director angles.
2. Track defects and active length scales, not just pixel-wise error.
3. Treat parameter inference and time forecasting as related but distinct
   tasks.

This branch implements the forecasting side first. A natural next step is a
parameter-estimation model:

```text
[Q(t0), Q(t1), ...] -> [activity alpha, elastic constant K, flow alignment xi]
```

That would mirror the hydrodynamic-parameter inference strategy used in the
active-nematic machine-learning work.
