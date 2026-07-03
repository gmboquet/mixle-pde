# mixle-pde

PDE and ODE-constrained Bayesian inverse problems for [mixle](https://github.com/gmboquet/mixle).

mixle-pde is a `mixle.ppl` plugin. Importing it wires a stack of differentiable forward solvers and
PDE-constrained state-space models into mixle's probabilistic-programming surface through mixle's
extension hooks. It never patches mixle: the plugin depends on mixle, not the reverse.

The organizing idea is that many quantities are observable only through a dynamical system they drive.
A rate constant shows up in a decay curve, a contaminant source in downstream concentrations, a
subsurface velocity in a seismic record, a diffusivity in a steady temperature field. mixle-pde gives
you the forward physics as differentiable solvers and the inverse machinery to recover a posterior over
the hidden drivers from noisy, partial, indirect observations.

## Install

mixle-pde installs into an environment that already has mixle (which brings torch, numba, and the rest).

```bash
pip install -e .            # from a checkout
pip install -e ".[test]"    # with the test extras
```

## Quickstart

### Recover a hidden driver from a dynamical system

`Differential` is an observation whose forward model is the solution of an ODE or PDE. The latent
drivers are `free` handles (or a shared field a `GP` carries); a single callback supplies the physics
through a backend-agnostic `ops` namespace, so it never imports a tensor library.

```python
import numpy as np
from mixle.ppl import free, joint
from mixle_pde import Differential

t = np.linspace(0.0, 5.0, 40)
y_obs = np.exp(-0.7 * t) + 0.02 * np.random.default_rng(0).standard_normal(t.size)

k = free(1, name="k", support="positive")
obs = Differential(y_obs, drivers=[k], y0=1.0, t_grid=t, scale=0.05,
                   rhs=lambda u, t, p, ops: -p.k * u)      # dy/dt = -k y
post = joint([obs]).fit(how="laplace")
k_mean, k_sd = post.posterior("k")                          # ~0.7 with a calibrated sd
```

Swap the `rhs`/`forward` callback for a spatial PDE and hand it a `GP` field to recover a whole source
map instead of a scalar. Fit with `how="laplace"` (or `"map"` for a point estimate, `"gauss_newton"`,
`"vi"`); `posterior` returns a mean and sd once a curvature-bearing fit like Laplace has run.

### Run a forward solver directly

Every solver steps through the same `ops` namespace, so the forward run is differentiable end to end.

```python
import numpy as np
from mixle_pde import WaveEquation3D
from mixle_pde.ops import make_ops

ops = make_ops()
n = 48
solver = WaveEquation3D(n, dt=0.4 / (n - 1), absorb_width=6)   # CFL-safe step, sponge layer

c2 = ops.tensor(np.ones(n**3))                                 # squared wave speed per cell
u0 = np.zeros((n, n, n)); u0[n // 2, n // 2, n // 2] = 1.0     # a point disturbance
state = solver.pack(u0.ravel(), np.zeros(n**3))
for _ in range(100):
    state = solver.step(state, c2, ops)
field = solver.displacement(state).reshape(n, n, n)
```

### Fit a PDE-constrained state space

`PDE(operator)` is a latent-field model whose linear state transition is fixed by the physics. Fit it on
a `(T, m)` array of noisy field snapshots: a Kalman/RTS smoother recovers the latent field while EM
estimates the process and observation noise.

```python
from mixle_pde import PDE, DiffusionOperator

model = PDE(DiffusionOperator(0.1, n)).fit(field_snapshots, dt=0.1)
```

## Solver catalog

### Forward solvers

All are differentiable through the `ops` backend and each ships with a test that checks it against an
exact analytical solution (a normal-mode frequency, a decaying eigenmode, a Poiseuille profile).

| Solver | Equation | Method |
| --- | --- | --- |
| `WaveEquation2D`, `WaveEquation3D` | acoustic wave | leapfrog, sponge absorbing layer |
| `NavierStokes2D`, `NavierStokes3D` | incompressible Navier-Stokes | streamfunction-vorticity (2D), Chorin projection (3D) |
| `TwoPhaseFlow2D` | immiscible two-fluid flow (core-annular / lubricated pipelining) | diffuse-interface phase field, variable-property Chorin projection |
| `Maxwell3D` | source-free Maxwell curl equations | FDTD on a Yee staggered grid |
| `ElasticWave3D` | isotropic elastodynamics (P and S waves) | velocity-stress staggered grid (Virieux) |
| `EulerBernoulliBeam` | slender-beam bending and vibration | 1D fourth-order (biharmonic) |
| `KirchhoffPlate` | thin-plate bending, static and dynamic | 2D biharmonic |

More equations live in their own modules: `gas_dynamics` (1D compressible Euler with an exact Riemann
solver), `schrodinger` (time-dependent Schrodinger, split-step Fourier), `spectral_flow` (pseudo-spectral
2D/3D Navier-Stokes with an optional Smagorinsky LES closure), `wave_pml` (2D acoustic wave with a
perfectly-matched-layer boundary), and `fem` (P1 triangular finite elements for Poisson).

### Inverse and inference layer

| Surface | What it does |
| --- | --- |
| `Differential` | an observation whose forward model is an ODE/PDE solution; recover latent drivers (rate constants, source fields, initial states, coefficients) with `joint([...]).fit(how=...)` |
| `PDE(operator).fit` | PDE-constrained latent-field state space (Kalman/RTS smoother + EM) over `DiffusionOperator`, `AdvectionOperator`, `AdvectionDiffusionOperator`, or any operator you `register_dynamics_operator` |
| `pde_solve` | adjoint-capable sparse PDE solves (differentiable Poisson / divergence-form) for large-scale inverse problems |
| `shape_optimize`, `level_set_material` | level-set shape optimization and inverse shape inference |
| `CoupledPDESystem`, `solve_poisson` | nD steady diffusion/Poisson and node-coupled multiphysics |

### Geophysics

Near-surface forward operators plus an Occam-style regularized inversion engine that inverts any
differentiable forward without per-problem prior tuning: `gravity_point_sensitivity`,
`magnetic_dipole_sensitivity`, `dc_resistivity` (ERT), `straight_ray_operator` (traveltime tomography),
`depth_weighting`, `roughness_operator`, `regularized_gauss_newton`, and `cross_gradient` /
`joint_inversion` for structural coupling of several property models.

### Petroleum systems

`geotherm` (steady conductive geotherm for a layered column) and `easy_ro` / `easy_ro_profile` (the
EASY%Ro vitrinite-reflectance maturation model of Sweeney & Burnham 1990), differentiable forwards for
heat-flow and thermal-history inversion.

### Cross-modal reasoning

`JointPotentialField`, `SpatialFieldStore`, and `MechanisticFieldReasoner` fuse geophysical modalities
into a spatial belief with uncertainty, feeding mixle's `reason` surface.

## How it works

Every solver and inverse callback talks to a single `ops` namespace (`mixle_pde/ops.py`): a float64
torch backend that provides N-dimensional finite differences (`ops.grad`, `ops.laplacian`), a
differentiable sparse Poisson solve with adjoint gradients (`ops.sparse_solve`), and the small array
algebra the steppers need. Because the physics is written against `ops` and never imports torch
directly, a forward solver is differentiable end to end, which is exactly what lets it drop into the
`Differential` inverse stack and be fit by gradient-based MAP, Laplace, Gauss-Newton, or VI.

## Tests

```bash
pytest                              # the full suite (-n auto via pyproject)
pytest tests/wave3d_test.py -q      # one file
```

## License

MIT. See [LICENSE](LICENSE).
