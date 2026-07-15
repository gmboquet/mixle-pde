# mixle-pde

PDE and ODE-constrained Bayesian inverse problems for [mixle](https://github.com/gmboquet/mixle).

For `0.8.0`, this repository is also the integrated compatibility profile for Mixle Physics, Simulation, and
Discrete artifacts. Existing solvers remain available; new canonical adapters expose explicit capability,
conversion, residual, parity, and migration receipts rather than turning this package into a second semantic owner.

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
pip install -e ".[docs]"    # with the docs extras
```

A base install (zero extras) imports `mixle_pde` and runs the `mixle_pde.problem_adapter`
compatibility-boundary tests with no optional backend present -- every heavy dependency below is a
lazy, guarded import behind the module that needs it, never a hard requirement of the package.

### Optional extras

mixle-pde is the batteries-included PDE deployment profile: capability-family extras layer
solver accelerators and differentiable backends on top of the numpy/scipy/pyproj base, and
data-format extras layer geoscience file-format ingest. Mix and match; none is required by the
others or by the core test suite.

| Extra | Unlocks |
| --- | --- |
| `fem` | AMG (`pyamg`) and CHOLMOD (`scikit-sparse`) accelerators for large assembled FEM/simplex systems |
| `mesh` | AMG preconditioning (`pyamg`) for mesh-scale assembled linear systems |
| `mpi` | Forward-declared distributed mesh/solve transport (`mpi4py`); no backend is wired to it yet |
| `fvm` | Differentiable finite-volume forwards (`torch`) plus AMG scale-up (`pyamg`) |
| `coupling` | Differentiable coupled/multiphysics forwards (`torch`) |
| `inverse` | Differentiable adjoints, randomized-SVD low-rank UQ, and sparse-scale posterior solves (`torch`, `scikit-learn`, `pyamg`, `scikit-sparse`) |
| `surrogate` | Neural surrogate distillation and calibration (`torch`, `scikit-learn`) |
| `all` | The union of the seven capability extras above |
| `raster` / `netcdf` / `grib` / `segy` / `las` / `potfield` | Single-format geoscience data ingest (`mixle_pde.io.*`, `mixle_pde.env_data`, `mixle_pde.reductions`) |
| `data` | Convenience bundle of every data-format backend above |

```bash
pip install -e ".[fem]"
pip install -e ".[inverse,surrogate]"
pip install -e ".[all]"
```

Commercial/optional adapters (e.g. the sibling `mixle_mlops` task-cascade integration
`mixle_pde.surrogate` can optionally bind to) are never declared as a dependency of any extra and
are never required to install or test mixle-pde.

## Documentation

The Sphinx manual starts at [`docs/index.rst`](docs/index.rst). It includes installation notes, the
package map, solver/API navigation, validation guidance, generated API reference pages, and the
release-facing 3D/4D field modeling guide.

```bash
make -C docs html SPHINXOPTS="-W --keep-going"
```

Release notes and the current changelog are in
[`docs/release-notes.rst`](docs/release-notes.rst) and
[CHANGELOG.md](CHANGELOG.md).

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

Release-facing documentation for the new 3D/4D field posterior stack lives in
[`docs/0.6.x-field-modeling.md`](docs/0.6.x-field-modeling.md). It covers
latent fields, observations, priors, inversion, assimilation, geoscience
likelihoods, posterior queries, posterior calibration diagnostics, meshes, and readiness checks.

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
| `AnisotropicElasticWave3D` | anisotropic (VTI/TTI) elastodynamics | Thomsen / Bond-rotated stiffness, velocity-stress staggered grid |
| `ViscoacousticWave1D` | constant-Q attenuating wave | GSLS memory variables (tau-method) |
| `BiotPoroelastic1D` | Biot poroelasticity (fast + slow P) | velocity-stress-pressure staggered grid |
| `TransientHeat` | transient heterogeneous heat conduction | divergence-form + checkpointed time stepping |
| `SAFEPlate`, `safe_dispersion` | guided-wave (Lamb / SH) dispersion | semi-analytical finite elements |
| `EulerBernoulliBeam` | slender-beam bending and vibration | 1D fourth-order (biharmonic) |
| `KirchhoffPlate` | thin-plate bending, static and dynamic | 2D biharmonic |

More equations live in their own modules: `gas_dynamics` (1D compressible Euler with an exact Riemann
solver), `schrodinger` (time-dependent Schrodinger, split-step Fourier), `spectral_flow` (pseudo-spectral
2D/3D Navier-Stokes with an optional Smagorinsky LES closure), `wave_pml` (2D acoustic wave with a
perfectly-matched-layer boundary), and `fem` (P1 triangular finite elements for Poisson).

The reusable mesh layer is `SimplexMesh`: `box_simplex_mesh` creates deterministic simplex meshes in any
dimension, `delaunay_mesh` wraps SciPy Delaunay for scattered point clouds, and `space_time_mesh` extrudes a
3D tetrahedral mesh through time into a 4D simplex mesh. This is the geometry foundation for moving-domain
and transient finite-element work; adaptive remeshing, ALE/FSI coupling, and curved/high-order elements are
still future solver work.

The 3D/4D Earth posterior surface is modular. `Field3D` and `PosteriorField3D` represent one gridded
physical property with units, optional bounds, posterior mean/MAP, covariance, credible intervals, and
sampling. `Observation`, `ForwardOperator`, and `ForwardOperatorRegistry` provide a common geometry/noise/
provenance contract for measurements; the built-in registry operators cover gravity, magnetics, and direct
borehole/sensor samples. `FieldGaussianPrior` supplies graph-Matern smoothness, `linear_gaussian_invert`
performs exact linear-Gaussian inversion, and `gauss_newton_invert` adds bounded MAP/Laplace inversion for
properties such as porosity, susceptibility, or concentration. `dc_resistivity_forward_operator` wraps the
existing DC/ERT forward as a nonlinear log-conductivity observation with local finite-difference
sensitivities for Gauss-Newton posterior construction.
`layered_mt_forward_operator`, `aem_layered_forward_operator`, `mt_2d_te_forward_operator`,
`mt_3d_forward_operator`, and `csem_3d_forward_operator` do the same for 1D layered MT/AEM, 2D
TE-mode, 3D curl-curl magnetotelluric, and 3D controlled-source EM soundings, mapping
log-conductivity to real-valued geophysical observations.

`sparse_linear_gaussian_invert` stores a linear-Gaussian posterior in sparse precision-factor form,
using sparse covariance solves for marginals and linear derived quantities without retaining a dense
covariance matrix. `PosteriorField4D`, `assimilate_4d`, and `assimilate_4d_ensemble` add a time axis through exact
Kalman/RTS smoothing for linear observations and ensemble Kalman filtering for nonlinear observations,
exposing each time slice as an ordinary `PosteriorField3D`. `FieldGaussianPrior` can
assemble either dense or sparse CSR graph precision matrices. `depth_weights`,
`depth_weighted_marginal_precision`, `depth_weighted_marginal_precision_sparse`, `CrossPropertyPrior`, and
`joint_linear_gaussian_invert` provide depth-aware priors and cross-property coupling so observations of
one property can regularize another.
`GeochemAssay`, `MultiElementAssay`, `assay_log_likelihood`, `multi_element_assay_log_likelihood`,
`additive_log_ratio`, `BiostratConstraint`, and `biostrat_log_likelihood` add geochemical
detection-limit/compositional likelihoods, multi-element covariance/batch effects, and fossil range-zone
constraints. `GeochronologyAge`, `StratigraphicCorrelation`, and `FaciesIntervalConstraint` add isotopic
age measurements, relative-age/horizon constraints, and facies/environment interval evidence. These
likelihoods can be converted into field observations where a project defines the property mapping.
`posterior_query` extracts point/section/region marginals, linear derived quantities such as total anomalous
mass, low-rank or diagonal Gaussian summaries, and ensemble samples. `posterior_calibration` measures
synthetic-truth coverage, held-out observation fit, uncertainty inflation away from data, and
insufficient-observation flags. Full reaction-path geochemistry,
paleoecological/basin-process simulators, full truncated multivariate censoring for multi-element assays,
production-scale adjoint sensitivities, iterative sparse posterior solvers, and full airborne loop/flight-line
AEM geometry remain future validated kernels. The ensemble 4D path is a stochastic
Gaussian-summary reference, not a production particle/MCMC smoother.

### Inverse and inference layer

| Surface | What it does |
| --- | --- |
| `Differential` | an observation whose forward model is an ODE/PDE solution; recover latent drivers (rate constants, source fields, initial states, coefficients) with `joint([...]).fit(how=...)` |
| `PDE(operator).fit` | PDE-constrained latent-field state space (Kalman/RTS smoother + EM) over `DiffusionOperator`, `AdvectionOperator`, `AdvectionDiffusionOperator`, or any operator you `register_dynamics_operator` |
| `pde_solve` | adjoint-capable sparse PDE solves (differentiable Poisson / divergence-form) for large-scale inverse problems |
| `nonlinear_solve` | differentiable nonlinear steady solves `F(u;θ)=0` (Newton forward, implicit-function-theorem adjoint); the base for nonlinear elliptic inverse problems |
| `rtm_image`, `born_modeling`, `lsrtm_step` | reverse-time migration and least-squares / Born imaging over the wave steppers |
| `misfit` (`envelope` / `xcorr` / `wasserstein`) | cycle-skip-robust FWI misfit functionals for any wave forward |
| `helmholtz_pml_operator` | frequency-domain Helmholtz with a PML boundary and a complex modulus (viscoacoustic attenuation) |
| `shape_optimize`, `level_set_material` | level-set shape optimization and inverse shape inference |
| `CoupledPDESystem`, `solve_poisson` | nD steady diffusion/Poisson and node-coupled multiphysics |

### Geophysics

Near-surface forward operators plus an Occam-style regularized inversion engine that inverts any
differentiable forward without per-problem prior tuning: `gravity_point_sensitivity`,
`magnetic_dipole_sensitivity`, `dc_resistivity` (ERT), `straight_ray_operator` (traveltime tomography),
`depth_weighting`, `roughness_operator`, `regularized_gauss_newton`, and `cross_gradient` /
`joint_inversion` for structural coupling of several property models.

Electromagnetics in the diffusive (induction) regime, distinct from the wave-regime `Maxwell3D`:
`layered_mt_impedance` (1D magnetotelluric / airborne EM), `mt_2d_te` (2D magnetotelluric), and `mt_3d` /
`csem_3d` (a 3D edge-element / Yee curl-curl solver for CSEM, magnetotellurics, borehole induction, and
eddy-current NDE), plus `cole_cole_conductivity` / `sip_forward` for spectral induced polarization
(disseminated-sulphide detection). Potential fields extend to `gravity_gradient_tensor` (full-tensor
gradiometry) and `magnetic_vector_sensitivity` / `magnetic_gradient_tensor`.

### Petroleum systems

`geotherm` (steady conductive geotherm for a layered column) and `easy_ro` / `easy_ro_profile` (the
EASY%Ro vitrinite-reflectance maturation model of Sweeney & Burnham 1990), differentiable forwards for
heat-flow and thermal-history inversion. `gassmann_ksat` / `fluid_substitute` give closed-form
differentiable Gassmann fluid substitution, turning elastic-FWI velocities into reservoir variables
(porosity, saturation).

### Biomolecular electrostatics and reaction-diffusion

`linearized_pbe` and `nonlinear_pbe` solve the Poisson-Boltzmann equation (linear Debye-Huckel and the
full sinh form) for biomolecular electrostatics, with `reaction_field_energy` for MM-PBSA-style binding
free energy. `pnp_equilibrium` is the equilibrium Poisson-Nernst-Planck ion-channel model, and
`smoluchowski_rate_radial` / `smoluchowski_rate_box` give diffusion-limited association on-rates. All
build on the differentiable `nonlinear_solve` keystone.

### Sonar and radar propagation

Long-range sonar and radar are the same problem: propagation through a range-varying medium. The shared
keystone is `ParabolicEquation2D`, a range-marched split-step Fourier one-way propagator that serves both
by swapping only the environmental potential and the boundary. Underwater it marches on the acoustic index
from `mackenzie` / `unesco` sound speed c(T,S,depth); in the troposphere it marches on the modified
refractivity from `refractivity` / `modified_refractivity` (ITU-R P.453) so radar ducting falls out. It is
differentiable, so it drops into `Differential`.

Around it: `boundaries` (seabed Rayleigh reflection with the critical grazing angle, rough-surface coherent
loss, and the radar Fresnel surface impedance); `attenuation` (Thorp / Francois-Garrison for seawater,
ITU-R P.676 gaseous and P.838 rain for the atmosphere) feeding the solvers' complex-modulus `Q` slot;
`NormalModes1D`, a KRAKEN-style differentiable depth-mode solver verified against the Pekeris waveguide and
cross-checked against the PE; and `WavenumberIntegration1D`, the OASES-style full-wave reference. For small
scenes and targets, `po_rcs` / `knife_edge_diffraction` / `two_ray_pattern` / `multipath_power` give
asymptotic radar cross sections and urban multipath.

The inverse problems come for free on the `Differential` stack: `refractivity_from_clutter` recovers an
atmospheric duct from radar clutter, and `ocean_sound_speed_inversion` recovers a sound-speed anomaly from a
received acoustic field. `env_data` assembles real profiles and bathymetry/terrain into the range-depth
coefficient fields (differentiable interpolation + seabed masking), with import-guarded loaders for GEBCO,
World Ocean Atlas / Argo, DEM, and ERA5 data behind an optional extra.

### Cross-modal reasoning

`JointPotentialField`, `SpatialFieldStore`, and `MechanisticFieldReasoner` fuse geophysical modalities
into a spatial belief with uncertainty, feeding mixle's `reason` surface.

### Modeling readiness

Applications should not infer solver availability from imports or README claims. The package exposes a
small capability catalog and deterministic readiness checks:

```python
from mixle_pde import readiness_report, assert_required_modeling

print(readiness_report())
assert_required_modeling()
```

Canonical callers should additionally inspect `native_backend_manifest()` and may use
`solve_p1_poisson_canonical()` for the first legacy-compatible, receipt-bearing FEM migration path.

The required modeling gate currently checks:

- 3D tetrahedral meshes, direct 4D simplex meshes, and 3D-to-4D space-time extrusion.
- Censored geochemical assay likelihoods, compositional transforms, biostratigraphic range-zone likelihoods,
  geochronology age likelihoods, stratigraphic correlation constraints, and facies/environment intervals with
  provenance and units.
- The common observation/forward-operator contract for gravity, magnetics, and borehole/sensor samples.
- Exact 3D linear-Gaussian field inversion and bounded-field Gauss-Newton MAP/Laplace inversion.
- Nonlinear DC/ERT log-conductivity posterior observations through the Gauss-Newton path.
- 4D random-walk Kalman assimilation plus RTS smoothing, and ensemble nonlinear 4D assimilation, with
  posterior time-slice extraction.
- Depth weighting, graph-Matern smoothness, and cross-property Gaussian coupling.
- Posterior extraction for points, regions/volumes, sections, linear derived quantities, low-rank/diagonal
  summaries, and ensemble samples.
- Posterior calibration diagnostics for truth coverage, held-out fit, uncertainty inflation, and insufficient
  observations.
- PDE-constrained state-space smoothing and forecasting.
- Transient diffusion / heat-equation decay against a discrete analytical rate.
- Potential-field geophysics sign and linearity.
- Mechanistic field reconstruction from sparse sensors.
- 2D acoustic wave propagation stability.

These checks are intentionally cheap smoke-and-physics scenarios. They do not replace the full analytic
test suite; they give applications and CI a quick answer to "is the required modeling surface actually
present and runnable in this environment?"

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
pytest tests/capabilities_test.py -q
```

## License

MIT. See [LICENSE](LICENSE).
