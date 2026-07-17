# Migrating legacy kernel calls to the canonical adapter boundary

This is an additive migration path, not a breaking change. Every legacy `mixle_pde` module -- `wave.py`,
`flow.py`, `maxwell.py`, `dynamics.py`, `fem.py`, and the rest -- keeps its existing public functions and
classes exactly as they are today. Nothing described here removes, renames, or reinterprets a legacy call.
What is new is a second, optional way to reach a subset of those same kernels: a typed, capability-checked
boundary that a caller (human or agent) can query for support *before* it runs anything, instead of calling
a specific solver class directly and hoping the parameters it passes make sense for that solver.

If your code already calls a legacy solver directly, it does not need to change. This guide is for callers
who want the capability-negotiation and evidence properties of the new boundary -- or who are building a
tool/agent surface that should be able to ask "can any registered backend solve this problem?" instead of
importing a specific module.

## Two adapter boundaries exist today -- do not conflate them

`mixle_pde` currently ships two distinct compatibility boundaries. They solve different problems and are at
different points of maturity; a caller migrating from a legacy direct call should pick the one that matches
what they are trying to do.

1. **`mixle_pde.canonical_adapter`** -- a receipt-bearing path for exactly one problem shape: scalar P1
   Poisson/diffusion on a simplex mesh. `solve_p1_poisson_canonical()` runs the legacy P1 assembly
   (`mixle_pde.fem`), also expresses the same linear system as an exact-rational
   `mixle.sim.finite-linear-system/v1` record, solves that record through
   `solve_sim_linear_system()`, and returns both the legacy-compatible solution array and a
   `CanonicalSolveReceipt` (record digest, source-problem digest, backend-capability digest, residual,
   relative residual, and an explicit `legacy_parity_max_abs` figure showing how far the canonical solve
   drifted from the legacy one). This is the path documented in `docs/migrations/0.8.0.md`. It covers one
   kernel family only.

2. **`mixle_pde.problem_adapter` + `mixle_pde.pde_backend_registry`** -- a general capability-negotiation
   boundary for solver-neutral `CON-MATH-PROBLEM-V1` study dictionaries. `problem_adapter.py` defines the
   contract (`PDEBackendProfile`, `PDECompatibilityReport`, `UnsupportedPDEProblem`,
   `require_compatible()`/`inspect_math_problem()`) but ships with no concrete backend of its own.
   `pde_backend_registry.py` is the wiring that registers real, unmodified legacy kernels behind that
   contract, each as a `PDEKernelRegistration` with typed `PDEPort` inputs/outputs, a `PDEKernelArtifact`
   describing the output shape, and a string `invoke_key` (never a live callable, so the registration stays
   safe to hash/serialize). `run_math_problem(problem, backend_id)` always calls `require_compatible()`
   first: an operator kind, discretization, mesh cell type, or evidence kind a profile does not declare
   raises `UnsupportedPDEProblem` rather than being silently solved by whichever backend happens to be
   registered. This is the boundary the rest of this guide covers, because it is the one that currently
   reaches multiple legacy kernel families.

`pde_backend_registry` is not re-exported from top-level `mixle_pde` (only `problem_adapter`'s contract
types are, via `mixle_pde/__init__.py`); import it explicitly:

```python
from mixle_pde.pde_backend_registry import get_kernel_registration, list_kernel_registrations, run_math_problem
```

## Legacy kernel families currently reachable through `pde_backend_registry`

As of this writing, `mixle_pde/pde_backend_registry.py` registers exactly five backends (confirmed by
reading the module directly, not inferred from documentation elsewhere):

| Backend id | Legacy source | What it does |
|---|---|---|
| `fem-p1-simplex` | `mixle_pde.fem` (`solve_simplex_poisson`, `assemble_simplex_stiffness_matrix`, `assemble_simplex_load_vector`) + `mixle_pde.mesh.box_simplex_mesh` | P1 simplex assembly/solve for `-div(diffusion grad u) = source` with Dirichlet boundaries. |
| `wave-fd-leapfrog` | `mixle_pde.wave.WaveEquation2D` | Explicit leapfrog FD stepper for the 2D acoustic wave equation, with an absorbing sponge boundary. |
| `flow-fd-streamfunction` | `mixle_pde.flow.NavierStokes2D` | FD streamfunction-vorticity stepper for 2D incompressible Navier-Stokes. |
| `em-fdtd-yee` | `mixle_pde.maxwell.Maxwell3D` | Yee-grid FDTD stepper for the source-free 3D Maxwell curl equations (PEC cavity). |
| `transport-fd-advdiff` | `mixle_pde.dynamics.AdvectionDiffusionOperator` | Method-of-lines advection-diffusion transport operator (implicit/explicit/exact transition schemes). |

Each registration's `evidence` payload is a **numerical-solve-correctness** check (finite state, bounded
growth, a small discretized residual, a preserved discrete invariant such as mass or `div(mu H)`) -- it is
not a claim that the underlying physics is experimentally validated. See each `_invoke_*` function's
docstring in `pde_backend_registry.py` for the exact evidence semantics of that backend.

A pseudo-spectral Navier-Stokes backend is proposed for registration in a separate, currently open and
**unmerged** pull request. Until that PR lands on `release/0.8.0`, it is not part of the reachable set and
is not covered by this guide; do not assume it is available.

## Worked example: the FD wave solver, old API and new boundary, side by side

Both snippets below drive the exact same underlying kernel (`mixle_pde.wave.WaveEquation2D`) with the same
parameters and produce numerically identical output. The canonical-adapter path does not reinterpret or
re-derive the physics; it is a typed front door onto the same call.

### Old direct API

```python
import torch

from mixle_pde.ops import make_ops
from mixle_pde.wave import WaveEquation2D

n = 16
dt = 0.02
n_steps = 10
wave_speed = 1.0
amplitude = 1.0
source_node = (n // 2) * n + n // 2

wave = WaveEquation2D(n, dt=dt)
ops = make_ops()
state = wave.pack(
    torch.zeros(n * n, dtype=torch.float64),
    torch.zeros(n * n, dtype=torch.float64),
)
c2 = wave_speed**2
for step in range(n_steps):
    source = ops.zeros(n * n)
    if step == 0:
        source = source.clone()
        source[source_node] = amplitude
    state = wave.step(state, c2, ops, source=source)

legacy_displacement = wave.displacement(state).detach().numpy()
```

### New canonical adapter path

```python
from mixle_pde.pde_backend_registry import run_math_problem

problem = {
    "id": "wave-migration-demo",
    "domains": [{"id": "domain", "kind": "mesh", "properties": {"mesh_cell_type": "structured_grid"}}],
    "unknowns": [{"id": "field", "domain_id": "domain"}],
    "operators": [
        {
            "id": "wave-migration-demo-operator",
            "kind": "time_stepping",
            "input_ids": ["field"],
            "output_ids": ["field"],
            "discretization": "FD-leapfrog",
        }
    ],
    "constraints": [],
    "objectives": [{"id": "solution", "sense": "satisfy", "expression": {}}],
    "evidence_requests": [{"kind": "convergence", "required": True}],
    "solve_plan": {
        "parameters": {
            "grid_size": n,
            "dt": dt,
            "n_steps": n_steps,
            "wave_speed": wave_speed,
            "amplitude": amplitude,
            "source_node": source_node,
        }
    },
}

result = run_math_problem(problem, "wave-fd-leapfrog")
canonical_displacement = result.solution  # numpy array, shape (n * n,)
```

`result.compatibility_report.supported` is `True`, and `result.evidence["convergence"]["finite"]` is
`True`. Comparing the two arrays (`legacy_displacement` vs `canonical_displacement`) gives a maximum
absolute difference of `0.0` -- both paths run the identical `WaveEquation2D` stepper with identical
parameters, so they agree exactly, not just approximately. `tests/migration_guide_example_test.py`
executes both snippets and asserts this, so the claim is checked on every test run rather than only stated
here.

If a caller instead asks `run_math_problem` for a discretization, evidence kind, or mesh cell type
`wave-fd-leapfrog` does not declare (for example requesting `"residual"` evidence, which this backend does
not produce), the call raises `UnsupportedPDEProblem` with the specific unsupported feature named in the
report -- it does not fall back to solving the problem some other way. `tests/pde_backend_registry_test.py`
already covers this rejection behavior for all five backends; this guide does not repeat that coverage.

## What is explicitly NOT yet migrated

Being honest about scope here matters more than looking complete:

- **Only 5 of `mixle_pde`'s ~94 inventoried public modules** (see `docs/capability-matrix.json`) are wired
  behind `pde_backend_registry`. The FEM registration alone covers `mixle_pde.fem`'s P1 simplex
  Poisson/diffusion assembler -- it does not cover mixed formulations, higher-order elements, or any other
  solver in `fem.py`.
- `mixle_pde.multiphysics` (coupled multi-field problems) has no `pde_backend_registry` registration at all.
- The inverse/assimilation/UQ/surrogate modules (`inverse.py`, `field_assimilation.py`, `field_gauss_newton.py`,
  `field_mcmc.py`, `field_priors.py`, the `surrogate`-family modules, etc.) have no registration; they remain
  reachable only through their existing direct APIs.
- `mixle_pde.mesh` more broadly (beyond the one `box_simplex_mesh` helper the FEM registration calls) is not
  exposed as a standalone canonical port.
- The `canonical_adapter` receipt-bearing path (digests, `CanonicalSolveReceipt`, exact-rational conversion)
  exists only for the single `fem-p1-simplex` / P1 Poisson case; the other four `pde_backend_registry`
  backends return evidence but do not yet produce a `CanonicalSolveReceipt`-style artifact.
- The pseudo-spectral Navier-Stokes backend mentioned above is not merged and is not part of the reachable
  set today.
- Full MP-M4 acceptance (twelve end-to-end examples across agent-driven multiphysics, Bayesian inversion,
  and certified hybrid surrogate cases) is a larger, separate body of work than this guide; this document
  and its accompanying test cover one worked example on one legacy kernel family, not the full worklist.

Treat `list_kernel_registrations()` as the source of truth for what is actually reachable at any given
moment -- it returns the live registration table, not a document that can drift out of date.

## Compatibility guarantee

Per this repository's own migration invariant (see `AGENTS.md` and `docs/migrations/0.8.0.md`): no artifact
or API is rewritten or removed automatically, and existing public imports and verified solver behavior
remain compatible until an explicit deprecation gate passes. Every legacy call shown in the "old direct
API" section above continues to work unmodified; nothing in this guide requires a caller to change existing
code. The full pre-existing public test suite (including `tests/wave_test.py`,
`tests/pde_backend_registry_test.py`, and `tests/problem_adapter_test.py`) remains green -- see the test
run reported alongside the change that added this guide.
