"""Concrete legacy-kernel registrations behind the :mod:`mixle_pde.problem_adapter` boundary.

``problem_adapter`` defines the compatibility contract (:class:`~mixle_pde.problem_adapter.PDEBackendProfile`,
:func:`~mixle_pde.problem_adapter.require_compatible`) but ships with no concrete backend -- nothing in
mixle-pde was actually reachable through a ``CON-MATH-PROBLEM-V1`` study. This module is the wiring: it
registers mixle-pde's existing FD/FDTD/FEM solvers as :class:`~mixle_pde.problem_adapter.PDEBackendProfile`
instances, each paired with typed input/output ports, an output artifact record, and a string-keyed
invocation function that drives the real (unmodified) legacy kernel.

Eight kernels are registered, covering the MP-E5 parity envelope:

* ``elastic-fd-leapfrog``       -- :class:`mixle_pde.elastic.ElasticWave3D`, a 3D isotropic velocity-stress
  staggered-leapfrog elastic-wave stepper.
* ``helmholtz-pml-fd``          -- :func:`mixle_pde.helmholtz_pml.solve_helmholtz_pml`, a frequency-domain
  complex Helmholtz solve with a perfectly-matched-layer absorbing boundary.
* ``groundwater-fd-transport``  -- :class:`mixle_pde.groundwater.GroundwaterTransportOperator`, reactive
  solute transport in a spatially-varying Darcy-flow velocity field.
* ``fem-p1-simplex``            -- :mod:`mixle_pde.fem`'s P1 simplex Poisson/diffusion assembler.
* ``wave-fd-leapfrog``          -- :class:`mixle_pde.wave.WaveEquation2D`, an explicit FD leapfrog wave stepper.
* ``flow-fd-streamfunction``    -- :class:`mixle_pde.flow.NavierStokes2D`, an FD streamfunction-vorticity flow solver.
* ``em-fdtd-yee``               -- :class:`mixle_pde.maxwell.Maxwell3D`, a Yee-grid FDTD Maxwell solver.
* ``transport-fd-advdiff``      -- :class:`mixle_pde.dynamics.AdvectionDiffusionOperator`, a method-of-lines
  advection-diffusion (transport) operator.

Registration only *declares* what each backend supports; it never widens what a backend actually does.
:func:`run_math_problem` always calls :func:`mixle_pde.problem_adapter.require_compatible` first, so a
``CON-MATH-PROBLEM-V1`` problem that asks for an operator kind, discretization, mesh cell type, or evidence
kind a profile does not declare is rejected with :class:`~mixle_pde.problem_adapter.UnsupportedPDEProblem`
rather than being silently solved by the nearest available kernel.

Per the project's "no live callables in canonical/serializable artifacts" rule, :class:`PDEKernelRegistration`
never stores a Python callable directly -- it stores an ``invoke_key`` string that indexes a private,
module-level, registered invocation table (the same "register, don't branch" pattern
:mod:`mixle_pde.dynamics` already uses for its operator factories).

A completed invocation here means the discretized linear algebra / time-stepping ran and produced finite,
internally-consistent output (a small residual, a bounded/finite time series, a preserved discrete
invariant); it is evidence about numerical solve correctness, not verified physical evidence -- see the
``evidence`` payload's docstrings on each ``_invoke_*`` function for exactly what is checked.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from mixle_pde.problem_adapter import (
    PDEBackendProfile,
    PDECompatibilityReport,
    require_compatible,
)

__all__ = [
    "PDEPort",
    "PDEKernelArtifact",
    "PDEKernelRegistration",
    "PDEStudyResult",
    "get_kernel_registration",
    "list_kernel_registrations",
    "run_math_problem",
]


@dataclass(frozen=True)
class PDEPort:
    """A typed input or output port a backend's invocation exposes.

    ``role`` is ``"input"`` or ``"output"``; ``kind`` names the port's semantic category (e.g.
    ``"coefficient_field"``, ``"point_source"``, ``"scalar_field"``); ``units`` is an SI (or explicitly
    dimensionless, ``"1"``) unit string. Plain data only -- no callables -- so a profile's port list stays
    safe to hash/serialize alongside the rest of a compatibility record.
    """

    id: str
    role: str
    kind: str
    units: str


@dataclass(frozen=True)
class PDEKernelArtifact:
    """Declares the shape of the artifact record a kernel invocation returns."""

    kind: str
    units: str
    description: str


@dataclass(frozen=True)
class PDEKernelRegistration:
    """A concrete legacy kernel wired behind a :class:`~mixle_pde.problem_adapter.PDEBackendProfile`.

    ``invoke_key`` is a string key into the private ``_INVOKERS`` table, never a live callable -- keeps this
    dataclass safe to treat as a canonical, serializable registration record (constraint: no arbitrary
    callbacks in hashed/serialized artifacts).
    """

    profile: PDEBackendProfile
    ports: tuple[PDEPort, ...]
    artifact: PDEKernelArtifact
    invoke_key: str
    source: str


@dataclass(frozen=True)
class PDEStudyResult:
    """The outcome of running a ``CON-MATH-PROBLEM-V1`` study through a registered backend."""

    backend_id: str
    compatibility_report: PDECompatibilityReport
    solution: Any
    evidence: Mapping[str, Any]


# ---------------------------------------------------------------------------
# Invocation registry ("register, don't branch" -- no callables stored on the dataclasses above)
# ---------------------------------------------------------------------------
_INVOKERS: dict[str, Callable[[Mapping[str, Any]], Mapping[str, Any]]] = {}


def _register_invoker(key: str) -> Callable[[Callable], Callable]:
    def _decorator(fn: Callable) -> Callable:
        _INVOKERS[key] = fn
        return fn

    return _decorator


def _solve_params(problem: Mapping[str, Any]) -> dict[str, Any]:
    """Extract the backend-specific keyword parameters carried by a study's ``solve_plan``.

    ``solve_plan`` is the CON-MATH-PROBLEM-V1 field a concrete study uses to hand a selected backend its
    numeric parameters (mesh/grid sizes, coefficients, time-stepping controls); the mathematical-problem
    shape itself (domains/unknowns/operators/objectives) stays backend-neutral.
    """
    plan = problem.get("solve_plan") or {}
    params = plan.get("parameters") or {}
    if not isinstance(params, Mapping):
        raise ValueError("solve_plan.parameters must be a mapping of backend keyword parameters.")
    return dict(params)


# ---------------------------------------------------------------------------
# elastic-fd-leapfrog -- mixle_pde.elastic.ElasticWave3D (3D velocity-stress staggered leapfrog)
# ---------------------------------------------------------------------------
_ELASTIC_PROFILE = PDEBackendProfile(
    id="elastic-fd-leapfrog",
    operator_kinds=frozenset({"time_stepping"}),
    discretizations=frozenset({"FD-staggered-leapfrog"}),
    objective_senses=frozenset({"satisfy"}),
    mesh_cell_types=frozenset({"structured_grid"}),
    evidence_kinds=frozenset({"convergence"}),
)


@_register_invoker("elastic-fd-leapfrog")
def _invoke_elastic_fd(problem: Mapping[str, Any]) -> dict[str, Any]:
    """Step the 3D isotropic velocity-stress elastic wave equation an explicit staggered leapfrog for ``n_steps``.

    A point body-force source excites the medium at the first step; the optional absorbing sponge
    (``absorb_width``/``absorb_strength`` -- the only absorbing-boundary option
    :class:`~mixle_pde.elastic.ElasticWave3D` implements, there is no PML here) and free surface are passed
    straight through when requested, never added to.

    Evidence: ``convergence`` reports whether the nine-component state stayed finite and the achieved Courant
    number ``vp_max * dt / spacing`` against the 3-D leapfrog stability limit ``1/sqrt(3)`` -- a numerical-
    stability check mirroring ``wave-fd-leapfrog``'s own convergence evidence, not a physical validation.
    """
    import torch

    from mixle_pde.elastic import ElasticWave3D
    from mixle_pde.ops import make_ops

    params = _solve_params(problem)
    n = int(params.get("grid_size", 10))
    dt = float(params.get("dt", 0.01))
    steps = int(params.get("n_steps", 15))
    vp = params.get("vp", 1.7)
    vs = params.get("vs", 1.0)
    rho = params.get("rho", 1.0)
    amplitude = float(params.get("amplitude", 1.0))
    source_index = tuple(params.get("source_index", (n // 2, n // 2, n // 2)))

    elastic = ElasticWave3D(
        n,
        dt=dt,
        spacing=params.get("spacing"),
        vp=vp,
        vs=vs,
        rho=rho,
        absorb_width=int(params.get("absorb_width", 0)),
        absorb_strength=float(params.get("absorb_strength", 2.0)),
        free_surface=bool(params.get("free_surface", False)),
    )
    ops = make_ops()
    state = elastic.zeros(ops)
    source = elastic.point_force_source(source_index, (0.0, 0.0, amplitude))

    for step in range(steps):
        state = elastic.step(state, ops, source=source if step == 0 else None)

    finite = bool(torch.isfinite(state).all().item())
    courant = float(elastic.vp_max * dt / elastic.h)
    courant_limit = 1.0 / np.sqrt(3.0)

    return {
        "solution": state.detach().numpy(),
        "evidence": {
            "convergence": {
                "finite": finite,
                "courant_number": courant,
                "courant_limit": courant_limit,
                "stable": bool(finite and courant <= courant_limit + 1.0e-9),
            }
        },
    }


# ---------------------------------------------------------------------------
# helmholtz-pml-fd -- mixle_pde.helmholtz_pml.solve_helmholtz_pml (frequency-domain complex Helmholtz + PML)
# ---------------------------------------------------------------------------
_HELMHOLTZ_PROFILE = PDEBackendProfile(
    id="helmholtz-pml-fd",
    operator_kinds=frozenset({"linear_operator"}),
    discretizations=frozenset({"FD-helmholtz-pml"}),
    objective_senses=frozenset({"satisfy"}),
    mesh_cell_types=frozenset({"structured_grid"}),
    evidence_kinds=frozenset({"residual", "convergence"}),
)


@_register_invoker("helmholtz-pml-fd")
def _invoke_helmholtz_pml(problem: Mapping[str, Any]) -> dict[str, Any]:
    """Solve the 2-D frequency-domain complex Helmholtz equation with a PML absorbing boundary.

    Evidence: ``residual`` independently reassembles the operator and right-hand side (the same public
    :func:`~mixle_pde.helmholtz_pml.helmholtz_pml_operator`/:func:`~mixle_pde.helmholtz_pml.pml_profile` calls
    :func:`~mixle_pde.helmholtz_pml.solve_helmholtz_pml` makes internally, not a shortcut through it) and
    reports the interior-node norm of ``L u - b`` -- a solver-convergence check mirroring ``fem-p1-simplex``'s
    own residual, not a claim that the modeled medium is physically correct; ``convergence`` reports whether
    the complex field stayed finite.
    """
    import torch

    from mixle_pde.helmholtz_pml import helmholtz_pml_operator, pml_profile, solve_helmholtz_pml

    params = _solve_params(problem)
    shape = tuple(int(s) for s in params.get("grid_shape", (16, 16)))
    if len(shape) != 2:
        raise ValueError(f"helmholtz-pml-fd is 2-D only; got grid_shape {shape}.")
    nx, nz = shape
    spacing = params.get("spacing", 1.0)
    omega = float(params.get("omega", 6.0))
    pml_width = params.get("pml_width", 4)
    pml_strength = params.get("pml_strength", 8.0)
    q_factor = params.get("Q")
    slowness2 = float(params.get("slowness2", 1.0))
    source_index = int(params.get("source_index", (nx // 2) * nz + nz // 2))

    m = torch.full((nx * nz,), slowness2, dtype=torch.float64)
    u = solve_helmholtz_pml(
        m,
        shape,
        source_index,
        omega=omega,
        spacing=spacing,
        pml_width=pml_width,
        pml_strength=pml_strength,
        Q=q_factor,
        torch=torch,
    )

    rows, cols, vals, n = helmholtz_pml_operator(
        m,
        shape,
        omega=omega,
        spacing=spacing,
        pml_width=pml_width,
        pml_strength=pml_strength,
        Q=q_factor,
        torch=torch,
    )
    b = torch.zeros(n, dtype=torch.complex128)
    b[source_index] = 1.0
    d = pml_profile(shape, spacing, pml_width, pml_strength, torch=torch)
    j = torch.tensor(1j, dtype=torch.complex128)
    sx = 1.0 + j * d[0].to(torch.complex128) / omega
    sz = 1.0 + j * d[1].to(torch.complex128) / omega
    b = b * (sx * sz)
    lu = torch.zeros(n, dtype=torch.complex128).index_add(0, rows, vals * u[cols])

    idx = np.arange(n).reshape(nx, nz)
    on_boundary = np.zeros((nx, nz), dtype=bool)
    on_boundary[0, :] = on_boundary[-1, :] = True
    on_boundary[:, 0] = on_boundary[:, -1] = True
    interior = torch.as_tensor(idx[~on_boundary].ravel(), dtype=torch.long)

    finite = bool(torch.isfinite(u.real).all().item() and torch.isfinite(u.imag).all().item())
    if finite and interior.numel():
        residual = float(torch.linalg.norm((lu - b)[interior]).item())
    else:
        residual = float("nan") if finite else float("inf")

    return {
        "solution": u.detach().numpy(),
        "evidence": {
            "residual": residual,
            "convergence": {"finite": finite, "interior_nodes": int(interior.numel()), "n_nodes": int(n)},
        },
    }


# ---------------------------------------------------------------------------
# groundwater-fd-transport -- mixle_pde.groundwater.GroundwaterTransportOperator (Darcy-velocity transport)
# ---------------------------------------------------------------------------
_GROUNDWATER_PROFILE = PDEBackendProfile(
    id="groundwater-fd-transport",
    operator_kinds=frozenset({"linear_operator"}),
    discretizations=frozenset({"FD-implicit", "FD-explicit", "FD-exact"}),
    objective_senses=frozenset({"satisfy"}),
    mesh_cell_types=frozenset({"structured_grid"}),
    evidence_kinds=frozenset({"convergence", "conservation"}),
)


@_register_invoker("groundwater-fd-transport")
def _invoke_groundwater_fd(problem: Mapping[str, Any]) -> dict[str, Any]:
    """Advance reactive solute transport in a spatially-varying Darcy-flow velocity field for ``n_steps``.

    The velocity is the steady Darcy specific-discharge field ``q = -K grad h``
    (:func:`~mixle_pde.groundwater.darcy_velocity`), a per-cell field rather than ``transport-fd-advdiff``'s
    single uniform scalar velocity. Scoped to 1-D/2-D grids only for this registration.

    Evidence: ``convergence`` reports finiteness/boundedness of the transported concentration field;
    ``conservation`` reports total solute mass before/after. Unlike ``transport-fd-advdiff``'s constant-
    velocity case, this is not an exactly-preserved discrete invariant: the advective term is the
    non-conservative ``v . grad(u)`` form, and a non-divergence-free ``v`` (a net volumetric Darcy source)
    can additionally create/destroy mass. The default recharge is a zero-net injection/extraction doublet
    (one source cell, one equal-and-opposite sink cell) specifically so the default accepted case stays
    close to conservative; a caller-supplied net-nonzero ``recharge`` will drift further. Reported as a
    discrete-invariant check either way, not a claim that any observed contaminant plume matches this
    linear model.
    """
    from mixle_pde.groundwater import GroundwaterTransportOperator, darcy_velocity

    params = _solve_params(problem)
    shape = tuple(int(s) for s in params.get("grid_shape", (14, 14)))
    if len(shape) not in (1, 2):
        raise ValueError(f"groundwater-fd-transport supports 1-D or 2-D grids only; got shape {shape}.")
    spacing = params.get("spacing", 1.0)
    boundary = params.get("boundary", "neumann")
    conductivity = float(params.get("hydraulic_conductivity", 1.0))
    recharge = params.get("recharge")
    if recharge is None:
        # zero-net injection/extraction doublet: drives real spatially-varying flow without a net
        # volumetric source, so the discrete mass stays close to (not exactly) conserved by default.
        recharge = np.zeros(shape)
        recharge[tuple(max(1, s // 4) for s in shape)] = 1.0
        recharge[tuple(min(s - 2, 3 * s // 4) for s in shape)] = -1.0
    dispersivity = params.get("dispersivity", 0.05)
    molecular_diffusion = float(params.get("molecular_diffusion", 0.001))
    decay = float(params.get("decay", 0.0))
    retardation = float(params.get("retardation", 1.0))
    dt = float(params.get("dt", 0.01))
    n_steps = int(params.get("n_steps", 20))
    scheme = params.get("scheme", "implicit")

    velocity_field = darcy_velocity(conductivity, recharge, shape, spacing=spacing, bc=boundary)
    operator = GroundwaterTransportOperator(
        velocity_field,
        dispersivity,
        shape,
        retardation=retardation,
        decay=decay,
        spacing=spacing,
        bc=boundary,
        scheme=scheme,
        molecular_diffusion=molecular_diffusion,
    )

    lo, hi = operator.domain_bounds
    center = 0.5 * (lo + hi)
    span = np.maximum(hi - lo, 1.0e-9)
    dist2 = np.sum(((operator.coords - center) / span) ** 2, axis=1)
    default_initial = np.exp(-dist2 / 0.02)
    initial = np.asarray(params.get("initial", default_initial), dtype=float)
    if initial.shape != (operator.n,):
        raise ValueError(f"initial condition must have shape ({operator.n},); got {initial.shape}")

    transition = operator.transition_matrix(dt)
    state = initial.copy()
    for _ in range(n_steps):
        state = transition @ state

    finite = bool(np.isfinite(state).all())
    mass_before = float(np.sum(initial))
    mass_after = float(np.sum(state)) if finite else float("nan")

    return {
        "solution": state,
        "evidence": {
            "convergence": {"finite": finite, "max_abs": float(np.max(np.abs(state))) if finite else float("nan")},
            "conservation": {"mass_before": mass_before, "mass_after": mass_after},
        },
    }


# ---------------------------------------------------------------------------
# fem-p1-simplex -- mixle_pde.fem's P1 simplex assembler (steady Poisson/diffusion)
# ---------------------------------------------------------------------------
_FEM_P1_PROFILE = PDEBackendProfile(
    id="fem-p1-simplex",
    operator_kinds=frozenset({"weak_form"}),
    discretizations=frozenset({"P1"}),
    objective_senses=frozenset({"satisfy"}),
    mesh_cell_types=frozenset({"triangle", "tetrahedron"}),
    evidence_kinds=frozenset({"residual", "convergence"}),
)


@_register_invoker("fem-p1-simplex")
def _invoke_fem_p1(problem: Mapping[str, Any]) -> dict[str, Any]:
    """Assemble and solve ``-div(diffusion grad u) = source`` with P1 elements on a box simplex mesh.

    Evidence: ``residual`` is the interior-node norm of ``stiffness @ u - load`` (the discretized PDE
    residual away from the Dirichlet rows, which are trivially exact by construction) -- a solver-convergence
    check, not a claim that the physics the field represents is correct.
    """
    from mixle_pde.fem import (
        assemble_simplex_load_vector,
        assemble_simplex_stiffness_matrix,
        solve_simplex_poisson,
    )
    from mixle_pde.mesh import box_simplex_mesh

    params = _solve_params(problem)
    shape = tuple(params.get("grid_shape", (6, 6)))
    lengths = tuple(params.get("lengths", (1.0,) * len(shape)))
    diffusion = params.get("diffusion", 1.0)
    source = params.get("source", 1.0)

    mesh = box_simplex_mesh(shape, lengths=lengths)
    solution = solve_simplex_poisson(mesh, source, diffusion=diffusion)
    stiffness = assemble_simplex_stiffness_matrix(mesh, diffusion=diffusion)
    load = assemble_simplex_load_vector(mesh, source)
    boundary = mesh.boundary_nodes()
    interior = np.setdiff1d(np.arange(mesh.n_nodes), boundary)
    residual = stiffness @ solution - load
    residual_norm = float(np.linalg.norm(residual[interior])) if interior.size else 0.0

    return {
        "solution": solution,
        "evidence": {
            "residual": residual_norm,
            "convergence": {"interior_nodes": int(interior.size), "n_nodes": int(mesh.n_nodes)},
        },
    }


# ---------------------------------------------------------------------------
# wave-fd-leapfrog -- mixle_pde.wave.WaveEquation2D (explicit leapfrog FD)
# ---------------------------------------------------------------------------
_WAVE_PROFILE = PDEBackendProfile(
    id="wave-fd-leapfrog",
    operator_kinds=frozenset({"time_stepping"}),
    discretizations=frozenset({"FD-leapfrog"}),
    objective_senses=frozenset({"satisfy"}),
    mesh_cell_types=frozenset({"structured_grid"}),
    evidence_kinds=frozenset({"convergence"}),
)


@_register_invoker("wave-fd-leapfrog")
def _invoke_wave_fd(problem: Mapping[str, Any]) -> dict[str, Any]:
    """Step the 2D acoustic wave equation an explicit leapfrog stepper for ``n_steps``.

    Evidence: ``convergence`` reports whether the displacement/velocity state stayed finite (the CFL-bounded
    explicit scheme did not blow up) -- a numerical-stability check, not a physical validation.
    """
    import torch

    from mixle_pde.ops import make_ops
    from mixle_pde.wave import WaveEquation2D

    params = _solve_params(problem)
    n = int(params.get("grid_size", 24))
    dt = float(params.get("dt", 0.01))
    steps = int(params.get("n_steps", 25))
    wave_speed = float(params.get("wave_speed", 1.0))
    amplitude = float(params.get("amplitude", 1.0))
    source_node = int(params.get("source_node", (n // 2) * n + n // 2))

    wave = WaveEquation2D(
        n,
        dt=dt,
        spacing=params.get("spacing"),
        absorb_width=int(params.get("absorb_width", 0)),
        absorb_strength=float(params.get("absorb_strength", 2.0)),
    )
    ops = make_ops()
    state = wave.pack(torch.zeros(n * n, dtype=torch.float64), torch.zeros(n * n, dtype=torch.float64))
    c2 = wave_speed**2

    for step in range(steps):
        source = ops.zeros(n * n)
        if step == 0:
            source = source.clone()
            source[source_node] = amplitude
        state = wave.step(state, c2, ops, source=source)

    finite = bool(torch.isfinite(state).all().item())
    displacement = wave.displacement(state).detach().numpy()

    return {
        "solution": displacement,
        "evidence": {
            "convergence": {
                "finite": finite,
                "max_abs_displacement": float(np.max(np.abs(displacement))) if finite else float("nan"),
            }
        },
    }


# ---------------------------------------------------------------------------
# flow-fd-streamfunction -- mixle_pde.flow.NavierStokes2D (FD streamfunction-vorticity)
# ---------------------------------------------------------------------------
_FLOW_PROFILE = PDEBackendProfile(
    id="flow-fd-streamfunction",
    operator_kinds=frozenset({"time_stepping"}),
    discretizations=frozenset({"FD-streamfunction-vorticity"}),
    objective_senses=frozenset({"satisfy"}),
    mesh_cell_types=frozenset({"structured_grid"}),
    evidence_kinds=frozenset({"convergence"}),
)


@_register_invoker("flow-fd-streamfunction")
def _invoke_flow_fd(problem: Mapping[str, Any]) -> dict[str, Any]:
    """Step 2D incompressible Navier-Stokes (streamfunction-vorticity) for ``n_steps``.

    Evidence: ``convergence`` reports whether vorticity/velocity stayed finite; the streamfunction
    formulation makes the recovered velocity field divergence-free by construction, so this is a
    solver-stability check, not a claim of physical flow-regime accuracy.
    """
    import torch

    from mixle_pde.flow import NavierStokes2D
    from mixle_pde.ops import make_ops

    params = _solve_params(problem)
    n = int(params.get("grid_size", 24))
    viscosity = float(params.get("viscosity", 0.05))
    dt = float(params.get("dt", 0.01))
    steps = int(params.get("n_steps", 15))
    implicit_diffusion = bool(params.get("implicit_diffusion", False))

    flow = NavierStokes2D(
        n,
        viscosity=viscosity,
        dt=dt,
        spacing=params.get("spacing"),
        implicit_diffusion=implicit_diffusion,
    )
    ops = make_ops()
    xx, yy = np.meshgrid(np.linspace(0.0, 1.0, n), np.linspace(0.0, 1.0, n), indexing="ij")
    center = params.get("vortex_center", (0.5, 0.5))
    width = float(params.get("vortex_width", 0.02))
    omega0 = np.exp(-(((xx - center[0]) ** 2 + (yy - center[1]) ** 2) / width))
    omega = torch.as_tensor(omega0.ravel(), dtype=torch.float64)

    for _ in range(steps):
        omega = flow.step(omega, ops)
    psi = flow.streamfunction(omega, ops)
    u, v = flow.velocity(psi, ops)

    finite = bool(
        torch.isfinite(omega).all().item() and torch.isfinite(u).all().item() and torch.isfinite(v).all().item()
    )

    return {
        "solution": omega.detach().numpy(),
        "evidence": {
            "convergence": {
                "finite": finite,
                "vorticity_norm": float(torch.linalg.norm(omega).item()) if finite else float("nan"),
            }
        },
    }


# ---------------------------------------------------------------------------
# em-fdtd-yee -- mixle_pde.maxwell.Maxwell3D (Yee-grid FDTD)
# ---------------------------------------------------------------------------
_EM_PROFILE = PDEBackendProfile(
    id="em-fdtd-yee",
    operator_kinds=frozenset({"time_stepping"}),
    discretizations=frozenset({"FDTD-yee"}),
    objective_senses=frozenset({"satisfy"}),
    mesh_cell_types=frozenset({"structured_grid"}),
    evidence_kinds=frozenset({"convergence", "conservation"}),
)


@_register_invoker("em-fdtd-yee")
def _invoke_em_fdtd(problem: Mapping[str, Any]) -> dict[str, Any]:
    """Step the 3D source-free Maxwell curl equations on a Yee grid (PEC cavity) for ``n_steps``.

    Evidence: ``convergence`` reports whether the six field components stayed finite; ``conservation``
    reports ``div(mu H)`` before/after stepping -- the Yee scheme keeps this at (near) machine precision by
    construction, so a growth here would flag a stepping/staggering defect, not "the electromagnetics is
    correct" in any experimentally-validated sense.
    """
    import torch

    from mixle_pde.maxwell import Maxwell3D
    from mixle_pde.ops import make_ops

    params = _solve_params(problem)
    n = int(params.get("grid_size", 8))
    eps = float(params.get("eps", 1.0))
    mu = float(params.get("mu", 1.0))
    spacing = float(params.get("spacing", 1.0 / (n - 1)))
    dt = params.get("dt")
    if dt is None:
        c = 1.0 / np.sqrt(eps * mu)
        dt = 0.5 * spacing / (c * np.sqrt(3.0))
    dt = float(dt)
    steps = int(params.get("n_steps", 8))
    amplitude = float(params.get("amplitude", 1.0))

    maxwell = Maxwell3D(n, dt=dt, spacing=spacing, eps=eps, mu=mu)
    ops = make_ops()
    state = maxwell.zeros(ops)
    Ex, Ey, Ez, Hx, Hy, Hz = maxwell.unpack(state)
    Ez = Ez.clone()
    center_index = (n // 2) * n * n + (n // 2) * n + (n // 2)
    Ez[center_index] = amplitude
    state = torch.cat([Ex, Ey, Ez, Hx, Hy, Hz])

    div_before = float(torch.linalg.norm(maxwell.div_H(state, ops)).item())
    for _ in range(steps):
        state = maxwell.step(state, ops)
    div_after = float(torch.linalg.norm(maxwell.div_H(state, ops)).item())
    finite = bool(torch.isfinite(state).all().item())

    return {
        "solution": state.detach().numpy(),
        "evidence": {
            "convergence": {"finite": finite},
            "conservation": {"div_h_before": div_before, "div_h_after": div_after},
        },
    }


# ---------------------------------------------------------------------------
# transport-fd-advdiff -- mixle_pde.dynamics.AdvectionDiffusionOperator (method of lines)
# ---------------------------------------------------------------------------
_TRANSPORT_PROFILE = PDEBackendProfile(
    id="transport-fd-advdiff",
    operator_kinds=frozenset({"linear_operator"}),
    discretizations=frozenset({"FD-implicit", "FD-explicit", "FD-exact"}),
    objective_senses=frozenset({"satisfy"}),
    mesh_cell_types=frozenset({"structured_grid"}),
    evidence_kinds=frozenset({"convergence", "conservation"}),
)


@_register_invoker("transport-fd-advdiff")
def _invoke_transport_fd(problem: Mapping[str, Any]) -> dict[str, Any]:
    """Advance ``du/dt = D d^2u/dx^2 - c du/dx`` by the requested method-of-lines transition scheme.

    Evidence: ``convergence`` reports finiteness/boundedness of the transported field; ``conservation``
    reports total mass before/after (exactly preserved under periodic boundaries for this discretization,
    approximately otherwise) -- a discrete-invariant check, not a claim that any observed transport process
    matches this linear advection-diffusion model.
    """
    from mixle_pde.dynamics import AdvectionDiffusionOperator

    params = _solve_params(problem)
    n = int(params.get("grid_size", 41))
    length = float(params.get("length", 1.0))
    diffusivity = float(params.get("diffusivity", 0.01))
    velocity = float(params.get("velocity", 0.5))
    dt = float(params.get("dt", 0.01))
    n_steps = int(params.get("n_steps", 20))
    boundary = params.get("boundary", "periodic")
    scheme = params.get("scheme", "implicit")

    operator = AdvectionDiffusionOperator(diffusivity, velocity, n, length=length, bc=boundary, scheme=scheme)
    x = operator.grid
    default_initial = np.exp(-((x - length / 2.0) ** 2) / (0.02 * length**2))
    initial = np.asarray(params.get("initial", default_initial), dtype=float)
    if initial.shape != (n,):
        raise ValueError(f"initial condition must have shape ({n},); got {initial.shape}")

    transition = operator.transition_matrix(dt)
    state = initial.copy()
    for _ in range(n_steps):
        state = transition @ state

    finite = bool(np.isfinite(state).all())
    mass_before = float(np.sum(initial))
    mass_after = float(np.sum(state)) if finite else float("nan")

    return {
        "solution": state,
        "evidence": {
            "convergence": {"finite": finite, "max_abs": float(np.max(np.abs(state))) if finite else float("nan")},
            "conservation": {"mass_before": mass_before, "mass_after": mass_after},
        },
    }


# ---------------------------------------------------------------------------
# Registration table
# ---------------------------------------------------------------------------
_REGISTRATIONS: dict[str, PDEKernelRegistration] = {
    "elastic-fd-leapfrog": PDEKernelRegistration(
        profile=_ELASTIC_PROFILE,
        ports=(
            PDEPort(id="vp", role="input", kind="coefficient_field", units="m/s"),
            PDEPort(id="vs", role="input", kind="coefficient_field", units="m/s"),
            PDEPort(id="rho", role="input", kind="coefficient_field", units="kg/m^3"),
            PDEPort(id="source", role="input", kind="point_source", units="1"),
            PDEPort(id="fields", role="output", kind="vector_field", units="mixed(m/s,Pa)"),
        ),
        artifact=PDEKernelArtifact(
            kind="time_series_field",
            units="mixed(m/s,Pa)",
            description=(
                "Packed (vx,vy,vz,sxx,syy,szz,sxy,sxz,syz) staggered-grid state snapshot after n_steps of "
                "the velocity-stress leapfrog."
            ),
        ),
        invoke_key="elastic-fd-leapfrog",
        source="mixle_pde.elastic.ElasticWave3D",
    ),
    "helmholtz-pml-fd": PDEKernelRegistration(
        profile=_HELMHOLTZ_PROFILE,
        ports=(
            PDEPort(id="slowness2", role="input", kind="coefficient_field", units="s^2/m^2"),
            PDEPort(id="source", role="input", kind="point_source", units="1"),
            PDEPort(id="field", role="output", kind="complex_scalar_field", units="1"),
        ),
        artifact=PDEKernelArtifact(
            kind="steady_field_solution",
            units="1",
            description="Complex frequency-domain field solving the PML Helmholtz equation for a point source.",
        ),
        invoke_key="helmholtz-pml-fd",
        source="mixle_pde.helmholtz_pml.solve_helmholtz_pml",
    ),
    "groundwater-fd-transport": PDEKernelRegistration(
        profile=_GROUNDWATER_PROFILE,
        ports=(
            PDEPort(id="hydraulic_conductivity", role="input", kind="coefficient_field", units="m/s"),
            PDEPort(id="darcy_velocity", role="input", kind="vector_field", units="m/s"),
            PDEPort(id="dispersivity", role="input", kind="coefficient", units="m"),
            PDEPort(id="concentration", role="output", kind="scalar_field", units="1"),
        ),
        artifact=PDEKernelArtifact(
            kind="time_series_field",
            units="1",
            description=(
                "Transported solute-concentration field snapshot after n_steps, advected by a "
                "spatially-varying Darcy-flow velocity field."
            ),
        ),
        invoke_key="groundwater-fd-transport",
        source="mixle_pde.groundwater.GroundwaterTransportOperator",
    ),
    "fem-p1-simplex": PDEKernelRegistration(
        profile=_FEM_P1_PROFILE,
        ports=(
            PDEPort(id="source", role="input", kind="field_source", units="1"),
            PDEPort(id="diffusion", role="input", kind="coefficient_field", units="m^2/s"),
            PDEPort(id="solution", role="output", kind="scalar_field", units="1"),
        ),
        artifact=PDEKernelArtifact(
            kind="steady_field_solution",
            units="1",
            description="P1 nodal field solving -div(diffusion grad u) = source with Dirichlet boundaries.",
        ),
        invoke_key="fem-p1-simplex",
        source="mixle_pde.fem.solve_simplex_poisson",
    ),
    "wave-fd-leapfrog": PDEKernelRegistration(
        profile=_WAVE_PROFILE,
        ports=(
            PDEPort(id="wave_speed", role="input", kind="coefficient_field", units="m/s"),
            PDEPort(id="source", role="input", kind="point_source", units="1"),
            PDEPort(id="displacement", role="output", kind="scalar_field", units="m"),
        ),
        artifact=PDEKernelArtifact(
            kind="time_series_field",
            units="m",
            description="Leapfrog acoustic displacement field snapshot after n_steps.",
        ),
        invoke_key="wave-fd-leapfrog",
        source="mixle_pde.wave.WaveEquation2D",
    ),
    "flow-fd-streamfunction": PDEKernelRegistration(
        profile=_FLOW_PROFILE,
        ports=(
            PDEPort(id="viscosity", role="input", kind="coefficient", units="m^2/s"),
            PDEPort(id="vorticity", role="output", kind="scalar_field", units="1/s"),
        ),
        artifact=PDEKernelArtifact(
            kind="time_series_field",
            units="1/s",
            description="Vorticity field snapshot after n_steps of the FD streamfunction-vorticity stepper.",
        ),
        invoke_key="flow-fd-streamfunction",
        source="mixle_pde.flow.NavierStokes2D",
    ),
    "em-fdtd-yee": PDEKernelRegistration(
        profile=_EM_PROFILE,
        ports=(
            PDEPort(id="eps", role="input", kind="coefficient", units="F/m"),
            PDEPort(id="mu", role="input", kind="coefficient", units="H/m"),
            PDEPort(id="fields", role="output", kind="vector_field", units="mixed(V/m,A/m)"),
        ),
        artifact=PDEKernelArtifact(
            kind="time_series_field",
            units="mixed(V/m,A/m)",
            description="Packed (Ex,Ey,Ez,Hx,Hy,Hz) Yee-grid state snapshot after n_steps of FDTD leapfrog.",
        ),
        invoke_key="em-fdtd-yee",
        source="mixle_pde.maxwell.Maxwell3D",
    ),
    "transport-fd-advdiff": PDEKernelRegistration(
        profile=_TRANSPORT_PROFILE,
        ports=(
            PDEPort(id="diffusivity", role="input", kind="coefficient", units="m^2/s"),
            PDEPort(id="velocity", role="input", kind="coefficient", units="m/s"),
            PDEPort(id="concentration", role="output", kind="scalar_field", units="1"),
        ),
        artifact=PDEKernelArtifact(
            kind="time_series_field",
            units="1",
            description="Transported scalar field snapshot after n_steps of the method-of-lines transition.",
        ),
        invoke_key="transport-fd-advdiff",
        source="mixle_pde.dynamics.AdvectionDiffusionOperator",
    ),
}


def list_kernel_registrations() -> tuple[PDEKernelRegistration, ...]:
    """Return every registered legacy-kernel backend, sorted by id."""
    return tuple(_REGISTRATIONS[key] for key in sorted(_REGISTRATIONS))


def get_kernel_registration(backend_id: str) -> PDEKernelRegistration:
    """Look up a registered backend by id."""
    try:
        return _REGISTRATIONS[backend_id]
    except KeyError as exc:
        raise KeyError(f"unknown PDE backend {backend_id!r}; registered: {sorted(_REGISTRATIONS)}") from exc


def run_math_problem(problem: Mapping[str, Any], backend_id: str) -> PDEStudyResult:
    """Run a ``CON-MATH-PROBLEM-V1`` study through a registered backend.

    Always calls :func:`mixle_pde.problem_adapter.require_compatible` first: an unsupported operator kind,
    discretization, mesh cell type, or evidence request raises
    :class:`~mixle_pde.problem_adapter.UnsupportedPDEProblem` rather than being solved by whatever kernel
    happens to be registered (never-silently-downgrade).
    """
    registration = get_kernel_registration(backend_id)
    report = require_compatible(problem, registration.profile)
    invoker = _INVOKERS[registration.invoke_key]
    result = invoker(problem)
    return PDEStudyResult(
        backend_id=registration.profile.id,
        compatibility_report=report,
        solution=result["solution"],
        evidence=result["evidence"],
    )
