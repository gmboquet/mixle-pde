"""Solver knowledge catalog: governing equations, applicability, and benchmark evidence (MP-J2).

Source: notes/mixle-pde-ai-native-multiphysics-work-plan.md section 8, MP-J2 ("Physics, material,
method, and solver knowledge catalog") -- "Provide structured physics/component, material, mesh,
coupling, observation, prior/likelihood, inference-engine, surrogate-method, and solver metadata
with equations/semantics, assumptions, applicability ranges, compatibility, dimensionless-regime
hints, benchmark evidence, and limitations." The M2 reconciliation ledger
(docs/reconciliation/mp-task-ledger.md) records MP-J2 as ``not-started``: "No structured catalog
with equations/applicability/benchmark-evidence metadata found anywhere."

This module is a deliberately narrow baseline slice of that much larger task, matching how the
closest sibling verification modules (mms.py/MP-K3, result_queries.py/MP-K2,
mcmc_diagnostics.py/MP-I8) each scoped down to a first slice of their own task rather than
attempting the full scope in one change: it covers *solver* knowledge only -- one entry per
:mod:`mixle_pde.pde_backend_registry` kernel, the governing equation it solves, its applicability
conditions/limitations, and a pointer to real, on-disk benchmark/verification evidence. Material,
mesh, coupling, observation, prior/likelihood, inference-engine, and surrogate-method metadata
(the rest of MP-J2's named scope) are explicitly not attempted here.

This module imports :mod:`mixle_pde.pde_backend_registry` read-only, purely to cross-check that
every catalog entry names a real, currently-registered backend id -- it never edits that module and
never widens what a backend actually does. As of this catalog's authoring, nine kernels are
registered there (``elastic-fd-leapfrog``, ``helmholtz-pml-fd``, ``groundwater-fd-transport``,
``fem-p1-simplex``, ``wave-fd-leapfrog``, ``flow-fd-streamfunction``, ``em-fdtd-yee``,
``transport-fd-advdiff``, ``flow-spectral-ns``) -- more than either this task's own briefing
snapshot or the M2 ledger's MP-E5 entry record, both of which predate PR #85 (``flow-spectral-ns``)
and PR #90 (the three newest kernels). ``tests/knowledge_catalog_test.py`` is the completeness/
no-drift invariant that keeps this catalog from silently falling behind the registry again:
every catalog entry must name a real registered backend id, every registered backend id must have
a catalog entry, and every cited test file/test name must actually exist on disk.

Methodology and honesty notes
------------------------------
* ``governing_equation`` is a string/LaTeX-ish description transcribed from each kernel's own module
  docstring (cited in ``source``), not a re-derivation and not a symbolic AST -- symbolic
  representation of PDE operators is out of scope for this module.
* ``applicability`` states conditions and limitations *of the registered invocation specifically*
  (grid/mesh type, dimension, boundary-condition support, discretization, what is and is not
  enforced automatically) -- it is deliberately narrower in places than the underlying class's full
  capability, because :mod:`mixle_pde.pde_backend_registry` only wires up one fixed invocation shape
  per kernel. Where the underlying module supports more than the registered invocation exposes, that
  gap is stated explicitly rather than silently inherited.
* ``benchmark_evidence`` cites only tests that exist on disk today, checked by
  ``tests/knowledge_catalog_test.py`` (file existence plus a literal ``def test_name`` search, so a
  renamed or deleted test fails this catalog's own test rather than rotting silently). A completed
  test proves the linear algebra ran and matched its stated reference to its stated tolerance; per
  this repository's standing verification-level convention, that is evidence about numerical/
  discretization correctness, not an independent physical validation.
* No entry here claims support, physical validity, or capability beyond what its cited evidence
  actually checks. Per this program's standing convention, "no unsupported mathematics-superiority
  claim" -- this catalog records what already exists and is already verified, nothing aspirational.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class BenchmarkEvidence:
    """One concrete, citable piece of verification evidence for a catalog entry.

    ``test_file`` is a repo-relative path (e.g. ``"tests/fem_test.py"``); ``test_name`` is the bare
    ``def test_...`` (or ``def test_...`` inside a ``unittest.TestCase``) function name.
    ``check_kind`` names the verification method (e.g. ``"manufactured_solution"``,
    ``"analytic_reference"``, ``"conservation_invariant"``, ``"closed_form_decay"``); ``summary`` is
    a one-line, honest description of exactly what the cited test checks and to what tolerance.
    """

    test_file: str
    test_name: str
    check_kind: str
    summary: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class KnowledgeCatalogEntry:
    """Governing equation, applicability, and benchmark evidence for one registered PDE backend.

    ``backend_id`` must match a real :class:`mixle_pde.problem_adapter.PDEBackendProfile.id`
    currently registered in :mod:`mixle_pde.pde_backend_registry`; ``source`` must match that same
    registration's ``PDEKernelRegistration.source`` string -- both are cross-checked by
    ``tests/knowledge_catalog_test.py`` against the live registry, not merely asserted here.
    """

    backend_id: str
    source: str
    physics_domain: str
    governing_equation: str
    method: str
    applicability: tuple[str, ...]
    benchmark_evidence: tuple[BenchmarkEvidence, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "backend_id": self.backend_id,
            "source": self.source,
            "physics_domain": self.physics_domain,
            "governing_equation": self.governing_equation,
            "method": self.method,
            "applicability": list(self.applicability),
            "benchmark_evidence": [evidence.as_dict() for evidence in self.benchmark_evidence],
        }


KNOWLEDGE_CATALOG: tuple[KnowledgeCatalogEntry, ...] = (
    KnowledgeCatalogEntry(
        backend_id="elastic-fd-leapfrog",
        source="mixle_pde.elastic.ElasticWave3D",
        physics_domain="elastodynamics / seismic wave propagation",
        governing_equation=(
            "rho d^2u/dt^2 = (lambda + mu) grad(div u) + mu laplacian(u) + f, rewritten as the "
            "first-order velocity-stress system (Virieux 1986): dv/dt = (1/rho) div(sigma), "
            "dsigma/dt = C : grad(v), with isotropic stiffness C giving a compressional speed "
            "vp = sqrt((lambda + 2 mu)/rho) and a shear speed vs = sqrt(mu/rho)."
        ),
        method=(
            "explicit 3-D staggered-grid (velocity-stress) leapfrog, second-order centered "
            "differences (Virieux 1986)"
        ),
        applicability=(
            "isotropic elastic medium only under this backend id -- VTI/TTI anisotropy is a "
            "separate, unregistered module (mixle_pde.elastic_aniso.AnisotropicElasticWave3D)",
            "structured regular 3-D grid only (mesh_cell_types={'structured_grid'})",
            "explicit conditionally-stable leapfrog; the registered invocation computes the 3-D "
            "Courant number vp_max*dt/spacing and reports it against the 1/sqrt(3) stability limit "
            "as convergence evidence, it does not reject an unstable caller-supplied dt/grid_size "
            "before stepping",
            "absorbing boundary is a sponge layer only (absorb_width/absorb_strength), not a PML; "
            "free surface is optional and off by default",
        ),
        benchmark_evidence=(
            BenchmarkEvidence(
                test_file="tests/elastic_test.py",
                test_name="test_p_wave_speed",
                check_kind="analytic_reference",
                summary="measured P-wave speed vs. analytic Poisson-solid vp, relative error < 8%",
            ),
            BenchmarkEvidence(
                test_file="tests/elastic_test.py",
                test_name="test_s_wave_speed",
                check_kind="analytic_reference",
                summary="measured S-wave speed vs. analytic Poisson-solid vs, relative error < 8%",
            ),
            BenchmarkEvidence(
                test_file="tests/pde_backend_registry_kernels2_test.py",
                test_name="test_elastic_accepted_study_steps_and_stays_finite_and_stable",
                check_kind="conservation_invariant",
                summary=(
                    "registered-backend study: finite state after n_steps and achieved Courant "
                    "number within the 1/sqrt(3) stability limit -- a stability check, not accuracy"
                ),
            ),
        ),
    ),
    KnowledgeCatalogEntry(
        backend_id="helmholtz-pml-fd",
        source="mixle_pde.helmholtz_pml.solve_helmholtz_pml",
        physics_domain="frequency-domain acoustics / electromagnetics",
        governing_equation=(
            "frequency-domain Helmholtz laplacian(u) + omega^2 m u = -f under complex "
            "coordinate stretching s_x(x) = 1 + i d_x(x)/omega near each boundary (PML), giving the "
            "complex-symmetric divergence form -div(A grad u) - s_x s_z omega^2 m u = s_x s_z f "
            "with A = diag(s_z/s_x, s_x/s_z)."
        ),
        method=(
            "node-centered finite-difference assembly of the stretched-coordinate complex-symmetric "
            "operator on a 2-D structured grid, direct sparse complex linear solve"
        ),
        applicability=(
            "2-D only (the registered invocation raises ValueError for grid_shape with length != 2)",
            "structured regular grid only",
            "scalar, caller-supplied per-node squared-slowness field m only; no anisotropic/tensor "
            "modulus",
            "PML is an approximate absorbing boundary (finite-width discrete layer): reflection is "
            "small, not exactly zero, at finite pml_width/pml_strength",
        ),
        benchmark_evidence=(
            BenchmarkEvidence(
                test_file="tests/helmholtz_pml_test.py",
                test_name="test_greens_function_interior",
                check_kind="analytic_reference",
                summary=(
                    "interior field vs. the analytic Hankel Green's function, median relative "
                    "error < 8%"
                ),
            ),
            BenchmarkEvidence(
                test_file="tests/helmholtz_pml_test.py",
                test_name="test_attenuation_decay_rate",
                check_kind="analytic_reference",
                summary="PML attenuation decay rate vs. the analytic rate, error < 10%",
            ),
            BenchmarkEvidence(
                test_file="tests/pde_backend_registry_kernels2_test.py",
                test_name="test_helmholtz_accepted_study_solves_and_returns_small_residual",
                check_kind="manufactured_solution",
                summary=(
                    "registered-backend study: operator/right-hand-side independently reassembled "
                    "and interior-node residual norm ||L u - b|| < 1e-8"
                ),
            ),
        ),
    ),
    KnowledgeCatalogEntry(
        backend_id="groundwater-fd-transport",
        source="mixle_pde.groundwater.GroundwaterTransportOperator",
        physics_domain="groundwater hydrology / reactive solute transport",
        governing_equation=(
            "retarded reactive advection-dispersion R dc/dt + v . grad(c) = div(D grad c) - "
            "lambda c in a steady Darcy specific-discharge field v = q = -K grad h, where h solves "
            "the steady Darcy-flow Poisson equation div(K grad h) = -recharge."
        ),
        method=(
            "method-of-lines: per-axis upwind advection plus velocity-dependent dispersion "
            "finite-difference operator, coupled to a separate steady finite-difference Poisson "
            "solve for the Darcy head/velocity field"
        ),
        applicability=(
            "1-D or 2-D grids only for this registration (the registered invocation raises "
            "ValueError outside that)",
            "structured regular grid only",
            "mass conservation is approximate, not exact, whenever the Darcy velocity field is not "
            "divergence-free (a net volumetric recharge source/sink) -- unlike "
            "transport-fd-advdiff's constant-velocity case, which conserves mass exactly under "
            "periodic boundaries",
            "the registered invocation's default recharge is a zero-net injection/extraction "
            "doublet specifically to keep the default accepted study close to conservative; a "
            "caller-supplied net-nonzero recharge drifts further",
        ),
        benchmark_evidence=(
            BenchmarkEvidence(
                test_file="tests/test_groundwater.py",
                test_name="test_pumping_tracer_plume_matches_ogata_banks",
                check_kind="analytic_reference",
                summary=(
                    "1-D pumping-test concentration profile vs. the classic Ogata-Banks analytic "
                    "continuous-injection solution, rtol=5e-2"
                ),
            ),
            BenchmarkEvidence(
                test_file="tests/pde_backend_registry_kernels2_test.py",
                test_name="test_groundwater_accepted_study_steps_and_stays_finite_with_near_conserved_mass",
                check_kind="conservation_invariant",
                summary=(
                    "registered-backend study: finite state after n_steps, mass conserved to "
                    "rel=0.05 under the default near-divergence-free recharge doublet"
                ),
            ),
        ),
    ),
    KnowledgeCatalogEntry(
        backend_id="fem-p1-simplex",
        source="mixle_pde.fem.solve_simplex_poisson",
        physics_domain="steady diffusion / potential theory",
        governing_equation="-div(kappa grad u) = f (steady linear diffusion/Poisson equation) with Dirichlet boundaries.",
        method=(
            "P1 (linear) simplex finite elements, assembled per-triangle/per-tetrahedron on an "
            "unstructured simplex mesh, direct sparse solve"
        ),
        applicability=(
            "Dirichlet boundaries only (mixle_pde.fem itself implements no other boundary kind)",
            "triangle or tetrahedron mesh cells only under this registration -- mixle_pde.mesh also "
            "supports a 4-D simplex extension not exercised by this invocation",
            "scalar diffusion coefficient and source only in the registered invocation (kappa, f "
            "passed through as scalars/arrays), not a tensor-valued kappa",
        ),
        benchmark_evidence=(
            BenchmarkEvidence(
                test_file="tests/fem_test.py",
                test_name="test_manufactured_solution_converges",
                check_kind="manufactured_solution",
                summary="P1 solve vs. a manufactured exact solution, checked for convergence under refinement",
            ),
            BenchmarkEvidence(
                test_file="tests/pde_backend_registry_test.py",
                test_name="test_fem_p1_accepted_study_solves_and_returns_small_residual",
                check_kind="manufactured_solution",
                summary=(
                    "registered-backend study: discretized-PDE interior-node residual "
                    "||stiffness @ u - load|| < 1e-8"
                ),
            ),
        ),
    ),
    KnowledgeCatalogEntry(
        backend_id="wave-fd-leapfrog",
        source="mixle_pde.wave.WaveEquation2D",
        physics_domain="acoustics / elastic-wave-adjacent scalar wave propagation",
        governing_equation=(
            "u_tt = c(x)^2 laplacian(u) + source(t) (2-D scalar acoustic wave equation), integrated "
            "as the first-order system (u, w = u_t) by an explicit symplectic leapfrog."
        ),
        method=(
            "explicit second-order centered-difference leapfrog on a 2-D structured grid, with an "
            "absorbing sponge layer (not a PML) near the boundary"
        ),
        applicability=(
            "2-D only under this backend id; no 3-D acoustic kernel is registered",
            "structured regular grid only",
            "explicit conditionally-stable scheme; the registered invocation reports only "
            "finiteness of the state after n_steps, it does not compute or enforce a CFL bound "
            "itself",
            "sponge absorbing boundary only, not an exact or PML radiation condition",
        ),
        benchmark_evidence=(
            BenchmarkEvidence(
                test_file="tests/wave_test.py",
                test_name="test_recovers_velocity_perturbation",
                check_kind="inverse_recovery",
                summary=(
                    "full-waveform-inversion recovery of a synthetic velocity-field amplitude from "
                    "noisy receiver waveforms, fit by Gauss-Newton (module-level, not "
                    "registry-invocation-level, evidence)"
                ),
            ),
            BenchmarkEvidence(
                test_file="tests/pde_backend_registry_test.py",
                test_name="test_wave_accepted_study_steps_and_stays_finite",
                check_kind="conservation_invariant",
                summary=(
                    "registered-backend study: finite displacement/velocity state after n_steps -- "
                    "a numerical-stability check only, explicitly not an accuracy claim per the "
                    "invocation's own evidence docstring"
                ),
            ),
        ),
    ),
    KnowledgeCatalogEntry(
        backend_id="flow-fd-streamfunction",
        source="mixle_pde.flow.NavierStokes2D",
        physics_domain="incompressible fluid dynamics",
        governing_equation=(
            "2-D incompressible Navier-Stokes in streamfunction-vorticity form: "
            "d(omega)/dt + (u . grad) omega = nu laplacian(omega), laplacian(psi) = -omega, "
            "with u = d(psi)/dy, v = -d(psi)/dx (automatically divergence-free by construction)."
        ),
        method=(
            "explicit or implicit-diffusion vorticity time-stepping plus a streamfunction Poisson "
            "solve each step"
        ),
        applicability=(
            "no-slip wall boundary conditions only under this registration "
            "(dirichlet_no_slip_walls)",
            "structured regular Cartesian grid only",
            "2-D only under this backend id -- mixle_pde.flow3d.NavierStokes3D is a separate, "
            "unregistered 3-D Chorin-projection solver",
            "streamfunction-vorticity avoids the pressure-velocity saddle point but does not "
            "expose an explicit pressure field",
        ),
        benchmark_evidence=(
            BenchmarkEvidence(
                test_file="tests/flow_test.py",
                test_name="test_recovers_upstream_amplitude",
                check_kind="inverse_recovery",
                summary=(
                    "Gauss-Newton recovery of a synthetic upstream-flow amplitude from noisy "
                    "sensor velocities, recovered value within 2 posterior standard deviations of "
                    "truth (module-level, not registry-invocation-level, evidence)"
                ),
            ),
            BenchmarkEvidence(
                test_file="tests/pde_backend_registry_test.py",
                test_name="test_flow_accepted_study_steps_and_stays_finite",
                check_kind="conservation_invariant",
                summary=(
                    "registered-backend study: finite vorticity/velocity state after n_steps -- a "
                    "solver-stability check; the divergence-free velocity field is a structural "
                    "property of the streamfunction formulation, not a separately verified claim "
                    "of physical flow-regime accuracy"
                ),
            ),
        ),
    ),
    KnowledgeCatalogEntry(
        backend_id="em-fdtd-yee",
        source="mixle_pde.maxwell.Maxwell3D",
        physics_domain="electromagnetics",
        governing_equation=(
            "source-free Maxwell curl equations dH/dt = -(1/mu) curl E (Faraday), "
            "dE/dt = (1/eps) curl H (Ampere, source-free)."
        ),
        method="Yee (1966) staggered-grid explicit leapfrog FDTD, E and H staggered a half time-step and half cell apart",
        applicability=(
            "PEC (perfect-electric-conductor) cavity walls only under this registration "
            "(tangential E = 0); no absorbing/PML boundary is registered for this id",
            "structured regular 3-D grid only",
            "explicit conditionally-stable scheme; stability requires "
            "dt <= spacing / (c * sqrt(3)) with c = 1/sqrt(eps*mu) (module docstring); the "
            "registered invocation computes a safe default dt from this formula when the caller "
            "omits dt, but does not reject a caller-supplied dt that violates it",
            "scalar (non-dispersive, isotropic) eps/mu only",
        ),
        benchmark_evidence=(
            BenchmarkEvidence(
                test_file="tests/maxwell_test.py",
                test_name="test_div_H_stays_zero",
                check_kind="conservation_invariant",
                summary="div(mu H) stays at machine precision under stepping for a divergence-free initial H",
            ),
            BenchmarkEvidence(
                test_file="tests/maxwell_test.py",
                test_name="test_tm110_resonant_frequency",
                check_kind="analytic_reference",
                summary="PEC-box cavity TM_110 mode ringing frequency vs. the analytic resonant frequency",
            ),
            BenchmarkEvidence(
                test_file="tests/pde_backend_registry_test.py",
                test_name="test_em_accepted_study_steps_and_preserves_div_h",
                check_kind="conservation_invariant",
                summary="registered-backend study: div(mu H) after stepping < 1e-8",
            ),
        ),
    ),
    KnowledgeCatalogEntry(
        backend_id="transport-fd-advdiff",
        source="mixle_pde.dynamics.AdvectionDiffusionOperator",
        physics_domain="scalar advection-diffusion transport",
        governing_equation="du/dt = D d^2u/dx^2 - c du/dx (1-D linear advection-diffusion).",
        method=(
            "method-of-lines: Laplacian (diffusion) + upwind-gradient (advection) "
            "finite-difference spatial operator, turned into a linear state-transition matrix per "
            "the requested scheme (FD-implicit/FD-explicit/FD-exact)"
        ),
        applicability=(
            "1-D only",
            "structured regular grid only",
            "scalar, spatially uniform velocity and diffusivity only -- contrast "
            "groundwater-fd-transport's spatially-varying Darcy velocity field",
            "mass is exactly conserved under periodic boundaries for this discretization, only "
            "approximately otherwise (dirichlet/neumann)",
        ),
        benchmark_evidence=(
            BenchmarkEvidence(
                test_file="tests/test_amd.py",
                test_name="test_amd_column_matches_first_order_benchmark",
                check_kind="analytic_reference",
                summary=(
                    "AdvectionDiffusionOperator wrapped with reactive source terms "
                    "(ReactiveTransport) vs. a first-order-kinetics analytic steady-state column "
                    "profile, rtol=1e-1 -- evidence for the composed reactive-transport wrapper, "
                    "not pure advection-diffusion in isolation"
                ),
            ),
            BenchmarkEvidence(
                test_file="tests/pde_backend_registry_test.py",
                test_name="test_transport_accepted_study_conserves_mass_under_periodic_bc",
                check_kind="conservation_invariant",
                summary="registered-backend study: mass_after == mass_before to rel=1e-6 under periodic boundaries",
            ),
        ),
    ),
    KnowledgeCatalogEntry(
        backend_id="flow-spectral-ns",
        source="mixle_pde.spectral_flow.incompressible_ns_spectral",
        physics_domain="incompressible fluid dynamics (spectral)",
        governing_equation=(
            "periodic-box incompressible Navier-Stokes du/dt + (u . grad) u = -grad p + "
            "nu laplacian(u), div u = 0."
        ),
        method=(
            "Fourier pseudo-spectral spatial discretization (exact incompressibility via "
            "longitudinal-mode projection, exact viscous multiplier -nu|k|^2, pseudo-spectral "
            "evaluation of the nonlinear advection term) with classical RK4 time-stepping; optional "
            "2/3-rule dealiasing"
        ),
        applicability=(
            "periodic boundary conditions only, by construction of the Fourier basis",
            "2-D or 3-D only (dim in {2, 3}); the registered invocation raises ValueError otherwise",
            "DNS core only under this registration -- the underlying module's optional Smagorinsky "
            "LES closure parameter is not exposed through the registered invocation's solve_plan",
        ),
        benchmark_evidence=(
            BenchmarkEvidence(
                test_file="tests/spectral_flow_test.py",
                test_name="test_2d_taylor_green_exact_decay",
                check_kind="closed_form_decay",
                summary="evolved 2-D field vs. the exact Taylor-Green vortex decay, to near machine precision",
            ),
            BenchmarkEvidence(
                test_file="tests/spectral_flow_test.py",
                test_name="test_3d_abc_beltrami_exact_decay",
                check_kind="closed_form_decay",
                summary="evolved 3-D field vs. the exact ABC/Beltrami flow decay, to near machine precision",
            ),
            BenchmarkEvidence(
                test_file="tests/pde_backend_registry_spectral_test.py",
                test_name="test_spectral_ns_accepted_study_matches_exact_taylor_green_decay",
                check_kind="closed_form_decay",
                summary="registered-backend study: max pointwise deviation from the exact Taylor-Green decay < 1e-8",
            ),
        ),
    ),
)


_BY_BACKEND_ID: dict[str, KnowledgeCatalogEntry] = {entry.backend_id: entry for entry in KNOWLEDGE_CATALOG}

if len(_BY_BACKEND_ID) != len(KNOWLEDGE_CATALOG):
    raise AssertionError("duplicate backend_id entries in KNOWLEDGE_CATALOG")


def get_entry(backend_id: str) -> KnowledgeCatalogEntry:
    """Look up a catalog entry by its :mod:`mixle_pde.pde_backend_registry` backend id."""
    try:
        return _BY_BACKEND_ID[backend_id]
    except KeyError as exc:
        raise KeyError(f"no knowledge_catalog entry for backend id {backend_id!r}") from exc


def catalog_backend_ids() -> tuple[str, ...]:
    """Every backend id this catalog covers, in registry order."""
    return tuple(entry.backend_id for entry in KNOWLEDGE_CATALOG)


def list_entries() -> tuple[KnowledgeCatalogEntry, ...]:
    """Every registered catalog entry."""
    return KNOWLEDGE_CATALOG


def entries_by_physics_domain() -> dict[str, tuple[KnowledgeCatalogEntry, ...]]:
    """Group catalog entries by their ``physics_domain`` label."""
    grouped: dict[str, list[KnowledgeCatalogEntry]] = {}
    for entry in KNOWLEDGE_CATALOG:
        grouped.setdefault(entry.physics_domain, []).append(entry)
    return {domain: tuple(entries) for domain, entries in grouped.items()}


def to_catalog_matrix() -> dict[str, Any]:
    """Machine-readable dict form of the full catalog."""
    return {
        "schema_version": 1,
        "source": "mixle_pde.verification.knowledge_catalog.KNOWLEDGE_CATALOG",
        "entries": [entry.as_dict() for entry in KNOWLEDGE_CATALOG],
    }
