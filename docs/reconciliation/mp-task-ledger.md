---
id: DOC-RECON-MP-LEDGER
schema_version: 1.0.0
document_version: 1.0.0
status: active
owner_project: PRJ-PDE
effective_at: 2026-07-16T00:00:00Z
---

# MP task reconciliation ledger (M2)

Maps every task id in `notes/mixle-pde-ai-native-multiphysics-work-plan.md` §8 ("Executable workstreams",
MP-A1 through MP-N8) to an owner or disposition under
`notes/mixle-physics-simulation-discrete-execution-work-plan.md`, per that plan's task **M2**: *"Reconcile
every task in the current PDE work plan to an owner/task here. Accept: No orphaned capability and no
duplicate source of truth."*

## Method

For each task this ledger records one of four statuses:

- **owned-elsewhere** — a specific repo/module implements the equivalent capability today, on a branch that
  is actually merged into that repo's `release/0.8.0` (or `main`). Where coverage is real but incomplete
  relative to the task's full description, the status reads `owned-elsewhere (partial)` and the evidence
  states exactly what is and is not covered.
- **superseded** — the newer architecture plan (§2.2, §3, or a named workstream) explicitly redefines or
  drops the task; the reason is stated, and — where a successor capability exists — it is cited too.
- **not-started** — no *landed* implementation evidence was found in any of the four surveyed repos.
- **unknown** — evidence was inconclusive within this survey's scope; the note states what would need
  checking to resolve it.

**An open, unmerged PR is never counted as `owned-elsewhere` on its own.** Several of the richest pieces of
evidence below (in mixle-physics, mixle-sim, mixle-discrete, and mixle-pde alike) exist only on open PR
branches as of 2026-07-16. Those are called out explicitly as *in-flight* under `not-started`, because
nothing has landed on the tracked release branch yet — this ledger reflects the actual state of
`release/0.8.0` in each repo, not work in review.

Ground truth was gathered by reading source, `git log --all --oneline`, and `gh pr list --state all` in:

- `mixle-pde` (this repo — surveyed directly by this task);
- `mixle-physics` (surveyed by an independent read-only agent pass);
- `mixle-sim` (surveyed by an independent read-only agent pass); and
- `mixle-discrete` (surveyed by an independent read-only agent pass, for generic mathematical/executable
  capabilities the PDE bridge tasks could adapt to rather than reimplement).

`mixle_pde/ownership.py` (`migration_inventory()`) independently classifies every `mixle_pde` module's final
target project (`PRJ-SIM`, `PRJ-CORE`, `PRJ-DATA`, `PRJ-INQUIRY`, or `PRJ-PDE`) and disposition
(`preserve`/`adapt`/`migrate`/`reference`/`retire`); this ledger cites that classification as corroborating
evidence but does not treat a `migrate`/`adapt` disposition as equivalent to a completed migration — the
module still lives in `mixle_pde` today unless a landed commit in the target repo is also cited.

## Summary counts

| Status | Count |
|---|---:|
| owned-elsewhere (incl. partial) | 60 |
| superseded | 3 |
| not-started | 34 |
| unknown | 0 |
| **Total** | **97** |

## MP-A — Baseline, contracts, and backend decisions

| ID | Description | Status | Owner | Evidence |
|---|---|---|---|---|
| MP-A1 | Freeze the current capability/limitation inventory | owned-elsewhere | mixle-pde | `mixle_pde/verification/capability_inventory.py` + `docs/capability-matrix.json`, PR #68 (`b9c56aa`, merged). Import-sweep registry test `tests/capability_inventory_test.py`. |
| MP-A2 | Establish the parity benchmark corpus (elliptic/parabolic/hyperbolic/mixed-saddle-point/H(div)/H(curl)/nonlinear/eigen/moving-domain/multiphysics/Bayesian/UQ/surrogate) | owned-elsewhere (partial) | mixle-sim | `mixle_sim/benchmarks.py` — pinned discretization parity benchmark corpus with `get_case()`/`cases_for_family()`/`build_benchmark_problem()`/`certify()`, covering the elliptic/parabolic/hyperbolic/mixed-saddle-point/H(div)/H(curl) **FEM subset**. PR #2 "SIM-A2: discretization parity benchmark corpus (FEM subset)" (`11bbd81`, merged). Nonlinear, eigen, moving-domain, multiphysics, Bayesian-inverse, UQ, and surrogate benchmark families are not yet covered anywhere. |
| MP-A3 | Freeze JSON Schemas and Python protocols (`ModelSpec`, `model-delta`, `inverse-study`, `posterior-artifact`, `surrogate-spec`, `run-artifact`, `verification-report`) | superseded | — | Newer plan §2.2: *"`ModelSpec` is not frozen as the public product surface."* Artifact identity/hash/extension protocol is reassigned to shared workstream B1 (`PhysicalTheoryArtifact`/canonical envelope). No repo has any of the seven named schemas verbatim; the successor artifacts are the per-repo typed nodes cited under MP-B1/B5 below. |
| MP-A4 | Backend proof spikes and selection ADRs (Gmsh→mesh IR, DOLFINx, FFCx, PETSc, MPI, checkpoint/restart, Torch adjoint, preCICE, OpenFOAM) | not-started | — | No ADR or spike code for Gmsh/DOLFINx/FFCx/PETSc/preCICE/OpenFOAM in mixle-pde, mixle-sim, or mixle-discrete (confirmed by direct grep in the mixle-sim survey: zero hits for "gmsh"/"opencascade"/"petsc"). `mixle_pde/adjoint.py` gives a working Torch-adjoint *pattern* but is not framed as this task's spike/ADR. `mixle-discrete` PR #5 "DISC-D0.9: backend adapter ADR template + worked FLINT example" (merged, `349d5b7`) establishes an ADR *template* discrete-side, but no PDE-relevant backend (Gmsh/DOLFINx/PETSc/preCICE/OpenFOAM) has been spiked against it. |
| MP-A5 | Dependency, licensing, deployment envelope (extras, zero-extras base install) | owned-elsewhere | mixle-pde | `pyproject.toml` extras (`fem`, `mesh`, `mpi`, `fvm`, `coupling`, `inverse`, `surrogate`, `all`), PR #69 (`d3580f7`, merged), verified by `tests/packaging_test.py`. |

## MP-B — Model IR, expressions, units, and semantic validation

mixle-physics carries two parallel data models: `theory.py` (legacy compatibility surface) and `semantics.py`
(the canonical current model, per that repo's own `docs/architecture.md`). Both are cited below where they
independently satisfy a task.

| ID | Description | Status | Owner | Evidence |
|---|---|---|---|---|
| MP-B1 | `ModelSpec` and symbol tables (typed nodes, stable names, references, selections) | superseded | mixle-physics (successor capability) | §2.2 rejects a single frozen `ModelSpec`. Successor: `mixle_physics/theory.py::PhysicalTheory`/`PhysicalDomain`/`Quantity`/`Parameter` (commit `4c0c936`, merged) and `mixle_physics/semantics.py::TheoryArtifact`/`DomainSpec`/`FieldSpec` (commit `c67879f`, PR #2, merged) are typed, cross-referenced model nodes — but `revisions.py`'s own docstring disclaims any unified cross-repo `ModelSpec`/`ModelDelta` product surface, confirming this is a redesign, not a rename. |
| MP-B2 | Safe tensor-expression AST | owned-elsewhere | mixle-physics | `mixle_physics/expr.py` — `Literal`/`Symbol`/`Add`/`Multiply`/`Subtract`/`Divide`/`Negate`/`Power`/`Contract`/`Gradient`/`Divergence`/`Curl`/`TimeDerivative`/`Call`/`Equation`/`parse_expr`. Commit `06245ae`, PR #5 "PHYS-B2" (merged). `PhysicalLaw.equation` is now typed as `Equation`. |
| MP-B3 | Dimensions, coordinate frames, units | owned-elsewhere | mixle-physics | `theory.py::Unit`/`CoordinateFrame`/`ScalingRecord`/`UnitRelation`/`MeasureKind` (commit `bd1fbf8`, PR #4, merged) plus a richer parallel model in `semantics.py::Dimension` (7-exponent SI vector)/`UnitSpec`/`FrameSpec` (commit `c67879f`, merged). |
| MP-B6 | Hierarchical systems, components, namespaces, reuse | owned-elsewhere (partial) | mixle-physics | `semantics.py::ComponentSpec` groups `domain_ids`/`port_ids`/`law_ids` under one id; `catalog.py::thermoelastic_theory()` demonstrates whole-theory composition (unions terms/laws of `heterogeneous_heat_theory()` + `elasticity_theory()`), commit `c67879f`, merged. No explicit nested-namespace hierarchy was found (repo-wide grep for "namespace"/"hierarch" returns nothing) — reuse-by-composition exists, multi-level namespacing does not. |
| MP-B7 | Three-level problem authoring (templates / coefficient-general PDE / raw weak forms) + schema introspection | not-started | — | Confirmed absent in both mixle-physics (repo-wide grep for template/coefficient-general/weak-form authoring returns nothing; `FormulationOffer` only tags an intended formulation kind, it does not derive one) and mixle-pde (single-level direct kernel APIs only). |
| MP-B8 | Parameters, functions, material datasets, coordinate data | owned-elsewhere (partial) | mixle-physics | `theory.py::Parameter` (fixed/free/prior role, bounds, measure, frame) and `semantics.py::PropertySpec`/`MaterialSpec`/`FieldSpec`, commit `c67879f`/`bd1fbf8`/`4c0c936`, merged. `CoreValueRef` deliberately references an external Core value/prior digest rather than duplicating it (`docs/contracts.md`) — by design, not a gap. No literal tabulated/image/voxel material-dataset evidence found; module is explicitly "mesh-independent." |
| MP-B4 | Semantic and well-posedness validator | owned-elsewhere | mixle-physics | `semantics.py::TheoryArtifact.__post_init__` → `_validate_references()`/`_validate_terms_and_laws()`/`_validate_coupling()`; `TheoryArtifact.analyze()` → `AnalysisReport`/`PhysicsGap`. Commit `c67879f`, PR #2 "add semantic validation kernel" (merged). |
| MP-B5 | Immutable revisions, semantic hashes, deltas | owned-elsewhere (partial) | mixle-physics | `mixle_physics/revisions.py::PatchOp`/`TheoryDelta`/`apply_patch()`/`TheoryRevision`/`RevisionHistory`/`DeltaConflict`/`find_conflict()`/`StaleDeltaError`. Commit `306e125`, PR #3 "MP-B5" (merged). Hashing primitives (`semantic_digest`/`canonical_json`) live in `semantics.py`. Scoped only to `PhysicalTheory` — by its own docstring, not a cross-repo delta format, so narrower than the original task's implied scope but a direct, real successor. |

## MP-C — Geometry and mesh generation

| ID | Description | Status | Owner | Evidence |
|---|---|---|---|---|
| MP-C1 | General mesh IR and topology model | owned-elsewhere | mixle-sim | `mixle_sim/mesh.py` — `Mesh`, `CellBlock`, `CellTopology`, `NodeSet`, `TagSet`, `Partition`, `PeriodicMap`, `AdaptationRecord`, `FacetRecord`, `mesh_from_simplex_arrays`, `mesh_to_dict`/`mesh_from_dict`, `mesh_hash`. PR #3 "SIM-C1" (`6900f8a`, merged), consolidated by PR #6 which retired the older `MeshArtifact2D` in favor of this general IR (`76ac663`, merged). mixle-pde's own `mixle_pde/mesh.py::SimplexMesh` remains a narrower legacy analog (`ownership.py` disposition `adapt`→`PRJ-SIM`, i.e. explicitly not yet migrated/superseded by the sim IR). |
| MP-C2 | Stable named selections and topology queries | owned-elsewhere (partial) | mixle-sim | Merged: `mixle_sim/artifacts.py::NamedSelection`/`SelectionEntity`/`mesh_selection_ids()` (2D mesh-level selections). A richer domain-level predicate/region/boundary/interface selection system exists in `mixle_sim/geometry.py::SelectionKind`/`Predicate`/`region()`/`boundary()`/`interface()` (PR #10, `2750965`) but that PR is **open/unmerged** — not counted as landed. mixle-physics's `semantics.py::RegionSpec` gives the physical-layer *semantic concept* of a named region (merged, `c67879f`) but no topology query engine, matching the newer plan's intended C1/E4 vs. C1(physics) split. |
| MP-C3 | Parametric geometry and CSG DSL | not-started | — | `mixle_sim/geometry.py::HyperRectangle`/`Ball`/`HalfSpace`/`PolylineCurve`/`Union`/`Intersection`/`Difference` is real, working CSG code but exists only on **open PR #10** (`2750965`), not merged into `release/0.8.0`. |
| MP-C4 | CAD and mesh interchange (STEP/IGES/BREP/STL/Gmsh MSH/XDMF/VTK/Exodus) | not-started | — | Zero hits for any of these formats in mixle-sim or mixle-pde. |
| MP-C5 | Gmsh/OpenCASCADE backend | not-started | — | Zero hits for "gmsh"/"opencascade"/"occ" anywhere in mixle-sim; `compiler.py::MeshPlan.generator` is an opaque unimplemented string field. |
| MP-C6 | Mesh quality, repair, deterministic policy | owned-elsewhere (partial) | mixle-sim | `mixle_sim/artifacts.py::MeshQuality`/`mesh_quality_2d()` and `generate_reference_mesh()` give deterministic 2D quality metrics/meshing. `compiler.py::GeometrySpec.repair_policy` is an unimplemented string field ("reject" default) — no actual repair algorithm. |
| MP-C7 | Adaptive refinement, coarsening, field transfer | owned-elsewhere (partial) | mixle-sim | `mixle_sim/artifacts.py::RefinementRelation`/`refinement_relation()`, `mesh.py::AdaptationRecord` (parent→children lineage), `programs.py::TransferPlan`/`TransferMethod` — these are lineage/contract *data structures*, not an error-driven adaptive-refinement engine. No estimator or execution loop found. |
| MP-C8 | Moving meshes, ALE, remeshing, overset decision | owned-elsewhere (partial) | mixle-pde (legacy only) | `mixle_pde/mesh.py::MovingSimplexMesh`/`moving_mesh`/`pipe_radial_deformation` (PR #55, "3D solver scale-up... telescoping meshes"). mixle-sim's `geometry.py` docstring explicitly states remeshing/motion survival is "out of scope here" — confirmed not picked up on the newer-architecture side yet. No geometric-conservation-law check found anywhere. |
| MP-C9 | Mixed-dimensional/embedded/unfitted/image-derived meshes | not-started | — | `mixle_sim/geometry.py::Embedding`/`Network`/`NetworkNode`/`NetworkEdge` (self-described as "minimal, non-BRep mixed-dimensional support") is real but exists only on **open PR #10** (`2750965`), unmerged. Unfitted/image-derived meshing has no evidence anywhere. |
| MP-C10 | Physics-aware automatic mesh planning and preview | not-started | — | `mixle_sim/compiler.py::MeshPlan` is a static plan record (generator/cell_type/target_size) with no physics-driven sizing logic or preview capability. |

## MP-D — Elements, function spaces, weak forms, and assembly

| ID | Description | Status | Owner | Evidence |
|---|---|---|---|---|
| MP-D1 | Typed expression/binder bridge to mathematical protocol | owned-elsewhere (partial) | mixle-sim | `mixle_sim/forms.py::Expression`/`ExpressionOperation` (REFERENCE/LITERAL/ADD/NEGATE/MULTIPLY/GRADIENT/INNER) is a typed, dimension-checked expression tree, PR #4 (`7109e61`, merged) — the sim-side half. The physics-side "binder" that lowers a `PhysicalTheory` into this kind of IR, `mixle_physics/elaborate.py` (`SourceRef`, `MathStatement`, `elaborate()`), exists only on **open PR #8** (`915f059`), unmerged — not counted as landed, so the cross-repo bridge is incomplete even though the sim-side expression IR is real. |
| MP-D2 | Trial/test/coefficient/mixed-space semantics | owned-elsewhere | mixle-sim | `forms.py::SymbolRole` (TRIAL/TEST/COEFFICIENT/NORMAL), `FunctionSpace` (continuity ∈ continuous/discontinuous/mixed), `FunctionSymbol`; mixed-space composition via `mixle_sim/elements/finite_element.py::mixed_element()`. PR #4 (`7109e61`) / PR #5 "SIM-D1" (`6888a3a`), both merged. |
| MP-D3 | Measures and variational form IR | owned-elsewhere | mixle-sim | `forms.py::Measure`/`MeasureKind` (VOLUME/BOUNDARY/INTERFACE)/`Integral`/`WeakForm` (validates trial/test participation and symbol/measure references). PR #4 (`7109e61`, merged). |
| MP-D4 | Boundary/interface/algebraic constraints (Dirichlet/Neumann/Robin/periodic/multipoint/Lagrange-multiplier/penalty/Nitsche) | owned-elsewhere (partial) | mixle-pde (legacy, Dirichlet-only landed) | `mixle_pde/fem.py::_dirichlet_map`/`solve_simplex_poisson` — strong Dirichlet elimination only, merged. A much fuller typed constraint system, `mixle_sim/constraints.py::ConstraintKind`/`EnforcementStrategy` (essential/natural/robin/periodic/multipoint/lagrange_multiplier/nitsche/penalty/mortar) with only `eliminate_essential_constraint()` fully implemented end-to-end, exists on **open PR #11** (`31f229f`), unmerged — not counted as landed. Note: `mixle_pde/boundaries.py` is a different domain entirely (underwater-acoustic reflection physics), not this capability — do not conflate. |
| MP-D5 | Symbolic residual differentiation | not-started | — | Confirmed absent in mixle-sim (`forms.py`/`WeakForm` has no `.diff()`/Jacobian machinery; `programs.py::DerivativeMode` is a declarative tag, not an implementation) and mixle-physics (`elaborate.py` docstring explicitly lists symbolic differentiation as out of scope, deferred). `mixle_pde/adjoint.py` gives Torch **automatic** (not symbolic) differentiation for Torch-differentiable forwards only — a narrower, different technique, kept here as the closest partial analog but not counted toward this task. |
| MP-D6 | Native reference form assembler | owned-elsewhere (partial) | mixle-sim | `mixle_sim/assembly.py::assemble_p1_diffusion()` (scalar P1 diffusion only, not a general residual/Jacobian assembler over arbitrary `WeakForm` IR), `reconstruct_scalar_field()`, `validate_interface_balance()`. PR #4 (`7109e61`, merged). Reference-element machinery (`elements/finite_element.py::lagrange_element`/`raviart_thomas_element`/`nedelec_element`/`bdm_element`/`mixed_element`/`enrich_scalar_element`, PR #5) is thorough and ready for a fuller assembler. `mixle_pde/fem.py` remains a separate, narrower P1 simplex assembler on the legacy side. |
| MP-D7 | DG fluxes and stabilization library (SIPG/NIPG, upwind, SUPG/PSPG/GLS) | not-started | — | No DG flux or stabilization scheme in mixle-sim (`elements/` has a `continuity` concept but no numerical-flux library) or mixle-pde. |

## MP-E — Compilation and production backends

| ID | Description | Status | Owner | Evidence |
|---|---|---|---|---|
| MP-E1 | Strong/coefficient/general-form lowering | owned-elsewhere | mixle-sim | `mixle_sim/compiler.py::compile_problem()` — the preserved "foundation" physics→math lowering (`SimulationPlan` with `GeometrySpec`/`MeshPlan`/`FieldPlan`/`WeakEquation`/`BoundaryRule`/`InterfaceCoupling` → `CompilationResult`), independent of and predating the newer mesh/forms/elements IR. PR #1 (`183e4f5`, merged). |
| MP-E2 | UFL/FFCx/DOLFINx compiler backend | not-started | — | Zero hits for UFL/FFCx/DOLFINx anywhere in mixle-sim or mixle-pde. |
| MP-E3 | PETSc solver and mesh bridge | not-started | — | Zero hits for PETSc/petsc4py anywhere in mixle-sim or mixle-pde; `mixle_pde/linear_solve.py` uses SciPy sparse solvers only. |
| MP-E4 | Compilation cache and reproducible manifests | owned-elsewhere (partial) | mixle-sim | `compiler.py::LoweringReceipt` (source_theory_hash/plan_hash/math_problem_hash via canonical-JSON SHA-256), PR #1, merged — reproducible manifests exist; no actual compile-artifact *cache* (store/retrieve by hash) implementation found anywhere. |
| MP-E5 | Legacy kernel adapters (wave/flow/EM/transport/simplex-FEM as backend participants) | owned-elsewhere | mixle-pde | `mixle_pde/pde_backend_registry.py` registers five `PDEBackendProfile` kernels — `fem-p1-simplex`, `wave-fd-leapfrog`, `flow-fd-streamfunction`, `em-fdtd-yee`, `transport-fd-advdiff`. PR #67 (`1dd3ed7`, merged). Confirmed out of scope for mixle-sim by that repo's own `docs/architecture.md` ("PDE provides optional specialist execution adapters"). **In-flight, not counted:** PR #72 (open) would add a sixth `spectral_flow` kernel. |
| MP-E6 | Capability negotiation and backend selection | owned-elsewhere | mixle-pde | `mixle_pde/problem_adapter.py::PDEBackendProfile`/`inspect_math_problem`/`require_compatible`/`UnsupportedPDEProblem`, commit `bc7b49e`, PR #66 (merged); called before every `run_math_problem` invocation. mixle-sim's `compiler.py::SimulationPlan.solve_plan` has only an unimplemented `backend: None` placeholder — no negotiation logic there. |

## MP-F — Solver studies, convergence, and adaptivity

| ID | Description | Status | Owner | Evidence |
|---|---|---|---|---|
| MP-F1 | Linear solvers and preconditioner policy (direct, CG, GMRES, AMG, Schwarz, field-split, Schur) | owned-elsewhere | mixle-discrete | `mixle_numerics/linear_solvers.py` — `SolverFamily{DIRECT, CG, MINRES, GMRES, BICGSTAB, AMG}`, `PreconditionerFamily{JACOBI, ILU, ICC, SCHWARZ, FIELD_SPLIT, SCHUR_COMPLEMENT, AMG}`, symmetry/definiteness-aware `select_default_solver`, `IncompatibleSolverError` pre-flight rejection, matrix-free `LinearOperator`. PR #8 "DISC-F1" (`0387d9c`, merged, 1116 lines). Module docstring is explicit that it's deliberately backend/assembly-agnostic — "a physics/simulation caller is expected to adapt its own assembled operator into a `LinearOperator`" — exactly the generic capability this task asks a PDE bridge to adapt to rather than reimplement. `mixle_pde/linear_solve.py` (direct sparse + factorization caching) remains a narrower legacy alternative. |
| MP-F2 | Nonlinear solves, continuation, constraints | owned-elsewhere (partial) | mixle-discrete | `mixle_numerics.nonlinear_solvers.NonlinearMethod.JFNK` — a real Jacobian-free Newton-Krylov implementation: matrix-free finite-difference Jacobian-vector products (via a new `NonlinearProblem.matrix_free` field wired into the module's existing selection mechanism) fed to the existing GMRES Krylov engine, verified against a known linear system, the existing circle/line closed-form root (cross-checked against full Newton), and a discretized 1-D Bratu problem with a known closed-form solution (cross-checked against full Newton to near machine precision). mixle-discrete PR #66 "MP-F2" (`ce33158`, merged). This covers only the JFNK sub-piece: continuation (natural-parameter and arclength/pseudo-arclength stepping) and bound/general constraints remain absent from mixle-discrete. `mixle_pde/continuation.py::natural_continuation`/`arclength_continuation` plus a Bratu-fold reference (`bratu_problem`/`bratu_reference_fold`) already exist legacy-side but are tracked separately from this new-architecture row and not counted toward it. |
| MP-F3 | Time integration and DAE studies | owned-elsewhere (partial) | mixle-pde (legacy, explicit-only) | Wave/flow/EM/transport kernels in `pde_backend_registry.py` use explicit leapfrog/FDTD/FD time stepping only; no implicit RK/BDF/generalized-alpha/IMEX/adaptive-step/event/adjoint-checkpoint framework anywhere in mixle-sim or mixle-discrete either. |
| MP-F4 | Eigenvalue, modal, harmonic, frequency sweeps | owned-elsewhere (partial) | mixle-pde (legacy, narrow) | `mixle_pde/normal_modes.py` covers acoustic normal modes specifically. mixle-discrete has no eigensolver portfolio (`np.linalg.eigvalsh` appears only as an internal lattice-conditioning check in `integer_ls.py`, not a public eigenproblem solver); mixle-sim's `benchmarks.py` documents a closed-form TE101 eigenvalue as a manufactured *reference target*, not an eigensolver. |
| MP-F5 | Error estimation and solve adaptation | not-started | — | mixle-sim's C7 evidence (`RefinementRelation`, `AdaptationRecord`) is lineage/data-structure only, not an estimator or adaptation-driving algorithm. |
| MP-F6 | Scaling, nondimensionalization, automatic solver hints | not-started | — | No nondimensionalization/scaling module in mixle-pde, mixle-sim, or mixle-discrete. |

## MP-G — Multiphysics coupling and field exchange

| ID | Description | Status | Owner | Evidence |
|---|---|---|---|---|
| MP-G6 | Boundary/interface ontology, typed conjugate ports | owned-elsewhere | mixle-physics | `semantics.py::PortSpec`, `InteractionSpec` (aliased `InterfaceSpec`), `ConjugatePairSpec`/`CONJUGATE_PAIRS`/`register_conjugate_pair`, `InterfaceLawKind`, `CouplingClass`, `ExchangeRole`, `PortCausality`, `PowerRelation`. Commit `b1c1b44`, PR #6 "PHYS-G6" (merged) — named explicitly after this task id in the commit message and `docs/architecture.md`. |
| MP-G1 | Coupling graph and participant protocol | owned-elsewhere (partial) | mixle-sim | `mixle_sim/programs.py::CouplingPlan`/`PortEndpoint` (component_id/port_id/selection_id/dimension/orientation) is a validated *data contract* for participants/ports, PR #1 (`183e4f5`, merged) — not an execution engine (no checkpoint/rollback/callback runtime). mixle-discrete has no generic participant-protocol equivalent at all. `mixle_pde/multiphysics.py::run_coupled` is a narrower legacy two-field FD coupling. **In-flight, not counted:** PR #71 in mixle-pde (open) adds a fuller monolithic+partitioned composite-heat reference. |
| MP-G2 | Mesh-to-mesh field mapping | owned-elsewhere (partial) | mixle-sim | `programs.py::TransferPlan`/`TransferMethod` (IDENTITY/INTERPOLATION/MORTAR/CONSERVATIVE_PROJECTION) is schema/validation only — PR #1, merged. No interpolation/projection numerics implemented anywhere. |
| MP-G3 | Monolithic coupled forms (block residual/Jacobian, field splits) | owned-elsewhere (partial) | mixle-discrete | `mixle_numerics/linear_solvers.py::BlockLinearSystem` — a generic 2×2 saddle-point block operator `[[A,Bt],[B,C]]` used by field-split/Schur-complement preconditioners, PR #8 (`0387d9c`, merged). This is linear-only (no nonlinear residual/Jacobian assembly, no N-block/graph generality), so it covers the block-operator half, not the monolithic-nonlinear-coupling half. mixle-sim's `CouplingStrategy.MONOLITHIC` is an unimplemented enum tag. `mixle_pde/multiphysics.py::solve_poisson`/`solve_elasticity`/`run_coupled` remains a narrower legacy FD-only path. |
| MP-G4 | Partitioned co-simulation | owned-elsewhere (partial) | mixle-sim | `programs.py::CouplingStrategy.PARTITIONED` plus `CouplingPlan.convergence_measure`/`maximum_iterations` enforce that a partitioned plan declares a convergence policy — schema-level only, PR #1, merged; no fixed-point/quasi-Newton execution loop implemented. **In-flight, not counted:** PR #71 in mixle-pde (open) claims "bounded partitioned composite-heat solves." |
| MP-G5 | Multirate time coordination and restart | owned-elsewhere (partial) | mixle-sim | `programs.py::TimeCoordination` (source_step/target_step/exchange ∈ synchronized/subcycled/interpolated) covers multirate coordination as a validated plan, PR #1, merged; no restart concept or execution found. |
| MP-G7 | Global equations and 0D/1D/network coupling | not-started | — | `mixle_sim/geometry.py::Network`/`NetworkNode`/`NetworkEdge` is 1-D *geometry* (wells/fractures/pipes), not lumped-parameter global-equation coupling, and exists only on open PR #10 (unmerged) regardless. |
| MP-G8 | Coupling template planner and interface diagnostics | not-started | — | No coupling-template catalog or interface-balance-receipt planner found anywhere. |

## MP-H — Composable physics and materials library

The newer plan splits this workstream: physical **laws** (balance/constitutive definitions, mixle-physics
territory, C4–C7) versus their **numerical solve** (mixle-sim/mixle-pde territory). mixle-physics has begun
building an independent law layer for two of the eight physics families; mixle-pde's pre-existing legacy
numerics cover the solve side for all eight (with varying depth) but with no separated law artifact behind
them.

| ID | Description | Status | Owner | Evidence |
|---|---|---|---|---|
| MP-H1 | Heat, diffusion, species transport, reactions | owned-elsewhere | mixle-physics (law) + mixle-pde (legacy numerics) | Law: `catalog.py::heterogeneous_heat_theory()` — Fourier heat conduction via `LawSpec`/`TermSpec`, registered `"heat-diffusion"` pack, commit `c67879f`, merged. Numerics: `mixle_pde/heat.py`, `material_transport.py`, `reactive_transport.py`, `smoluchowski.py`. |
| MP-H2 | Solid mechanics, materials, contact | owned-elsewhere | mixle-physics (law, partial) + mixle-pde (legacy numerics) | Law: `catalog.py::elasticity_theory()` — small-strain isotropic elasticity + momentum balance, `"elasticity"` pack, commit `c67879f`, merged (plasticity/hyperelasticity/contact not covered by the law layer). Numerics: `mixle_pde/elastic.py`, `elastic_aniso.py`, `plate.py`, `beam.py`, `poroelastic.py`. |
| MP-H3 | Fluid mechanics and CFD backend | owned-elsewhere (partial) | mixle-pde (legacy numerics only; no law layer) | `mixle_pde/flow.py` (`NavierStokes2D`, registered in `pde_backend_registry.py`), `flow3d.py`, `gas_dynamics.py`. No RANS/LES turbulence interface, no FVM path, no mixle-physics fluid law found. |
| MP-H4 | Electromagnetics and electrostatics | owned-elsewhere (partial) | mixle-pde (legacy numerics only; no law layer) | `mixle_pde/maxwell.py` (`Maxwell3D` Yee-grid FDTD, registered), `electrostatics.py`, `em_diffusion.py`, `em_diffusion_3d.py`, `induced_polarization.py`, `pnp.py`, `guided_wave.py`. mixle-physics has zero Maxwell/EM hits (only a generic `voltage`/`current` conjugate-pair stub). |
| MP-H5 | Acoustics, waves, absorbing domains | owned-elsewhere (partial) | mixle-pde (legacy numerics only; no law layer) | `mixle_pde/wave.py` (registered), `wave3d.py`, `wave_pml.py`, `helmholtz_pml.py`, `normal_modes.py`, `parabolic_equation.py`, `ray_scattering.py`, `sound_speed.py`, `refractivity.py`, `attenuation.py`, `dispersion.py`. mixle-physics has only an `acoustic-pressure-normal-velocity` conjugate-pair stub, no acoustic law pack. |
| MP-H6 | Porous media, multiphase, phase field | owned-elsewhere (partial) | mixle-pde (legacy numerics, narrow; no law layer) | `mixle_pde/poroelastic.py`, `groundwater.py` (Darcy/Biot-class only). No Cahn-Hilliard/Allen-Cahn or two-phase (Buckley-Leverett-class) evidence anywhere; mixle-physics has zero porous/multiphase/phase-field hits. |
| MP-H7 | Reacting flow, particles, specialty participants | owned-elsewhere (partial) | mixle-pde (legacy numerics, narrow) | `mixle_pde/reactive_transport.py`, `smoluchowski.py` (aggregation/coagulation). No Lagrangian particle tracking/deposition or DEM/SPH participant contract found. |
| MP-H8 | "Physics Builder" and reusable term catalog | superseded | mixle-physics `catalog.py` (successor, narrower) | Newer plan §2.2: *"'Physics Builder' is not used as a Mixle feature or product name"* — confirmed: repo-wide grep for "physics builder" across mixle-physics code and docs returns zero hits. The underlying reusable-term-catalog capability has a real but narrower successor: `catalog.py::PhysicsCatalog` (`discover()`/`build()`) currently registers exactly three fixed, hand-built packs (heat-diffusion, elasticity, thermoelastic) rather than an open, string-keyed extensible library — commit `c67879f`, merged. |

## MP-I — Bayesian inversion, optimization, and UQ

mixle-pde carries the largest body of pre-existing Bayesian/UQ code of any workstream, all still resident in
`mixle_pde` itself. `ownership.py` classifies nearly all of it `migrate` → `PRJ-CORE` (i.e. the newer plan's
intended final owner is Core/mixle-discrete's generic inference layer), but as of this survey **none of it
has landed in mixle-discrete under that name** — mixle-discrete's own survey found no MCMC/HMC/SMC/VI/adjoint
machinery at all (see notes below). It is recorded here as `owned-elsewhere: mixle-pde` (today's real,
pre-migration location), not as `not-started`, because the capability genuinely exists and runs today.

| ID | Description | Status | Owner | Evidence |
|---|---|---|---|---|
| MP-I1 | Tangent/discrete-adjoint backend contract | owned-elsewhere (partial) | mixle-pde | `mixle_pde/adjoint.py::torch_adjoint_jacobian`/`torch_adjoint_jvp` — automatic (not discrete-adjoint) differentiation for Torch-differentiable forwards only, no checkpointing/transient-adjoint support. mixle-discrete's `expr.py` (open PR #11, unmerged) has zero AD/gradient/adjoint symbols — confirmed no generic successor exists yet anywhere. `ownership.py`: `adjoint` → `migrate`/`PRJ-CORE`. |
| MP-I2 | Data/sensor/observation-operator architecture | owned-elsewhere | mixle-pde | `mixle_pde/observations.py` (`Observation`, used throughout `field_*` modules) plus `io/` format ingest (`segy.py`, `las.py`, `em_soundings.py`, `insar.py`, `potfield.py`, `assays.py`, `gis.py`). `ownership.py`: `observations` → `migrate`/`PRJ-CORE`. |
| MP-I3 | Unknowns, transformations, priors, hierarchical parameter models | owned-elsewhere | mixle-pde | `field_priors.py::FieldGaussianPrior`, `blocky_priors.py` (minimum-support/TV/dip-rotated-gradient priors), `latent.py`. `ownership.py`: all three → `migrate`/`PRJ-CORE`. |
| MP-I4 | Likelihood, discrepancy, numerical-error models | owned-elsewhere | mixle-pde | `misfit.py`, `model_error.py` (PR #34, "Model-error (theory-error) term in the likelihood"). `ownership.py`: both → `migrate`/`PRJ-CORE`. |
| MP-I5 | Inference engines and capability negotiation (MAP/GN, Laplace, HMC/NUTS, pCN, SMC, VI, ensemble Kalman) | owned-elsewhere (partial) | mixle-pde | `field_gauss_newton.py::gauss_newton_invert` (MAP/Gauss-Newton), `field_mcmc.py::metropolis_field_invert` (Metropolis, not HMC/NUTS), `field_assimilation.py::particle_assimilate_4d` (particle filter with pCN rejuvenation), `assimilate_4d_ensemble` (ensemble Kalman). mixle-discrete's `annealing.py` is discrete combinatorial Metropolis (CVP/QUBO/ILP), a different problem class — not a continuous-parameter inference engine; its `runtime.py::BackendRegistry` gives a reusable capability-negotiation *pattern* but no inference backends are registered against it. No HMC/NUTS, SMC-tempering, or VI engine found anywhere. |
| MP-I6 | General inverse-study bridge and posterior artifacts | owned-elsewhere | mixle-pde | `posterior_query.py` (PR #6, "posterior extraction + compact storage: marginals, derived quantities, low-rank"). `ownership.py`: → `migrate`/`PRJ-CORE`. **Complementary, in-flight, not counted:** mixle-discrete's `multiphysics_reference.py` (open PR #9, `3c4de57`) adds a generic `PosteriorPoint`/`IdentityBundle`/credible-interval/identifiability-warning bridge explicitly designed to be PDE-agnostic — unmerged. |
| MP-I7 | Field, boundary, geometry, shape inversion | owned-elsewhere (partial) | mixle-pde | `field_inversion.py`, `field_gauss_newton.py` (field-level); `shape.py::shape_optimize`/`level_set_material` (level-set/parameterized-geometry), though framed more as forward shape optimization than Bayesian shape inversion — suggestive, not conclusive, of full coverage. |
| MP-I8 | Posterior validation, calibration, identifiability, model comparison | owned-elsewhere | mixle-pde | `posterior_calibration.py` (`truth_coverage`, `heldout_observation_check`, `observation_sensitivity`, `Recalibration` — PR #20), `model_selection.py` (`log_evidence_laplace`, `bayes_factor`, `rank_hypotheses`), `informativeness.py` (`prior_dominated_mask` — PR #18/#2). |
| MP-I9 | Scalable, restartable, reproducible Bayesian execution | not-started | — | `mixle_pde/verification/capability_inventory.py`'s own methodology note: *"parallel_status is 'single_process' for every entry: a repo-wide sweep found no mpi4py, no multiprocessing, and no concurrent.futures/joblib usage anywhere under mixle_pde/"* — an honest, self-declared gap. mixle-discrete's `math_ir.ExecutionPolicy.checkpoint_ref` is an unimplemented field, not a runtime. |
| MP-I10 | PDE-constrained optimization and experimental design | owned-elsewhere (partial) | mixle-pde | `monitoring_design.py::GaussianSourcePosterior` (EIG-based sensor placement, PR #65), `voi.py::expected_variance_reduction`/`next_best_observation`. mixle-discrete's `QueryKind.OPTIMIZE` is backed only by discrete bounded-integer exhaustive search, not continuous/PDE-constrained optimization. No shape/topology-optimization-under-constraint or TAO-class integration found. |
| MP-I11 | Forward UQ, global sensitivity, reliability | owned-elsewhere (partial) | mixle-pde | `decision_quantities.py::region_mass`/`prob_exceed` (sampling-based forward uncertainty propagation of derived quantities). No Sobol/global-sensitivity, QMC, polynomial-chaos, or reliability-analysis evidence anywhere, including mixle-discrete (confirmed zero hits). |

## MP-J — AI-native model construction and repair

| ID | Description | Status | Owner | Evidence |
|---|---|---|---|---|
| MP-J1 | Bounded modeling tool service (the fifteen tools) | owned-elsewhere (partial) | mixle-pde | `mixle_pde/tools.py::run_inversion` — one bounded, typed-ref-in/typed-result-out entry point, not the full fifteen-tool construct/mesh/compile/solve/verify/train/query surface. mixle-physics has a parallel, more architecturally-aligned start (`theory.inspect`/`theory.patch`, PR #10, **open, unmerged**) but that is not counted as landed. |
| MP-J2 | Physics/material/method/solver knowledge catalog | not-started | — | No structured catalog with equations/applicability/benchmark-evidence metadata found anywhere. |
| MP-J3 | Diagnostic ontology and deterministic repair recipes | not-started | — | No diagnostic-code/repair-action catalog found. |
| MP-J4 | Coarse-to-fine agent execution loop | not-started | — | No requirements→draft→coarse-solve→refine loop with bounded patches/retries found. |
| MP-J5 | Structured knowledge handoff | not-started | — | No IC-13-style knowledge item/bundle/delta representation found. |
| MP-J6 | Agent safety and authority boundaries | not-started | — | No workspace/object-store scope, input allowlist, or approval-gate enforcement found. |

## MP-K — Artifacts, result queries, verification, and parity evidence

| ID | Description | Status | Owner | Evidence |
|---|---|---|---|---|
| MP-K1 | Content-addressed simulation artifact store | owned-elsewhere (partial) | mixle-pde + mixle-discrete (primitives only) | mixle-pde: `io/artifacts.py` (contended file — read but not modified by this task) plus `canonical_adapter.py`'s sha256 digest of a normalized linear-system record, and `ownership.py::migration_inventory_digest()`. mixle-discrete: `math_ir.semantic_digest`/`canonical_json`, `problem.MathematicalProblem.semantic_hash()`, and `research.ResearchEvent`'s hash-chained ledger (PR #3/#7, merged) are strong generic content-addressing *primitives* — but no `put`/`get`/blob-storage/lineage-query *store* implementation exists anywhere (confirmed by direct grep for store-shaped symbols). |
| MP-K2 | Backend-neutral result and postprocessing queries | owned-elsewhere (partial) | mixle-pde | `posterior_query.py`, `decision_quantities.py` give posterior-summary/derived-quantity queries; no point/line/surface/volume probe or pathline/spectra query framework across native/FEM backends. |
| MP-K3 | Verification engine and convergence receipts | owned-elsewhere (partial) | mixle-discrete | `mixle_numerics/problem.py::SolveReceipt` (residual/tolerance/status invariants enforced in `__post_init__`, PR #2) plus the independent `receipt_verifier.py::verify_envelope` (PR #6 "DISC-IQ3", `3f516da`, **merged**) — a solid, already-landed generic pass/fail receipt protocol that re-derives legality from raw fields and detects tampering via digest binding. **In-flight, not counted:** `multiphysics_reference.py::verify_reference_evidence` (open PR #9) adds an actual MMS-style convergence-order check (`_refinement_orders()` against a `minimum_refinement_order` threshold) plus residual/conservation/interface-jump/coupling checks — unmerged. mixle-pde's own `multiphysics_reference.py`/`reference_lifecycle.py` (PR #71, open) claims similar MMS evidence, also unmerged. |
| MP-K4 | Cross-backend and competitor-reference parity harness | not-started | — | No FEniCSx or external-reference comparison harness found anywhere. |
| MP-K5 | Experimental and real-data validation tiers | not-started | — | mixle-pde's `io/` ingests real field data as *inputs* (SEG-Y, LAS, InSAR, assays), but no `manufactured`/`analytic`/`code-to-code`/`experimental`/`field` validation-tier labeling scheme was found. |

## MP-L — Scale, operations, security, and release

| ID | Description | Status | Owner | Evidence |
|---|---|---|---|---|
| MP-L1 | MPI mesh partitioning and distributed assembly | not-started | — | Confirmed absent repo-wide in mixle-pde (see MP-I9 evidence). mixle-sim's `mesh.py::Partition` is a data structure representing rank/owned/ghost cells with validation, but no MPI, no partitioning algorithm (METIS/ParMETIS), and `assembly.py`'s assembler is single-rank only. |
| MP-L2 | Matrix-free, multigrid, large-problem paths | not-started | — | Zero hits for multigrid/matrix-free-operator paths in mixle-sim or mixle-discrete beyond `linear_solvers.py`'s matrix-free `LinearOperator` type (a building block, not a multigrid/large-problem path). |
| MP-L3 | GPU/backend acceleration decision | not-started | — | No device-capability negotiation or measured-GPU-benefit ADR found anywhere; incidental Torch usage gives device portability, not a deliberate acceleration decision. |
| MP-L4 | Sandboxed job orchestration and resource governance | not-started | — | `mixle_pde/simulation_service.py` has scenario/result plumbing but no queue/quota/heartbeat/retry governance. **In-flight, not counted:** PR #71 (open) claims "durable create, inspect, monitor, cancel, resume, run, and retrieve operations with typed failures." |
| MP-L5 | Packaging, compatibility, observability, release gates | owned-elsewhere (partial) | mixle-pde | PR #69 (extras/zero-extras contract) and `docs/capability-matrix.json` cover packaging; no logs/metrics/traces observability or deprecation-policy telemetry found. |

## MP-M — Training data, evaluations, migration, and demonstrations

| ID | Description | Status | Owner | Evidence |
|---|---|---|---|---|
| MP-M1 | Verified modeling-trace corpus | not-started | — | No requirements→assumptions→repair trace corpus found anywhere. |
| MP-M2 | AI curriculum and training objectives | not-started | — | No curriculum/reward-schema artifact found. |
| MP-M3 | Held-out agent modeling evaluation | not-started | — | No held-out task suite with topology/equation/coupling holdout found. |
| MP-M4 | Migration, examples, AI-first documentation | not-started | — | **In-flight, not counted:** PR #73 (open) adds `docs/migrations/legacy-to-canonical-adapters.md`, scoped explicitly to the `problem_adapter`/`pde_backend_registry` boundary, and is itself explicit that multiphysics/inverse/UQ/surrogate modules and most of `mesh.py` are *not yet* migrated — unmerged as of this survey. |

## MP-N — Surrogate and reduced-order modeling lifecycle

| ID | Description | Status | Owner | Evidence |
|---|---|---|---|---|
| MP-N1 | Surrogate contracts, artifacts, typed replacement boundary | owned-elsewhere (partial) | mixle-pde | `surrogate.py::Surrogate` — a distilled, conformally-calibrated student model with an honest `defer()` gate (precision-floor + OOD density check), `distill_forward`, `to_task_cascade_adapter`. A working single-model surrogate contract, not the full `SurrogateSpec` schema (fidelity levels, conservation constraints, port/unit/frame compatibility, semantic versioning). |
| MP-N2 | DOE, adaptive sampling, training-data manifests | not-started | — | No design-of-experiments or active-learning sampling module found. |
| MP-N3 | Projection and reduced-basis models (POD/hyper-reduction) | not-started | — | No POD/Galerkin-ROM code found. `sensitivity_compress.py` is wavelet/Haar compression of a sensitivity Jacobian for UQ — not a reduced-order forward model; do not conflate the two. |
| MP-N4 | Probabilistic scalar/QoI emulators (GP/PC/neural) | owned-elsewhere (partial) | mixle-pde | `surrogate.py::Surrogate` (calibrated neural-emulator-class student) partially covers this; no GP or polynomial-chaos adapter found. |
| MP-N5 | Spatial/temporal/geometry-varying operator surrogates | not-started | — | No FNO/DeepONet/graph-neural/geometry-conditioned operator surrogate found. |
| MP-N6 | Certification, calibration, validity, error budgets (surrogate-specific) | owned-elsewhere (partial) | mixle-pde | `surrogate.py`'s conformal calibration (`qhat` precision floor) and OOD `defer()` gate cover part of this; no independent third-party validation, drift monitoring, or stability/conservation test suite for surrogates found. (`posterior_calibration.py` covers the analogous concept for *inversion* posteriors, a related but distinct capability — not counted here.) |
| MP-N7 | Hybrid full-order/surrogate coupling and inversion | owned-elsewhere (partial) | mixle-pde | `surrogate.py::distill_forward` gives basic full/surrogate substitution. **In-flight, not counted:** PR #71 (open) claims "validates surrogate use with authoritative fallback" as part of its coupled reference scenario. |
| MP-N8 | Registry, promotion, monitoring, retraining, rollback | owned-elsewhere (partial) | mixle-mlops | `mixle_mlops/control/pde_surrogate.py` — a real registry/promotion bridge for the operator surrogate: `train_operator_surrogate_job` is a `control.worker` `WorkerHandler` that fits and calibrates a real `mixle_pde.operator_surrogate.LinearOperatorSurrogate` on the pre-existing durable-job machinery (claim/lease/checkpoint/retry/completion); `land_pde_artifact` bridges a surrogate already stored in `mixle_pde.artifact_store.ArtifactStore` into this platform's `LocalArtifactStore`, sha256 content-hash verified equal across both stores; `register_pde_operator_surrogate` turns the landed artifact into an immutable `ModelCandidate` so `check_registry_integrity` and `registry.promote`/`rollback` govern it like any other candidate, gated on the surrogate's own real calibration-imprecision check (`not calibration.imprecise`), never fabricated or loosened. mixle-mlops PR #64 "MP-N8" (`2d9a3d2`, merged), 6 new tests covering the full train → register → integrity-check → promote → resolve lifecycle; mixle-pde itself required zero changes. Covers only the operator-surrogate (MP-N5) registry/promotion/rollback path: MP-N1's fuller `Surrogate` has no registry path yet, and no drift-triggered retraining or cross-repo monitoring wiring was added — `mixle_pde/drift_monitor.py` (MP-N6) remains same-repo-only, not bridged into this registry. |

## Cross-cutting observations

- **Duplicate-effort risk, not duplicate source of truth:** mixle-physics, mixle-sim, mixle-discrete, and
  mixle-pde each independently opened a PR titled "Add \[executable/physical/multiphysics\] ... inversion
  reference" on 2026-07-15/16 (mixle-pde #71, mixle-physics #7, mixle-sim #9, mixle-discrete #9) — four
  parallel, unmerged, differently-scoped attempts at overlapping golden-scenario/verification work. None had
  landed as of this survey. This is worth flagging to whoever reviews those four PRs together: it is exactly
  the kind of near-duplicate-source-of-truth risk M2 exists to catch, even though none of the four capability
  claims is yet real enough to score as `owned-elsewhere` on its own.
- **Open-PR backlog is architecturally coherent, not noise:** across mixle-physics (#7–#10), mixle-sim
  (#7–#11), and mixle-discrete (#9–#11), the still-open work consistently targets exactly the gaps this
  ledger identifies as `not-started` (symbolic differentiation, full constraint enforcement, CSG geometry,
  MMS verification). If those land in the next review cycle, roughly a dozen of today's `not-started` rows
  above (MP-C3, MP-C9, MP-D1's physics half, MP-D4's full-constraint half, MP-K3's MMS half) would flip to
  `owned-elsewhere` without any new work being started — worth re-running this survey after that wave merges.
- **mixle-pde's Bayesian/UQ code (MP-I) is real, substantial, and un-migrated:** ten of eleven MP-I tasks
  have working code today, entirely inside `mixle_pde`, with `ownership.py` already recording `PRJ-CORE` as
  the intended final owner for most of it. No migration has started. This is the single largest body of
  "genuinely done, needs relocation, not reimplementation" capability found in this survey.
