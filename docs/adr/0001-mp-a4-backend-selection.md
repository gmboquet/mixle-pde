---
id: ADR-0001
title: "MP-A4: external numerical backend selection (Gmsh, DOLFINx/FFCx, PETSc, MPI, preCICE, OpenFOAM)"
status: proposed
date: 2026-07-16
deciders: mixle-pde maintainer, PRJ-PDE backend/runtime owner (MP-A backend-decision track)
related_task_ids: [MP-A4]
supersedes: null
---

# ADR-0001: External numerical backend selection

## Format note

`mixle-discrete`'s `docs/adr/0000-backend-adapter-template.md` (`DISC-D0.9`, PR #5, `349d5b7`) is the closest
existing ADR precedent anywhere in the program, and this record borrows its discipline: YAML front matter, one
grounded capability-gap claim per backend, an explicit license/packaging cost, and an unhedged
Adopt/Defer-until-X/Never verdict with no placeholder text. It does not reuse that template's section-by-section
shape verbatim, because that template is built for a single confirmed adapter (its `R4`/`R5`/`R7` sections map a
concrete `BackendAdapter` contract, walk isolation controls, and specify a differential-test plan for a backend
already being built). This record is the opposite motion: a portfolio survey across six backends, most of which
this ADR concludes should not be built yet. Each backend section below instead answers three questions in a
fixed order -- **capability gap**, **cost**, **decision** -- so a reviewer can still approve or reject each verdict
from the section alone, without inheriting sections (contract mapping, isolation controls) that would be
unanswerable placeholder text for a backend this record defers.

## Context and problem statement

`docs/reconciliation/mp-task-ledger.md`'s `MP-A4` row records: *"No ADR or spike code for
Gmsh/DOLFINx/FFCx/PETSc/preCICE/OpenFOAM in mixle-pde, mixle-sim, or mixle-discrete (confirmed by direct grep in
the mixle-sim survey: zero hits for 'gmsh'/'opencascade'/'petsc'). `mixle_pde/adjoint.py` gives a working
Torch-adjoint pattern but is not framed as this task's spike/ADR. `mixle-discrete` PR #5 ... establishes an ADR
template discrete-side, but no PDE-relevant backend ... has been spiked against it."* This record closes that
gap for the six backends the ledger names that are actually PDE-adjacent (`checkpoint-restart` landed separately
as `MP-I9`, PR #99, `98d9105`, and is a native `mixle_pde.field_mcmc` capability with no backend dependency;
`Torch-adjoint` is `mixle_pde/adjoint.py`, an existing, working pattern, not an unresolved backend choice --
neither needs a selection decision here).

This repo already has substantial numerical infrastructure of its own: nine registered legacy kernels
(`mixle_pde/pde_backend_registry.py` -- FD/FDTD/FEM/spectral solvers for elastic, Helmholtz, groundwater
transport, Poisson/diffusion, acoustic wave, streamfunction-vorticity flow, electromagnetics, advection-diffusion,
and spectral Navier-Stokes), its own sparse linear-solve stack (`mixle_pde/linear_solve.py`), and its own mesh
utilities (`mixle_pde/mesh.py`). It has never written down why it has not adopted the heavier external backends
named in `notes/mixle-pde-ai-native-multiphysics-work-plan.md`, or under what conditions it should. That plan
already states an aspirational position for five of the six backends below (Gmsh, FEniCSx/DOLFINx/FFCx, PETSc,
OpenFOAM, preCICE all appear by name in its "Open numerical infrastructure" and "Partitioned-coupling precedent"
framing) and explicitly gates two of them on this exact task: *"No coupling dependency is adopted before the
MP-A4 decision gate passes"* (preCICE) and *"OpenFOAM is the first candidate, selected only after an ADR and
license/deployment review"* (OpenFOAM). This record is that gate.

The standing program convention this record must respect (`notes/parallel-implementation-tasks/README.md`,
"Rules for every card agent," item 4) names four of these six backends explicitly: *"Heavy backends (FLINT,
Gmsh, DOLFINx, PETSc, SLEPc, GAP, PARI, MPI, Torch) go behind **optional extras** with a pure-Python/NumPy
fallback or a mock, so your module imports with none of them installed."* The same document's standing
conventions add: *"Reference kernels stay small and legible (pure Python/NumPy); production power comes from
adapters."* `docs/solver-selection-and-inversion-guide.rst`'s "Optional Backends" section states the same rule
in this repo's own voice: *"SciPy and sparse linear algebra are part of the normal package path. Torch, GPU, or
other accelerated paths should remain optional ... Importing the package without optional accelerators should
not fail."* Every verdict below is written against that constraint: adoption never means a new hard dependency,
only a new optional extra with a working fallback.

## Backend-kind classification

Borrowing `DISC-D0.9`'s seven-way taxonomy (`native-reference` / `native-compiled` / `in-process-library` /
`sandboxed-subprocess` / `persistent-worker` / `remote-distributed` / `formal-kernel`) to pre-classify each
candidate, since it drives the packaging and isolation reasoning in every section below:

| Backend | Kind | Why |
|---|---|---|
| Gmsh + OpenCASCADE | in-process-library | Official `gmsh` Python SDK wraps the C++ kernel in-process; no case-directory protocol needed for mesh generation calls. |
| DOLFINx/FFCx | in-process-library | Python bindings (`dolfinx`, `ffcx`) over a compiled C++ core with a JIT form compiler; called in-process, not shelled out to. |
| PETSc | in-process-library | `petsc4py` bindings called in-process; the library itself manages MPI internally rather than requiring a subprocess boundary. |
| MPI (`mpi4py`) | in-process-library | A transport, not a solver or mesh capability -- classified separately from PETSc because (see below) it has no standalone justification in this repo today. |
| preCICE | in-process-library (coordinating separate processes) | The `precice` Python bindings run in-process per participant, but the entire point of the library is coordinating *other*, separately-executed participant processes -- the isolation question is about the other side of the coupling, not this binding. |
| OpenFOAM | sandboxed-subprocess | No Python binding is the primary interface; OpenFOAM is a set of solver binaries driven by a case-directory file protocol. GPL-3 licensing (see below) makes this the only defensible integration pattern regardless. |

## Gmsh (+ OpenCASCADE)

**Capability gap.** `mixle_pde/mesh.py` provides `box_simplex_mesh` (deterministic structured-box simplex
meshes), `delaunay_mesh` (unconstrained Delaunay triangulation of a scattered point cloud via
`scipy.spatial.Delaunay`, convex-hull boundary only -- no facet constraints, no holes, no local sizing field),
`moving_mesh`/`space_time_mesh` (fixed-connectivity deformation and time extrusion), and quality *measurement*
(`SimplexMesh.simplex_quality`/`validate`, reporting min/mean quality and low-quality-cell counts) with no
quality *repair* or optimization. There is no CAD import (STEP/IGES/BREP), no constrained or boundary-conforming
unstructured meshing, and no physical-group tagging anywhere in this repo -- the ledger's `MP-C4`/`MP-C5` rows
confirm zero hits for any CAD/mesh-interchange format or for "gmsh"/"opencascade"/"occ" in `mixle-pde` or
`mixle-sim`. Every one of `pde_backend_registry.py`'s nine registered kernels declares `mesh_cell_types` of
`structured_grid`, `{triangle, tetrahedron}` (the FEM P1 kernel, fed by `box_simplex_mesh`), or `periodic_grid`
-- no registered kernel requests a CAD-derived or general unstructured cell type today.

**Ownership note.** `mixle_pde/ownership.py` classifies `mesh` (and `multiphysics`) as `_SIM_MODULES`:
disposition `"adapt"`, final owner `"PRJ-SIM"`, rationale *"Canonical numerical representation belongs to Sim;
the PDE import remains a tested compatibility/reference adapter."* This repo's own `docs/architecture.md` states
the same direction: *"Canonical dependency direction is Core values plus Physics theory → Sim lowering → Discrete
planning/verification → PDE specialist/native execution adapter → Sim reconstruction."* Meshing is upstream of
this repo by this repo's own recorded design; a Gmsh adapter's natural home is `mixle-sim`'s mesh-IR track
(`MP-C1`, already landed there per the ledger; `MP-C5` "Gmsh/OpenCASCADE backend," still not-started), not
`mixle-pde`.

**Cost.** The Gmsh SDK plus OpenCASCADE is a large native CAD-kernel dependency, categorically heavier than
anything currently in this repo's `all` extra (`pyamg`, `scikit-sparse`, `mpi4py`, `torch`, `scikit-learn` are
all self-contained Python packages with no CAD-toolchain requirement). `meshio` (named alongside Gmsh in the
work plan for XDMF/HDF5/VTK interchange) is a much lighter, pure-Python alternative worth distinguishing: file
interchange does not require the CAD kernel and could be adopted independently if a concrete need for it
appears, without pulling in Gmsh/OpenCASCADE at all.

**Decision: Defer.** Trigger: `mixle-pde` adopts `mixle-sim`'s mesh IR (its own `mesh` module is already tagged
`adapt`→`PRJ-SIM`, i.e. this migration is already committed, just not yet executed) **and** a concrete study
needs CAD-derived or constrained-boundary geometry that `box_simplex_mesh`/`delaunay_mesh` cannot express. Even
once that trigger fires, the adapter should be built in `mixle-sim` against `MP-C5`, not in this repo -- adopting
Gmsh directly into `mixle-pde` today would duplicate a capability this repo's own ownership record already
assigns elsewhere.

## DOLFINx/FFCx

**Capability gap.** `mixle_pde/fem.py` is a hand-written P1-simplex assembler for
`-div(diffusion grad u) = source` only (`assemble_simplex_stiffness_matrix`, `assemble_simplex_load_vector`,
`solve_simplex_poisson`) -- one element, one equation family, no general weak-form compilation. `mixle-sim`
already built the element and expression machinery a real compiler backend would consume:
`elements/finite_element.py`'s `lagrange_element`/`raviart_thomas_element`/`nedelec_element`/`bdm_element`/
`mixed_element` (per the ledger's `MP-D2` row, merged) has no general residual/Jacobian assembler behind it --
`mixle-sim/assembly.py` only assembles scalar P1 diffusion (`MP-D6`, "partial"). There is no H(div)/H(curl)
assembly, no higher-order or mixed-space assembly, and no DG flux/stabilization library (`MP-D7`, not-started)
anywhere in the program.

The work plan states the intended resolution directly: *"FEniCSx components—UFL, Basix, FFCx, DOLFINx—are the
primary production FEM path instead of recreating a full form compiler and element library."* Its risk table is
explicit about the alternative: *"Rebuilding mature FEM/HPC internals | years of avoidable work | use
FEniCSx/PETSc; native backend stays a reference subset."* Building a general higher-order/mixed/H(div)/H(curl)
form compiler and assembler by hand, on top of the element library `mixle-sim` already has, is exactly the kind
of multi-year internals rebuild that quote warns against.

**Cost.** DOLFINx's standard distribution links against PETSc for its core `Vec`/`Mat` linear-algebra types and
always runs under an MPI communicator (size 1 in the serial case) -- it is not a lightweight `pip install`
alongside this repo's existing extras; realistic installs use conda-forge or a container image. This cost is
inseparable from the PETSc and MPI decisions below: adopting DOLFINx necessarily pulls in `petsc4py` and an MPI
implementation as DOLFINx's own internal dependencies, not as an independent choice this repo makes twice.

**Sequencing.** The work plan places this adoption after the weak-form IR lands: *"MP-D1–D6, MP-E1–E4, and
MP-G6: trial/test spaces, weak forms, BCs, and typed physical interfaces compiled to native and FEniCSx/PETSc."*
The ledger's `MP-D1` row confirms the physics→math elaboration bridge (`mixle_physics/elaborate.py`) exists only
on an open, unmerged PR (`#8`, `915f059`) -- not landed. There is not yet a typed form IR for a compiler backend
to lower from.

**Decision: Adopt**, sequenced explicitly behind `MP-D1` landing (the elaboration bridge) and a minimally
sufficient slice of `MP-D2`–`D6` (trial/test spaces through assembly) on the `mixle-sim` side. This is a real
"build this" verdict, not a deferral of the decision itself -- the capability gap is large, the work plan commits
to this path unconditionally, and the alternative (hand-rolling a form compiler) is explicitly named as years of
avoidable work. The gate is purely about *order*: there is nothing yet for a DOLFINx/FFCx adapter to compile
against. First follow-up task once `MP-D1` lands: a `probe()`/`capabilities()`-only adapter skeleton behind a
`fem` extra (already reserved in `pyproject.toml`), reporting cleanly unavailable when the extra is absent,
before any `lower`/`execute`/`normalize` wiring is attempted.

## PETSc

**Capability gap.** `mixle_pde/linear_solve.py` is single-process SciPy: `spd_solve` offers CG/MINRES with a
`pyamg` smoothed-aggregation AMG preconditioner (falling back to SciPy `spilu` incomplete-LU when `pyamg` is
absent) and a cached sparse-LU (`splu`) fallback when the iterative solve stalls; `dense_spd_solve` replaces
`np.linalg.inv` with a cached Cholesky factor for the small/medium dense-covariance paths. There is no
distributed-memory solve anywhere -- confirmed by direct grep (no `mpi4py`, `multiprocessing`, or
`concurrent.futures` usage under `mixle_pde/`; see the MPI section below for the exact repo-audited quote).
PETSc's marginal value here is specifically **distributed-memory** parallelism, not serial AMG preconditioning
-- that capability already exists via `pyamg`. Beyond linear solves, PETSc's `SNES` (Newton-Krylov with
globalization/line-search/trust-region), `TS` (adaptive implicit/IMEX time integration), and `TAO`
(optimization) would each close real gaps: the ledger's `MP-F2` row shows `mixle_pde/nonlinear.py`'s Newton
continuation (`MP-F2 baseline`, PR #97, `41dde11`) is a hand-rolled solver, not `SNES`; `MP-F3` records every
registered time-stepping kernel in `pde_backend_registry.py` as explicit-only (leapfrog/FDTD/FD), with no
implicit RK/BDF/adaptive-step framework anywhere in the program; `mixle_pde/field_gauss_newton.py` is a
hand-rolled Gauss-Newton optimizer, not `TAO`. Separately, `mixle-discrete`'s `mixle_numerics/linear_solvers.py`
(ledger `MP-F1`, merged) already generalizes solver-family selection (`SolverFamily{DIRECT, CG, MINRES, GMRES,
BICGSTAB, AMG}`, `PreconditionerFamily{... SCHWARZ, FIELD_SPLIT, SCHUR_COMPLEMENT, AMG}`) behind a
backend/assembly-agnostic contract, by that module's own docstring -- any PETSc adapter should extend that
existing cross-repo abstraction, not build a second, competing one inside `mixle-pde`.

**Ownership note.** `mixle_pde/ownership.py` classifies `linear_solve` as `_CORE_MODULES`: disposition
`"migrate"`, final owner `"PRJ-CORE"`, rationale *"Generic inference, value, posterior, or uncertainty semantics
belong to Core; PDE retains compatibility until migration gates pass."* This repo's own solver stack is already
committed to migrating out of `mixle-pde`. Building a `mixle-pde`-local `petsc4py` bridge on top of
`linear_solve.py` today would become dead weight the moment that already-recorded migration happens; a PETSc
bridge belongs behind `mixle-discrete`'s already-generic `linear_solvers.py` contract or the `PRJ-CORE` migration
target, not as new `mixle-pde`-local code.

**Cost.** `petsc4py` is a substantially heavier dependency than anything else in this repo's extras: it requires
a working PETSc build (BLAS/LAPACK, and, transitively, an MPI implementation) rather than a self-contained
wheel, which is why the work plan treats its platform matrix as a named risk (*"MPI/GPU nondeterminism |
irreproducible results | tolerance-aware deterministic receipts and backend manifests"*).

**Decision: Defer** the standalone scope the ledger's `MP-E3` row names -- *"Map meshes/function spaces to
DM/DMPlex, matrices/vectors to PETSc, field splits/nullspaces/options prefixes"* for `mixle-pde`'s own
hand-assembled operators, independent of DOLFINx, plus freestanding `SNES`/`TS`/`TAO` usage. This is distinct
from DOLFINx's bundled, internal use of `petsc4py` (authorized above as a consequence of the DOLFINx decision,
not a separate one). Trigger for the standalone scope: (a) a concrete workload's memory or wall-time exceeds
what the existing serial CG+AMG path handles -- `linear_solve.py`'s own module docstring sizes its motivating
problem at a `50^3 = 125,000`-cell normal-equations system (~100 GB if formed densely, which is exactly what
that module exists to avoid); no case in this repo today exceeds what the sparse serial path already handles --
or (b) `mixle-sim`'s `MP-L1` (MPI mesh partitioning and distributed assembly, not-started) lands and produces an
actually-distributed matrix for a `DM`/`DMPlex` bridge to receive, whichever comes first. Consequence, stated
plainly rather than left implicit: the work plan gates production-FEM claims on this decision (*"If MP-A4 cannot
establish a supported FEniCSx/PETSc deployment on the target platform matrix, pause production-FEM claims and
choose a replacement backend before building physics packs"*) -- deferring the standalone PETSc scope means
those claims stay paused until one of the two triggers above fires, and that consequence is accepted here, not
hidden.

## MPI (`mpi4py`)

**Capability gap.** There is none today, and this repo's own tooling says so directly.
`mixle_pde/verification/capability_inventory.py`'s methodology note: *"`parallel_status` is `"single_process"`
for every entry: a repo-wide sweep found no `mpi4py`, no `multiprocessing`, and no
`concurrent.futures`/`joblib` usage anywhere under `mixle_pde/`."* `mixle_pde/mcmc_checkpoint.py` (`MP-I9`, PR
#99) confirms the same boundary from the other direction: it implements checkpoint/restart for a single chain
and explicitly does **not** implement *"MPI/multiprocessing/distributed execution, counter-based (splittable)
random streams, or checkpointing for any sampler other than `metropolis_field_invert`."* `pyproject.toml`
already reserves an `mpi` extra (`mpi4py`), but its own comment records the same gap: *"mpi4py is forward-
declared for the distributed mesh/solve transport described in the architecture plan; no mixle_pde module
imports it yet, so this extra currently installs a dependency with no wired call site rather than unlocking new
behavior."*

**Scope correction.** The work plan assigns embarrassingly-parallel execution -- *"MPI/container launch,
multi-chain/particle/ensemble execution"* -- to `mixle-mlops` integration (line 1089), consistent with the
program README's portfolio routing: *"mixle-mlops: durable jobs, registry, deployment, monitoring and rollback
without owning domain semantics."* The one class of workload this repo actually has that could use parallelism
today -- `marginal_std_cg`'s Hutchinson probes (`linear_solve.py`, independent per-probe solves) and multi-chain
MCMC (`MP-I8`, `mixle_pde/field_mcmc.py`) -- is same-machine, embarrassingly parallel, and does not need MPI's
distributed-memory model at all; stdlib `concurrent.futures`/`multiprocessing` (zero extra dependency, also
currently unused everywhere in this repo) or `mixle-mlops`' job orchestration are both strictly cheaper fits
than hand-rolling `mpi4py` calls inside `mixle-pde` for that workload class.

**Decision: Defer** standalone, hand-rolled `mpi4py` use inside `mixle-pde` -- route embarrassingly-parallel
ensemble/multi-chain workloads to `mixle-mlops` or stdlib `concurrent.futures` first. The `mpi` extra's real
justification is as PETSc's (and, transitively, DOLFINx's) internal transport dependency, not a standalone
`mixle-pde` capability; it activates as an automatic consequence of the DOLFINx/PETSc decisions above, on their
triggers, not on a separate one. This record does not change `pyproject.toml` (that extra already exists and
already correctly ships zero call sites), but a future docs-only pass should tighten that comment to say
"reserved for petsc4py's transitive dependency" rather than "distributed mesh/solve transport," so the next
reader does not conclude `mixle-pde` itself is expected to call `mpi4py` APIs directly.

## preCICE

**Capability gap.** `mixle_pde/multiphysics.py`'s `CoupledPDESystem`/`run_coupled` already does in-process,
monolithic, node-local block coupling of several fields on a shared structured grid (thermo-elastic-style
exchange, reaction-diffusion between species) -- this is a real, working "minimal in-process implementation,"
not a placeholder. `mixle-sim/programs.py`'s `CouplingPlan`/`PortEndpoint`/`TransferPlan`/`CouplingStrategy`
(`MONOLITHIC`/`PARTITIONED` tags) are validated data contracts only, per the ledger's `MP-G1`/`MP-G2`/`MP-G4`
rows -- no execution engine, no fixed-point/quasi-Newton iteration loop, and no interpolation/projection numerics
exist anywhere in the program yet. preCICE's actual value-add is different in kind from what either module has:
black-box coupling of *independently executed* participant codes, nonmatching-mesh mapping, and
explicit/implicit serial/parallel coupling schemes with Aitken/quasi-Newton acceleration -- the work plan's own
framing (*"preCICE separates participant communication, nonmatching-mesh mapping, coupling scheme, convergence,
and acceleration"*).

**Decision gate.** The work plan is explicit that this task is the gate: *"A partitioned-coupling adapter is
added after evaluation of preCICE versus a minimal in-process implementation. No coupling dependency is adopted
before the MP-A4 decision gate passes."*

**Cost / missing referent.** preCICE's entire value proposition is coordinating two or more *separate* solver
processes. `mixle-pde` has no second participant to couple to today: every multiphysics coupling in this repo is
same-process Python function calls inside `CoupledPDESystem`. OpenFOAM (a plausible second participant) is
itself deferred below; DOLFINx is not yet adopted; nothing in this program currently runs as a second process
that preCICE would have anything to broker between.

**Decision: Defer.** Trigger: a second, separately-executed participant solver is actually adopted (most likely
OpenFOAM, once its own trigger below fires, or an external non-Python code). Adopting preCICE before that would
be infrastructure with no consumer -- precisely the "orphaned capability" failure mode
`docs/reconciliation/mp-task-ledger.md`'s own acceptance bar is written to catch (*"No orphaned capability and
no duplicate source of truth"*). Until the trigger fires, `CoupledPDESystem`'s monolithic in-process coupling
remains the right tool for every coupling case this repo actually has.

## OpenFOAM

**Capability gap.** `mixle_pde/flow.py` (`NavierStokes2D`, FD streamfunction-vorticity) and
`mixle_pde/spectral_flow.py` (`incompressible_ns_spectral`, periodic pseudo-spectral RK4, *"with an optional
Smagorinsky LES closure"* per its own module docstring) already cover basic incompressible flow, including a
first-order LES capability -- so "turbulence modeling" in the abstract is not a clean gap. What is genuinely
missing is unstructured-mesh, wall-bounded RANS/LES, industrial compressible/multiphase flow, and sliding/
rotating mesh support, none of which a periodic-box spectral method or a 2-D structured FD stepper can express.
`mixle_pde/gas_dynamics.py` states its own boundary directly: its zero-D reactive-chamber model *"is a fast
simulator kernel for engine-cylinder and explosion studies, not a substitute for turbulent CFD or detailed
chemistry."* `mixle_pde/capabilities.py`'s matching limitation entry for that module: *"combustion uses one-step
Arrhenius chemistry; no turbulent flame fronts, detonation, or detailed reaction mechanisms yet."* These are the
repo's own admissions of exactly the gap OpenFOAM's turbulence and combustion libraries would close.

**Decision gate.** The work plan names OpenFOAM as *"the first candidate, selected only after an ADR and
license/deployment review"* for a finite-volume backend adapter -- this section is that review.

**License.** OpenFOAM (both the OpenFOAM Foundation and ESI/OpenCFD distributions) is **GPL-3**. This is
categorically different from every other dependency named in this record or already present in `mixle-pde`'s
`all` extra (`numpy`/`scipy`/`torch`/`scikit-learn`/`pyamg`/`scikit-sparse`/`mpi4py` are all permissive or
LGPL-family) and from `mixle-discrete`'s own ADR-0001 precedent (FLINT, LGPL-2.1, "compatible-with-obligations"
as a dynamically-linked optional extra). GPL-3's copyleft terms make *linking* OpenFOAM's solver code
unworkable for an MIT-licensed package; the only defensible integration pattern is the
**sandboxed-subprocess** kind classified above -- driving unmodified OpenFOAM solver binaries through their own
case-directory file protocol, the standard "mere aggregation" pattern that keeps a GPL binary at arm's length.
That in turn means a real, nontrivial cost this record has not scoped: OpenFOAM's dictionary-file case format is
its own protocol, and per the isolation discipline `DISC-D0.9` documents for this exact backend kind, the
adapter must own its emitter/parser and never lean on parsing OpenFOAM's pretty-printed output when a
machine-readable path exists.

**Cost, beyond licensing.** A subprocess adapter needs a mesh to hand OpenFOAM, and CAD/mesh interchange
(`MP-C4`) is not-started anywhere in the program (see the Gmsh section) -- OpenFOAM's `MP-A4` review depends on
the same missing prerequisite Gmsh does.

**Decision: Defer.** Trigger: a concrete study needs turbulence closure or unstructured-mesh compressible/
multiphase flow beyond what `flow.py`/`spectral_flow.py`/`gas_dynamics.py` can serve. Even once that trigger
fires, adoption must (a) use the sandboxed-subprocess pattern exclusively, never linking, and (b) pass its own
follow-up license/deployment ADR addendum scoping the case-directory emitter/parser and redistribution
obligations in more depth than this survey-level pass does -- mirroring how `DISC-D0.9`'s FLINT worked example
treated its own (much lighter) LGPL obligations as a named, separately-tracked legal item rather than a box to
check once.

## Decision summary

| Backend | Kind | Verdict | Trigger / condition | Extras slot |
|---|---|---|---|---|
| Gmsh + OpenCASCADE | in-process-library | Defer -- route to `mixle-sim` `MP-C5` | `mixle-pde` adopts the Sim mesh IR *and* a CAD-derived study exists | `mesh` (reserved, unused) |
| DOLFINx/FFCx | in-process-library | **Adopt**, sequenced after `MP-D1` | `MP-D1` elaboration bridge + minimal `MP-D2`-`D6` land | `fem` (reserved, unused) |
| PETSc (standalone `DM`/`DMPlex`/`SNES`/`TS`/`TAO`) | in-process-library | Defer | workload exceeds serial CG+AMG, or `MP-L1` lands | `mpi` (reserved, unused) |
| PETSc (bundled inside DOLFINx) | in-process-library | Authorized as a DOLFINx consequence | same as DOLFINx | `fem` |
| MPI (`mpi4py`, standalone) | in-process-library | Defer -- route to `mixle-mlops`/stdlib | same as PETSc standalone | `mpi` (reserved, unused) |
| preCICE | in-process-library, coordinates separate processes | Defer | a second, separately-executed participant solver is adopted | `coupling` (reserved, unused) |
| OpenFOAM | sandboxed-subprocess | Defer | turbulence/unstructured-flow study exists; then a license/deployment ADR addendum | `fvm` (reserved, unused) |

## Consequences

- No `pyproject.toml`, code, or capability-inventory change is made by this record. Every extras slot named
  above already exists (`MP-A5`, PR #69, `d3580f7`) with zero call sites; adopting this record's verdicts
  introduces no new dependency by itself.
- Per the work plan's own gate, deferring PETSc's standalone scope means production-FEM claims stay paused until
  one of its two named triggers fires -- stated here explicitly so it is not rediscovered as a surprise later.
- The one unconditional "build this" verdict (DOLFINx/FFCx) is sequenced, not immediate: its first authorized
  follow-up task is gated on `MP-D1` landing, not on this ADR merging.

## Risks and open questions

- DOLFINx's transitive PETSc/MPI dependency means the DOLFINx and PETSc verdicts must be re-read together, not
  independently -- a future implementer must not read "Defer PETSc" in isolation and conclude DOLFINx can be
  adopted without it.
- If OpenFOAM's trigger fires, its case-directory emitter/parser is sizable, unstarted work in its own right,
  not a short follow-up -- do not schedule it as a small task once the trigger condition is met.
- These verdicts are grounded in the 2026-07-16 ledger and work-plan state. If a dependent task (`MP-C1` mesh-IR
  adoption, `MP-D1` elaboration bridge, `MP-L1` distributed assembly) lands, the corresponding trigger above
  should be re-checked against the ledger at that time rather than assumed to still hold.

## Reviewer checklist

- [x] Every backend section cites specific, checkable current-repo evidence (module, function, or docstring
      quote) for its capability-gap claim, not general knowledge about the backend.
- [x] Every backend section states a license or packaging cost grounded in this repo's existing extras
      convention or a named source document.
- [x] Every backend section ends with an explicit Adopt / Defer-until-X / Never verdict; none is a hedge.
- [x] The decision-summary table's verdicts match the per-section decisions exactly.
- [x] No `pyproject.toml`, code, or capability-inventory change is required to merge this record.
- [x] This document follows the attribution convention in `notes/parallel-implementation-tasks/README.md`
      (describes the change itself; no tool or vendor byline), aside from citing the pre-existing repository
      file `notes/mixle-pde-ai-native-multiphysics-work-plan.md` by its real name.
