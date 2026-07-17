"""Geometric conservation law (GCL) violation detector for moving simplex meshes (MP-C8 remainder).

The M2 reconciliation ledger (``docs/reconciliation/mp-task-ledger.md``, MP-C8) records
``mixle_pde/mesh.py::MovingSimplexMesh``/``moving_mesh``/``pipe_radial_deformation`` as real,
legacy-only moving-domain geometry, and states plainly: "No geometric-conservation-law check found
anywhere." ``notes/mixle-pde-ai-native-multiphysics-work-plan.md``'s MP-C8 acceptance bar is "a piston
or deforming-channel case moves, remeshes, transfers state, and satisfies the geometric conservation
law; no ALE/FSI claim before this passes." This module fills exactly the GCL-check half of that gap --
quality-triggered remeshing, field transfer, and restart remain separate, not-yet-started MP-C8 scope
and are not claimed here.

What the geometric conservation law is. For a control volume (cell) whose boundary moves in time, the
GCL is the statement that two independently computable quantities must agree exactly:

* the cell's volume/area *rate of change*, computed purely from the velocity of its boundary/nodes
  (the quantity an ALE/moving-mesh solver's flux terms actually use), and
* the cell's *actual* volume/area change, read directly off its geometry at the two time endpoints
  (no velocity involved at all).

A solver whose mesh-velocity field is inconsistent with how its mesh nodes actually move violates this
identity -- a classic, well-documented source of spurious mass/energy generation in ALE codes (Thomas &
Lombard 1979; Farhat, Geuzaine & Grandmont 2001, "The discrete geometric conservation law and the
nonlinear stability of ALE schemes").

How this module computes each side, independently, for a simplex cell
-----------------------------------------------------------------------
Let ``E(t)`` be the ``dim x dim`` matrix of edge vectors from a simplex's first vertex to its other
``dim`` vertices, so the simplex's signed volume is ``V(t) = det(E(t)) / dim!``. Between two recorded
time samples, :class:`~mixle_pde.mesh.MovingSimplexMesh` moves every node in a straight line (the same
linear interpolation :meth:`~mixle_pde.mesh.MovingSimplexMesh.at_time` already performs), so each node's
velocity is constant over the interval and ``E(t)`` is affine in ``t``.

* **Flux/velocity side** (``GCLCellCheck.flux_volume_rate``): by Jacobi's formula for the derivative of
  a determinant, the *instantaneous* signed-volume rate at any instant is
  ``dV/dt(t) = det(E(t)) / dim! * trace(E(t)^-1 dE/dt)``, where ``dE/dt`` is the matrix of *edge*
  velocities (vertex velocities differenced the same way as the edge vectors themselves). This is the
  same physical quantity a boundary-flux integral ``oint (v . n) dA`` would give (both are restatements
  of the same divergence-theorem identity for a simplex with affinely-interpolated velocity across each
  face) but needs no facet-normal bookkeeping to evaluate. Integrating ``dV/dt(t)`` exactly over
  ``[t_n, t_{n+1}]`` gives this side; because ``E(t)`` is affine in ``t``, ``dV/dt(t)`` is a polynomial
  in ``t`` of degree at most ``dim - 1``, and Simpson's rule (exact through degree 3) integrates it
  exactly for every ``dim <= 4`` this repo's mesh module supports (``mixle_pde/mesh.py``'s own docstring:
  a 3-D tetrahedral mesh extruded through time becomes a 4-D mesh of pentachora -- ``dim = 4`` is the
  documented ceiling). This module evaluates the integrand at the interval's two endpoints and midpoint
  and combines them with the standard Simpson weights.
* **Direct side** (``GCLCellCheck.direct_volume_change``): simply ``V(t_{n+1}) - V(t_n)``, the signed
  measure difference read directly from the stored node positions at the two time samples. No velocity,
  derivative, or quadrature is involved.

When ``node_velocity`` is left at its default (``None``), it is derived by finite-differencing the
*same* stored positions used for the direct side (``(nodes[n+1] - nodes[n]) / dt``) -- the actual,
literal velocity implied by the recorded motion. Substituting that into the flux side above and
integrating makes the flux side mathematically identical to the direct side (both sides are then just
two different ways of writing the same polynomial's total change), so any nonzero discrepancy in that
default configuration reflects only floating-point round-off, confirmed empirically at the ~1e-16
relative level across the 3-D box-stretch, annular-pipe, and 2-D shear scenarios this module's tests
reuse from ``tests/mesh_test.py``. Passing an explicit ``node_velocity`` lets a caller check whether
*that* velocity field -- e.g. one a solver actually used for its flux terms, which may have been
computed by a different rule, prescribed analytically, or simply be wrong -- is geometrically consistent
with how the mesh really moved; a deliberately inconsistent field (see
``tests/geometric_conservation_test.py``) produces an O(1) relative discrepancy, not noise.

Honesty notes
--------------
* This module only checks the GCL identity on a given node-position time series; it does not perform
  mesh motion, remeshing, field transfer, or restart, and it makes no ALE/FSI capability claim by
  itself -- see the MP-C8 acceptance bar quoted above.
* A cell whose quality (:meth:`~mixle_pde.mesh.SimplexMesh.simplex_quality`) or absolute measure is
  below ``min_quality``/``min_measure`` at either time endpoint, or whose edge matrix is exactly
  singular at any quadrature point, is reported ``UNKNOWN`` rather than a guessed PASS or FAIL -- the
  GCL identity compares two rates of change of a volume that is itself near-vanishing or ill-conditioned
  there, so neither side is numerically trustworthy. ``min_quality``/``min_measure`` default to exactly
  the thresholds :meth:`~mixle_pde.mesh.SimplexMesh.validate`/:meth:`~mixle_pde.mesh.MovingSimplexMesh.
  validate` already use, rather than inventing new ones.
* Does not touch ``mixle_pde/io/artifacts.py``, ``mixle_pde/verification/capability_inventory.py``, or
  ``mixle_pde/pde_backend_registry.py`` -- registering this module in the frozen capability inventory is
  left as an explicit follow-up, matching the same exclusion recent sibling PRs (#88, #89, #90, #93,
  #97, #100, #107) used for the same contended files.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

from mixle_pde.mesh import MovingSimplexMesh, SimplexMesh

__all__ = [
    "GCLVerdict",
    "GCLCellCheck",
    "GCLStepReceipt",
    "GCLReport",
    "check_geometric_conservation_law",
]


class GCLVerdict(str, Enum):
    """A typed pass/fail/unknown verdict -- never a fabricated boolean when a check could not run."""

    PASS = "pass"
    FAIL = "fail"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class GCLCellCheck:
    """One simplex's independently-computed GCL identity over one ``[time_start, time_end]`` interval.

    ``flux_volume_rate`` and ``direct_volume_change`` are the two sides of the identity described in
    this module's docstring, computed by entirely separate code paths (a velocity-driven quadrature vs.
    a bare difference of stored-geometry measures); ``discrepancy`` is their difference, never hidden
    even when ``verdict`` is UNKNOWN (in which case it is ``nan``, not a fabricated zero).
    """

    cell_index: int
    flux_volume_rate: float
    direct_volume_change: float
    discrepancy: float
    quality_start: float
    quality_end: float
    verdict: GCLVerdict
    detail: str


@dataclass(frozen=True)
class GCLStepReceipt:
    """The GCL verdict for every cell over one ``[time_start, time_end]`` interval.

    ``verdict`` is FAIL if any cell FAILs, else UNKNOWN if any cell is UNKNOWN (and none FAILed), else
    PASS -- summing or averaging per-cell discrepancies before judging would let a positive discrepancy
    in one cell cancel a negative one in another and hide a real per-cell violation, so this receipt
    never does that; every cell's own verdict is preserved on ``cells``.
    """

    step_index: int
    time_start: float
    time_end: float
    cells: tuple[GCLCellCheck, ...]
    verdict: GCLVerdict
    detail: str


@dataclass(frozen=True)
class GCLReport:
    """The full typed GCL verdict for a :class:`~mixle_pde.mesh.MovingSimplexMesh`'s entire time series.

    ``verdict`` rolls up every :class:`GCLStepReceipt` the same way each receipt rolls up its cells
    (FAIL beats UNKNOWN beats PASS); a moving mesh with fewer than two recorded time steps has no
    interval to check at all and is reported UNKNOWN with an empty ``steps``, never a vacuous PASS.
    """

    dim: int
    n_simplices: int
    n_steps: int
    steps: tuple[GCLStepReceipt, ...]
    verdict: GCLVerdict
    detail: str


def _signed_measure(coords: np.ndarray) -> float:
    """Signed measure of one simplex given its ``(dim + 1, dim)`` vertex coordinates."""
    dim = coords.shape[1]
    edges = coords[1:] - coords[0]
    return float(np.linalg.det(edges) / math.factorial(dim))


def _resolve_velocity(
    node_velocity: Any,
    nodes0: np.ndarray,
    nodes1: np.ndarray,
    dt: float,
    step: int,
    n_intervals: int,
) -> np.ndarray:
    """Return the constant-over-this-interval ``(n_nodes, dim)`` node velocity to check.

    ``None`` derives the actual finite-difference velocity implied by the stored motion (the
    self-consistent default -- see the module docstring). An explicit array is accepted either as one
    velocity applied to every interval (shape matching ``nodes0``) or one velocity per interval (a
    leading axis of size ``n_intervals``), mirroring :func:`mixle_pde.mesh.moving_mesh`'s own
    ``displacement`` shape convention.
    """
    if node_velocity is None:
        return (nodes1 - nodes0) / dt
    velocity = np.asarray(node_velocity, dtype=float)
    if velocity.shape == nodes0.shape:
        return velocity
    if velocity.shape == (n_intervals, *nodes0.shape):
        return velocity[step]
    raise ValueError(
        f"node_velocity must have shape {nodes0.shape} (one velocity field for every interval) or "
        f"{(n_intervals, *nodes0.shape)} (one velocity field per interval); got {velocity.shape}."
    )


def _check_cell(
    cell_index: int,
    simplex: np.ndarray,
    nodes0: np.ndarray,
    nodes1: np.ndarray,
    velocity: np.ndarray,
    dt: float,
    quality0: float,
    quality1: float,
    *,
    rtol: float,
    atol: float,
    min_quality: float,
    min_measure: float,
) -> GCLCellCheck:
    x0 = nodes0[simplex]
    x1 = nodes1[simplex]
    measure0 = _signed_measure(x0)
    measure1 = _signed_measure(x1)
    direct_volume_change = measure1 - measure0

    if quality0 < min_quality or quality1 < min_quality or abs(measure0) < min_measure or abs(measure1) < min_measure:
        return GCLCellCheck(
            cell_index=cell_index,
            flux_volume_rate=float("nan"),
            direct_volume_change=float(direct_volume_change),
            discrepancy=float("nan"),
            quality_start=quality0,
            quality_end=quality1,
            verdict=GCLVerdict.UNKNOWN,
            detail=(
                f"cell {cell_index}: near-degenerate at an endpoint (quality {quality0:.3g}/"
                f"{quality1:.3g}, min_quality={min_quality:g}); the GCL identity is not numerically "
                "meaningful for a near-vanishing simplex."
            ),
        )

    v = velocity[simplex]
    e0 = x0[1:] - x0[0]
    e1 = x1[1:] - x1[0]
    de_dt = v[1:] - v[0]
    dim = e0.shape[0]

    def _instantaneous_rate(weight: float) -> float | None:
        edges = e0 + weight * (e1 - e0)
        try:
            solved = np.linalg.solve(edges, de_dt)
        except np.linalg.LinAlgError:
            return None
        return float(np.linalg.det(edges) / math.factorial(dim) * np.trace(solved))

    rates = [_instantaneous_rate(0.0), _instantaneous_rate(0.5), _instantaneous_rate(1.0)]
    if any(rate is None for rate in rates):
        return GCLCellCheck(
            cell_index=cell_index,
            flux_volume_rate=float("nan"),
            direct_volume_change=float(direct_volume_change),
            discrepancy=float("nan"),
            quality_start=quality0,
            quality_end=quality1,
            verdict=GCLVerdict.UNKNOWN,
            detail=f"cell {cell_index}: edge matrix exactly singular mid-interval; cannot evaluate the flux quadrature.",
        )

    # Simpson's rule, exact for the degree-<=3-in-t integrand every dim<=4 this repo supports produces.
    flux_volume_rate = (dt / 6.0) * (rates[0] + 4.0 * rates[1] + rates[2])
    discrepancy = flux_volume_rate - direct_volume_change
    scale = max(abs(flux_volume_rate), abs(direct_volume_change))
    tolerance = atol + rtol * scale
    passed = abs(discrepancy) <= tolerance
    verdict = GCLVerdict.PASS if passed else GCLVerdict.FAIL
    detail = (
        f"cell {cell_index}: flux-integrated rate {flux_volume_rate:.6g} vs direct change "
        f"{direct_volume_change:.6g} (discrepancy {discrepancy:.3g}, tolerance {tolerance:.3g}) "
        f"-> {verdict.value.upper()}"
    )
    return GCLCellCheck(
        cell_index=cell_index,
        flux_volume_rate=flux_volume_rate,
        direct_volume_change=float(direct_volume_change),
        discrepancy=discrepancy,
        quality_start=quality0,
        quality_end=quality1,
        verdict=verdict,
        detail=detail,
    )


def _rollup(verdicts: set[GCLVerdict]) -> GCLVerdict:
    if GCLVerdict.FAIL in verdicts:
        return GCLVerdict.FAIL
    if GCLVerdict.UNKNOWN in verdicts:
        return GCLVerdict.UNKNOWN
    return GCLVerdict.PASS


def check_geometric_conservation_law(
    moving: MovingSimplexMesh,
    *,
    node_velocity: np.ndarray | None = None,
    rtol: float = 1e-6,
    atol: float = 1e-9,
    min_quality: float = 1.0e-8,
    min_measure: float = 1.0e-14,
) -> GCLReport:
    """Independently compute both sides of the GCL identity for every cell and interval, and verdict.

    ``node_velocity`` defaults to the finite-difference velocity implied by ``moving``'s own stored
    node positions (the self-consistent case, expected to PASS to floating-point round-off). Pass an
    explicit array -- shaped like one node-velocity field applied to every interval, or one per
    interval -- to check whether *that* velocity field is geometrically consistent with the mesh's
    actual recorded motion; see the module docstring. ``min_quality``/``min_measure`` reuse
    :meth:`~mixle_pde.mesh.SimplexMesh.validate`'s own default thresholds. Never raises on degenerate
    *data* (near-vanishing cells, too few time steps) -- those produce an UNKNOWN verdict instead;
    invalid *call* arguments (a wrongly shaped ``node_velocity``, a negative tolerance) still raise
    ``ValueError`` immediately.
    """
    if rtol < 0.0 or atol < 0.0:
        raise ValueError("rtol and atol must both be >= 0.")
    if min_quality < 0.0 or min_measure < 0.0:
        raise ValueError("min_quality and min_measure must both be >= 0.")

    n_intervals = moving.n_steps - 1
    if n_intervals < 1:
        return GCLReport(
            dim=moving.dim,
            n_simplices=moving.n_simplices,
            n_steps=moving.n_steps,
            steps=(),
            verdict=GCLVerdict.UNKNOWN,
            detail=f"moving mesh has only {moving.n_steps} recorded time step(s); no interval to check.",
        )

    steps: list[GCLStepReceipt] = []
    for step in range(n_intervals):
        nodes0 = moving.nodes_over_time[step]
        nodes1 = moving.nodes_over_time[step + 1]
        time_start = float(moving.times[step])
        time_end = float(moving.times[step + 1])
        dt = time_end - time_start

        if not (np.all(np.isfinite(nodes0)) and np.all(np.isfinite(nodes1))):
            steps.append(
                GCLStepReceipt(
                    step_index=step,
                    time_start=time_start,
                    time_end=time_end,
                    cells=(),
                    verdict=GCLVerdict.UNKNOWN,
                    detail=f"step {step}: non-finite node positions; cannot evaluate the GCL identity.",
                )
            )
            continue

        velocity = _resolve_velocity(node_velocity, nodes0, nodes1, dt, step, n_intervals)
        quality0 = SimplexMesh(nodes0, moving.simplices).simplex_quality()
        quality1 = SimplexMesh(nodes1, moving.simplices).simplex_quality()

        cells = tuple(
            _check_cell(
                cell_index,
                simplex,
                nodes0,
                nodes1,
                velocity,
                dt,
                float(quality0[cell_index]),
                float(quality1[cell_index]),
                rtol=rtol,
                atol=atol,
                min_quality=min_quality,
                min_measure=min_measure,
            )
            for cell_index, simplex in enumerate(moving.simplices)
        )
        step_verdict = _rollup({cell.verdict for cell in cells})
        n_fail = sum(1 for cell in cells if cell.verdict is GCLVerdict.FAIL)
        n_unknown = sum(1 for cell in cells if cell.verdict is GCLVerdict.UNKNOWN)
        finite_discrepancies = [abs(cell.discrepancy) for cell in cells if math.isfinite(cell.discrepancy)]
        max_discrepancy = max(finite_discrepancies) if finite_discrepancies else float("nan")
        steps.append(
            GCLStepReceipt(
                step_index=step,
                time_start=time_start,
                time_end=time_end,
                cells=cells,
                verdict=step_verdict,
                detail=(
                    f"step {step} [{time_start:g}, {time_end:g}]: {len(cells)} cell(s), {n_fail} "
                    f"failing, {n_unknown} unknown, max finite |discrepancy| {max_discrepancy:.3g} "
                    f"-> {step_verdict.value.upper()}"
                ),
            )
        )

    overall = _rollup({step.verdict for step in steps})
    n_fail_steps = sum(1 for step in steps if step.verdict is GCLVerdict.FAIL)
    n_unknown_steps = sum(1 for step in steps if step.verdict is GCLVerdict.UNKNOWN)
    detail = (
        f"{moving.dim}-D moving mesh, {moving.n_simplices} simplices, {len(steps)} interval(s): "
        f"{n_fail_steps} failing, {n_unknown_steps} unknown -> {overall.value.upper()}"
    )
    return GCLReport(
        dim=moving.dim,
        n_simplices=moving.n_simplices,
        n_steps=moving.n_steps,
        steps=tuple(steps),
        verdict=overall,
        detail=detail,
    )
