"""Tests for MP-K2's baseline backend-neutral result queries (point probe / integrate / extrema).

Acceptance criteria under test:

* Point probes, domain integrals, and extrema are computed on the mesh/grid a completed solve
  already carries -- no new mesh or quadrature representation is invented.
* Every query result carries the field's declared unit, read from the originating backend/receipt
  (``None`` when the source result genuinely carries no unit metadata, e.g. the canonical-adapter
  Poisson path).
* Cross-backend agreement (MP-K2's acceptance bar): the SAME point probe / integral / extrema query
  against results from two different already-registered solver kernels -- the native-rational-linear
  backend (:mod:`mixle_pde.canonical_adapter`) and the ``fem-p1-simplex`` backend
  (:mod:`mixle_pde.pde_backend_registry`) -- agree within a documented tolerance, including against
  a known closed-form analytic solution.
"""

from __future__ import annotations

import numpy as np
import pytest

from mixle_pde.canonical_adapter import solve_p1_poisson_canonical
from mixle_pde.dynamics import AdvectionDiffusionOperator
from mixle_pde.fem import solve_simplex_poisson
from mixle_pde.mesh import box_simplex_mesh
from mixle_pde.pde_backend_registry import run_math_problem
from mixle_pde.verification.result_queries import (
    FieldSample,
    extrema,
    field_from_canonical_poisson,
    field_from_kernel_study,
    integrate,
    point_probe,
)

# Two independent code paths solving the same discretized system agree far tighter than this in
# practice (see test_native_and_fem_backends_agree_to_near_machine_precision), but the tolerance
# documented for the acceptance criterion is deliberately generous: it is a P1-discretization-level
# agreement bound, not a claim that these two adapters are byte-identical implementations.
CROSS_BACKEND_TOLERANCE = 1e-8


def _fem_poisson_problem(
    *, shape: tuple[int, int], lengths: tuple[float, float], diffusion: float, source: float
) -> dict:
    return {
        "id": "result-queries-poisson-study",
        "domains": [{"id": "domain", "kind": "mesh", "properties": {"mesh_cell_type": "triangle"}}],
        "unknowns": [{"id": "field", "domain_id": "domain"}],
        "operators": [
            {
                "id": "op",
                "kind": "weak_form",
                "input_ids": ["field"],
                "output_ids": ["field"],
                "discretization": "P1",
            }
        ],
        "constraints": [],
        "objectives": [{"id": "solution", "sense": "satisfy", "expression": {}}],
        "evidence_requests": [{"kind": "residual", "required": True}],
        "solve_plan": {
            "parameters": {"grid_shape": shape, "lengths": lengths, "diffusion": diffusion, "source": source}
        },
    }


# ---------------------------------------------------------------------------
# FieldSample construction / validation
# ---------------------------------------------------------------------------
def test_field_sample_rejects_mismatched_or_non_finite_input():
    mesh = box_simplex_mesh((3, 3), lengths=(1.0, 1.0))
    with pytest.raises(ValueError, match="coordinates row count"):
        FieldSample(
            field_name="u",
            values=np.zeros(mesh.n_nodes - 1),
            coordinates=mesh.nodes,
            unit=None,
            backend_id="test",
        )
    with pytest.raises(ValueError, match="finite"):
        FieldSample(
            field_name="u",
            values=np.full(mesh.n_nodes, np.nan),
            coordinates=mesh.nodes,
            unit=None,
            backend_id="test",
        )
    with pytest.raises(ValueError, match="mesh.nodes"):
        FieldSample(
            field_name="u",
            values=np.zeros(mesh.n_nodes),
            coordinates=mesh.nodes + 1.0,
            unit=None,
            backend_id="test",
            mesh=mesh,
        )


# ---------------------------------------------------------------------------
# point_probe / integrate / extrema mechanics on a known-analytic mesh field
# ---------------------------------------------------------------------------
def test_point_probe_and_integrate_and_extrema_match_a_known_linear_field_on_a_mesh():
    # u(x, y) = x is exactly representable by P1 elements, so a hand-built FieldSample over it is a
    # closed-form check on the query mechanics themselves (independent of any solver).
    mesh = box_simplex_mesh((9, 9), lengths=(1.0, 1.0))
    values = mesh.nodes[:, 0]
    sample = FieldSample(
        field_name="ramp", values=values, coordinates=mesh.nodes, unit="m", backend_id="synthetic", mesh=mesh
    )

    probe = point_probe(sample, (0.5, 0.5))
    assert probe.value == pytest.approx(0.5, abs=1e-12)
    assert probe.method == "p1-barycentric-interpolation"
    assert probe.unit == "m"

    result = integrate(sample)
    assert result.value == pytest.approx(0.5, abs=1e-10)  # analytic: int_0^1 int_0^1 x dx dy = 1/2
    assert result.method == "p1-consistent-mass-matrix-quadrature"
    assert result.domain_measure == pytest.approx(1.0, abs=1e-10)
    assert result.unit == "m"

    bounds = extrema(sample)
    assert bounds.min_value == pytest.approx(0.0, abs=1e-12)
    assert bounds.max_value == pytest.approx(1.0, abs=1e-12)
    assert bounds.min_location[0] == pytest.approx(0.0, abs=1e-12)
    assert bounds.max_location[0] == pytest.approx(1.0, abs=1e-12)
    assert bounds.unit == "m"


def test_point_probe_outside_meshed_domain_raises_instead_of_extrapolating():
    mesh = box_simplex_mesh((3, 3), lengths=(1.0, 1.0))
    sample = FieldSample(
        field_name="u", values=np.zeros(mesh.n_nodes), coordinates=mesh.nodes, unit=None, backend_id="test", mesh=mesh
    )
    with pytest.raises(ValueError, match="outside the meshed domain"):
        point_probe(sample, (5.0, 5.0))


def test_integrate_rejects_multi_dimensional_bare_grid_samples():
    coordinates = np.column_stack([np.linspace(0, 1, 5), np.linspace(0, 1, 5)])
    sample = FieldSample(
        field_name="u", values=np.ones(5), coordinates=coordinates, unit=None, backend_id="test", mesh=None
    )
    with pytest.raises(ValueError, match="only implemented for 1-D"):
        integrate(sample)


# ---------------------------------------------------------------------------
# Nearest-node / trapezoidal path on a bare structured-grid kernel result
# ---------------------------------------------------------------------------
def test_grid_backed_kernel_result_supports_nearest_node_probe_and_trapezoidal_integral():
    n, length, diffusivity, velocity, dt, n_steps = 41, 1.0, 0.01, 0.5, 0.01, 20
    problem = {
        "id": "transport-study",
        "domains": [{"id": "domain", "kind": "mesh", "properties": {"mesh_cell_type": "structured_grid"}}],
        "unknowns": [{"id": "field", "domain_id": "domain"}],
        "operators": [
            {
                "id": "op",
                "kind": "linear_operator",
                "input_ids": ["field"],
                "output_ids": ["field"],
                "discretization": "FD-implicit",
            }
        ],
        "constraints": [],
        "objectives": [{"id": "solution", "sense": "satisfy", "expression": {}}],
        "evidence_requests": [],
        "solve_plan": {
            "parameters": {
                "grid_size": n,
                "length": length,
                "diffusivity": diffusivity,
                "velocity": velocity,
                "dt": dt,
                "n_steps": n_steps,
                "boundary": "periodic",
            }
        },
    }
    result = run_math_problem(problem, "transport-fd-advdiff")
    # Reuse the operator's own grid construction (mixle_pde.dynamics.AdvectionDiffusionOperator) --
    # the same geometry the kernel already built internally -- rather than inventing a new one; the
    # registry result itself carries no coordinates.
    grid = AdvectionDiffusionOperator(diffusivity, velocity, n, length=length).grid
    sample = field_from_kernel_study(result, grid)

    assert sample.unit == "1"  # transport-fd-advdiff's concentration output port declares units="1"
    assert sample.field_name == "concentration"
    assert sample.mesh is None

    probe = point_probe(sample, grid[n // 2])
    assert probe.method == "nearest-node"
    assert probe.value == pytest.approx(result.solution[n // 2])

    total = integrate(sample)
    assert total.method == "trapezoidal"
    assert total.value == pytest.approx(np.trapezoid(result.solution, grid))
    assert np.isfinite(total.value)

    bounds = extrema(sample)
    assert bounds.min_value == pytest.approx(float(np.min(result.solution)))
    assert bounds.max_value == pytest.approx(float(np.max(result.solution)))


# ---------------------------------------------------------------------------
# Unit honesty: canonical_adapter carries no unit metadata
# ---------------------------------------------------------------------------
def test_canonical_poisson_field_reports_unit_none_honestly():
    mesh = box_simplex_mesh((5, 5), lengths=(1.0, 1.0))
    result = solve_p1_poisson_canonical(mesh.nodes, mesh.simplices, 1.0, conductivity=1.0)
    sample = field_from_canonical_poisson(result, mesh)
    assert sample.unit is None
    probe = point_probe(sample, (0.5, 0.5))
    assert probe.unit is None


# ---------------------------------------------------------------------------
# MP-K2 acceptance: cross-backend agreement (native-rational-linear vs fem-p1-simplex)
# ---------------------------------------------------------------------------
def test_native_and_fem_backends_agree_on_a_known_analytic_ramp_solution():
    """u(x, y) = x solves Laplace's equation exactly (source = 0, boundary = x on every edge).

    P1 elements represent linear functions exactly, so the Galerkin solution on either backend
    should reproduce this analytic field up to floating-point/rational-round-trip precision: the
    domain integral is int_0^1 int_0^1 x dx dy = 1/2, the minimum is 0 (x = 0 edge), the maximum is 1
    (x = 1 edge), and the center point probe is 0.5.
    """
    mesh = box_simplex_mesh((9, 9), lengths=(1.0, 1.0))
    dirichlet = {int(node): float(mesh.nodes[node, 0]) for node in mesh.boundary_nodes()}

    canonical_result = solve_p1_poisson_canonical(
        mesh.nodes, mesh.simplices, 0.0, conductivity=1.0, dirichlet=dirichlet
    )
    canonical_sample = field_from_canonical_poisson(canonical_result, mesh)

    fem_solution = solve_simplex_poisson(mesh, 0.0, diffusion=1.0, dirichlet=dirichlet)
    fem_sample = FieldSample(
        field_name="solution",
        values=fem_solution,
        coordinates=mesh.nodes,
        unit="1",
        backend_id="fem-p1-simplex",
        mesh=mesh,
    )

    for point in ((0.5, 0.5), (0.25, 0.75), (0.9, 0.1)):
        canonical_probe = point_probe(canonical_sample, point)
        fem_probe = point_probe(fem_sample, point)
        assert canonical_probe.value == pytest.approx(fem_probe.value, abs=CROSS_BACKEND_TOLERANCE)
        assert canonical_probe.value == pytest.approx(point[0], abs=CROSS_BACKEND_TOLERANCE)  # analytic: u(x, y) = x

    canonical_integral = integrate(canonical_sample)
    fem_integral = integrate(fem_sample)
    assert canonical_integral.value == pytest.approx(fem_integral.value, abs=CROSS_BACKEND_TOLERANCE)
    assert canonical_integral.value == pytest.approx(0.5, abs=CROSS_BACKEND_TOLERANCE)  # analytic value

    canonical_bounds = extrema(canonical_sample)
    fem_bounds = extrema(fem_sample)
    assert canonical_bounds.min_value == pytest.approx(fem_bounds.min_value, abs=CROSS_BACKEND_TOLERANCE)
    assert canonical_bounds.max_value == pytest.approx(fem_bounds.max_value, abs=CROSS_BACKEND_TOLERANCE)
    assert canonical_bounds.min_value == pytest.approx(0.0, abs=CROSS_BACKEND_TOLERANCE)
    assert canonical_bounds.max_value == pytest.approx(1.0, abs=CROSS_BACKEND_TOLERANCE)


def test_native_and_fem_backends_agree_to_near_machine_precision_on_a_nontrivial_source():
    """A homogeneous-Dirichlet Poisson study run through both backends via the registered kernel path.

    Unlike the analytic-ramp test above, ``fem-p1-simplex`` is driven through
    :func:`mixle_pde.pde_backend_registry.run_math_problem` (its actual registered CON-MATH-PROBLEM-V1
    entry point, which only supports the homogeneous Dirichlet default), demonstrating
    :func:`field_from_kernel_study` end to end. Both backends assemble the identical P1 stiffness/load
    system, so agreement here is far tighter than the documented cross-backend tolerance -- this
    pins that near-machine-precision behavior explicitly.
    """
    shape, lengths, diffusion, source = (9, 9), (1.0, 1.0), 1.0, 1.0
    mesh = box_simplex_mesh(shape, lengths=lengths)

    canonical_result = solve_p1_poisson_canonical(mesh.nodes, mesh.simplices, source, conductivity=diffusion)
    canonical_sample = field_from_canonical_poisson(canonical_result, mesh)

    study = run_math_problem(
        _fem_poisson_problem(shape=shape, lengths=lengths, diffusion=diffusion, source=source), "fem-p1-simplex"
    )
    fem_sample = field_from_kernel_study(study, mesh.nodes, mesh=mesh)

    assert fem_sample.unit == "1"
    assert fem_sample.field_name == "solution"

    canonical_probe = point_probe(canonical_sample, (0.5, 0.5))
    fem_probe = point_probe(fem_sample, (0.5, 0.5))
    assert canonical_probe.value == pytest.approx(fem_probe.value, abs=1e-9)

    canonical_integral = integrate(canonical_sample)
    fem_integral = integrate(fem_sample)
    assert canonical_integral.value == pytest.approx(fem_integral.value, rel=1e-9)

    canonical_bounds = extrema(canonical_sample)
    fem_bounds = extrema(fem_sample)
    assert canonical_bounds.max_value == pytest.approx(fem_bounds.max_value, rel=1e-9)
    assert canonical_bounds.min_value == pytest.approx(fem_bounds.min_value, abs=1e-12)
