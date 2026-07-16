import math

import numpy as np
import pytest

from mixle_pde.mesh import (
    box_simplex_mesh,
    moving_mesh,
    pipe_radial_deformation,
    pipe_simplex_mesh,
)
from mixle_pde.verification.geometric_conservation import (
    GCLVerdict,
    check_geometric_conservation_law,
)

# Every scenario below is reused verbatim (same mesh, same displacement/map_fn, same times) from
# tests/mesh_test.py's existing MovingSimplexMesh coverage -- no new mesh geometry is invented here.


def test_self_consistent_velocity_passes_on_the_3d_box_stretch_scenario():
    """Reuses tests/mesh_test.py::test_moving_mesh_interpolates_and_extrudes_deformed_geometry."""
    mesh = box_simplex_mesh((2, 2, 2), lengths=(1.0, 1.0, 1.0))
    moving = moving_mesh(mesh, [0.0, 1.0], lambda nodes, t: t * nodes * np.array([1.0, 0.0, 0.0]))

    report = check_geometric_conservation_law(moving)

    assert report.dim == 3
    assert report.n_simplices == mesh.n_simplices
    assert report.n_steps == 2
    assert report.verdict is GCLVerdict.PASS
    assert len(report.steps) == 1
    step = report.steps[0]
    assert step.verdict is GCLVerdict.PASS
    assert len(step.cells) == mesh.n_simplices
    for cell in step.cells:
        assert cell.verdict is GCLVerdict.PASS
        # Both sides are independently computed (velocity-driven quadrature vs. bare geometry
        # difference); a real, nonzero direct_volume_change confirms this is not a vacuous check.
        assert abs(cell.direct_volume_change) > 1e-6
        assert abs(cell.discrepancy) < 1e-9


def test_self_consistent_velocity_passes_on_the_pipe_radial_deformation_scenario():
    """Reuses tests/mesh_test.py::test_pipe_radial_deformation_scales_cross_section_volume."""
    mesh = box_simplex_mesh((2, 2, 3), lengths=(1.0, 1.0, 2.0), origin=(-0.5, -0.5, 0.0))
    moving = moving_mesh(mesh, [0.0, 1.0], pipe_radial_deformation(axis="z", radial_strain=lambda t: 0.2 * t))

    report = check_geometric_conservation_law(moving)

    assert report.verdict is GCLVerdict.PASS
    assert all(cell.verdict is GCLVerdict.PASS for cell in report.steps[0].cells)


def test_self_consistent_velocity_passes_on_the_annular_pipe_mesh_scenario():
    """Reuses tests/mesh_test.py::test_pipe_simplex_mesh_deforms_radially."""
    mesh = pipe_simplex_mesh(inner_radius=0.5, outer_radius=1.0, length=2.0, n_theta=24, n_axial=2)
    moving = moving_mesh(mesh, [0.0, 1.0], pipe_radial_deformation(axis="z", radial_strain=lambda t: 0.2 * t))

    report = check_geometric_conservation_law(moving)

    assert report.verdict is GCLVerdict.PASS
    assert report.n_simplices == mesh.n_simplices


def test_explicit_velocity_array_matching_the_true_motion_passes_identically_to_the_default():
    """The array-shaped node_velocity override path, fed the exact finite-difference velocity, must
    reproduce the same PASS verdict and (up to round-off) the same discrepancies as the default."""
    mesh = box_simplex_mesh((2, 2, 2), lengths=(1.0, 1.0, 1.0))
    moving = moving_mesh(mesh, [0.0, 1.0], lambda nodes, t: t * nodes * np.array([1.0, 0.0, 0.0]))
    true_velocity = moving.nodes_over_time[1] - moving.nodes_over_time[0]  # dt == 1.0

    default_report = check_geometric_conservation_law(moving)
    explicit_report = check_geometric_conservation_law(moving, node_velocity=true_velocity)

    assert explicit_report.verdict is GCLVerdict.PASS
    for default_cell, explicit_cell in zip(default_report.steps[0].cells, explicit_report.steps[0].cells, strict=True):
        assert explicit_cell.flux_volume_rate == pytest.approx(default_cell.flux_volume_rate, abs=1e-12)


def test_deliberately_inconsistent_velocity_field_is_flagged_as_failing():
    """A synthetic, deliberately-wrong node_velocity (zero, when the mesh genuinely stretches) must
    fail every cell, not silently pass or get coerced to UNKNOWN. Same reused box-stretch geometry as
    the passing test above -- only the checked velocity field is synthetic."""
    mesh = box_simplex_mesh((2, 2, 2), lengths=(1.0, 1.0, 1.0))
    moving = moving_mesh(mesh, [0.0, 1.0], lambda nodes, t: t * nodes * np.array([1.0, 0.0, 0.0]))
    wrong_velocity = np.zeros_like(moving.nodes_over_time[0])

    report = check_geometric_conservation_law(moving, node_velocity=wrong_velocity)

    assert report.verdict is GCLVerdict.FAIL
    step = report.steps[0]
    assert step.verdict is GCLVerdict.FAIL
    assert all(cell.verdict is GCLVerdict.FAIL for cell in step.cells)
    for cell in step.cells:
        # With zero claimed velocity the flux side must be exactly zero, so the discrepancy equals
        # (minus) the real, nonzero direct volume change -- a genuine O(1) relative mismatch, not noise.
        assert cell.flux_volume_rate == pytest.approx(0.0, abs=1e-12)
        assert abs(cell.discrepancy) > 1e-3
        assert cell.discrepancy == pytest.approx(-cell.direct_volume_change, abs=1e-9)


def test_deliberately_scaled_velocity_field_is_flagged_as_failing():
    """A second, differently-wrong synthetic velocity (double the true rate) also fails -- confirms
    the detector is not merely keying off a zero/sentinel value."""
    mesh = box_simplex_mesh((2, 2, 2), lengths=(1.0, 1.0, 1.0))
    moving = moving_mesh(mesh, [0.0, 1.0], lambda nodes, t: t * nodes * np.array([1.0, 0.0, 0.0]))
    true_velocity = moving.nodes_over_time[1] - moving.nodes_over_time[0]

    report = check_geometric_conservation_law(moving, node_velocity=2.0 * true_velocity)

    assert report.verdict is GCLVerdict.FAIL
    assert all(cell.verdict is GCLVerdict.FAIL for cell in report.steps[0].cells)


def test_degenerate_step_is_reported_unknown_not_a_fabricated_pass_or_fail():
    """Reuses tests/mesh_test.py::test_moving_mesh_quality_series_flags_degenerate_step, whose second
    step scales x by (1 - t) down to a literal zero-volume mesh at t=1."""
    mesh = box_simplex_mesh((2, 2, 2), lengths=(1.0, 1.0, 1.0))
    moving = moving_mesh(mesh, [0.0, 1.0], map_fn=lambda nodes, t: nodes * np.array([1.0 - t, 1.0, 1.0]))

    report = check_geometric_conservation_law(moving)

    assert report.verdict is GCLVerdict.UNKNOWN
    step = report.steps[0]
    assert step.verdict is GCLVerdict.UNKNOWN
    assert all(cell.verdict is GCLVerdict.UNKNOWN for cell in step.cells)
    for cell in step.cells:
        assert math.isnan(cell.discrepancy)
        assert math.isnan(cell.flux_volume_rate)


def test_fewer_than_two_time_steps_is_unknown_with_no_intervals():
    mesh = box_simplex_mesh((2, 2, 2), lengths=(1.0, 1.0, 1.0))
    moving = moving_mesh(mesh, [0.0], displacement=None)

    report = check_geometric_conservation_law(moving)

    assert report.verdict is GCLVerdict.UNKNOWN
    assert report.steps == ()


def test_wrongly_shaped_node_velocity_raises_value_error():
    mesh = box_simplex_mesh((2, 2, 2), lengths=(1.0, 1.0, 1.0))
    moving = moving_mesh(mesh, [0.0, 1.0], lambda nodes, t: t * nodes * np.array([1.0, 0.0, 0.0]))

    with pytest.raises(ValueError, match="node_velocity"):
        check_geometric_conservation_law(moving, node_velocity=np.zeros((3, 3)))


def test_negative_tolerance_raises_value_error():
    mesh = box_simplex_mesh((2, 2, 2), lengths=(1.0, 1.0, 1.0))
    moving = moving_mesh(mesh, [0.0, 1.0], lambda nodes, t: t * nodes * np.array([1.0, 0.0, 0.0]))

    with pytest.raises(ValueError, match="rtol"):
        check_geometric_conservation_law(moving, rtol=-1.0)
