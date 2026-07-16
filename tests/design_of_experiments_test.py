"""Design-of-experiments / adaptive sampling: DoD conformance for mixle_pde.design_of_experiments.

Three things are under test: (1) each design kind (plain random, Latin-Hypercube QMC, Sobol' QMC,
adaptive) produces exactly the declared point count, all within bounds, and every design run's
``TrainingDesignManifest`` content hash is reproducible for a fixed seed and changes for a different
seed or point set; (2) the three space-filling samplers satisfy ``distill_forward``'s own
``sampler(n, rng) -> Sequence`` contract directly, including against a real registered PDE kernel
(``mixle_pde.fem.solve_simplex_poisson``) as a worked example; (3) the adaptive design's real,
checkable claim -- refitting a surrogate on the full adaptive design (initial batch plus
error-concentrated refinement) genuinely lowers held-out error relative to refitting on the initial
batch alone, on a synthetic teacher with a sharp, spatially localized feature a sparse initial batch
under-resolves.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from mixle_pde.design_of_experiments import (
    AdaptiveDesignResult,
    TrainingDesignManifest,
    adaptive_design,
    latin_hypercube_design,
    latin_hypercube_sampler,
    random_design,
    random_sampler,
    sobol_design,
    sobol_sampler,
)
from mixle_pde.surrogate import Surrogate, distill_forward

_BOUNDS_2D = [(-2.0, 2.0), (-1.0, 3.0)]
_DESIGN_FUNCTIONS = {
    "random": random_design,
    "latin_hypercube": latin_hypercube_design,
    "sobol": sobol_design,
}
_SAMPLER_FACTORIES = {
    "random": random_sampler,
    "latin_hypercube": latin_hypercube_sampler,
    "sobol": sobol_sampler,
}


def _analytic_teacher(x):
    """Cheap stand-in "teacher": a smooth nonlinear scalar function of two physical parameters
    (the same shape used by tests/test_e6_surrogate.py's teacher for surrogate.py itself)."""
    a, b = x
    return math.sin(a) + 0.5 * b * b


def _bumpy_teacher(x):
    """A smooth linear background plus one sharp, spatially localized Gaussian bump. A small
    space-filling batch over [0, 1]^2 will rarely land near the bump and so will under-resolve it,
    which is exactly the failure mode adaptive, error-concentrated refinement should fix."""
    x1, x2 = x
    bump = 3.0 * math.exp(-(((x1 - 0.75) ** 2 + (x2 - 0.25) ** 2) / (2 * 0.04)))
    return 0.3 * x1 - 0.2 * x2 + bump


def _replay_sampler(points):
    """Test-local helper: replay an already-realized coordinate array through distill_forward's own
    sampler contract, so a Surrogate can be refit on exactly a design's realized points rather than a
    freshly (and possibly different) redrawn set."""
    frozen = np.asarray(points, dtype=float)

    def sampler(n, rng):
        assert n == len(frozen)
        return frozen

    return sampler


@pytest.fixture(autouse=True)
def _torch_default_dtype_is_float32():
    """distill_forward's student net hard-codes float32 tensors (mixle.task.regress._fit_reg_mlp), so
    any test that builds a Surrogate crashes with a dtype mismatch if torch's process-global default
    dtype was left at float64 by an earlier test sharing this worker process. Dozens of mixle_pde test
    files call ``torch.set_default_dtype(torch.float64)`` without resetting it, and this package's
    tests/ (unlike the sibling mixle core package's tests/conftest.py) has no isolation fixture for
    that -- so under pytest-xdist, whether a given test lands on a "clean" or "poisoned" worker is
    scheduling luck. Force float32 before every test here and restore whatever was ambient afterward,
    so these tests are self-contained and deterministic regardless of run order or worker assignment.
    """
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    yield
    torch.set_default_dtype(previous)


# ---------------------------------------------------------------------------
# Random / Latin-Hypercube / Sobol designs: point count, bounds, manifest hash
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", sorted(_DESIGN_FUNCTIONS))
def test_design_produces_the_declared_point_count_within_bounds(kind):
    design_fn = _DESIGN_FUNCTIONS[kind]
    points, manifest = design_fn(_BOUNDS_2D, 40, seed=3)

    assert points.shape == (40, 2)
    assert manifest.kind == kind
    assert manifest.requested_points == 40
    assert manifest.actual_points == 40

    lo = np.array([b[0] for b in _BOUNDS_2D])
    hi = np.array([b[1] for b in _BOUNDS_2D])
    assert np.all(points >= lo)
    assert np.all(points <= hi)


@pytest.mark.parametrize("kind", sorted(_DESIGN_FUNCTIONS))
def test_manifest_content_hash_is_reproducible_for_a_fixed_seed(kind):
    design_fn = _DESIGN_FUNCTIONS[kind]
    points_a, manifest_a = design_fn(_BOUNDS_2D, 24, seed=5)
    points_b, manifest_b = design_fn(_BOUNDS_2D, 24, seed=5)

    assert manifest_a.content_hash == manifest_b.content_hash
    np.testing.assert_array_equal(points_a, points_b)


@pytest.mark.parametrize("kind", sorted(_DESIGN_FUNCTIONS))
def test_manifest_content_hash_changes_for_a_different_seed(kind):
    design_fn = _DESIGN_FUNCTIONS[kind]
    _, manifest_a = design_fn(_BOUNDS_2D, 24, seed=5)
    _, manifest_b = design_fn(_BOUNDS_2D, 24, seed=6)

    assert manifest_a.content_hash != manifest_b.content_hash


@pytest.mark.parametrize("kind", sorted(_DESIGN_FUNCTIONS))
def test_manifest_content_hash_changes_for_a_different_point_set(kind):
    design_fn = _DESIGN_FUNCTIONS[kind]
    _, manifest_a = design_fn(_BOUNDS_2D, 24, seed=5)
    _, manifest_b = design_fn(_BOUNDS_2D, 32, seed=5)

    assert manifest_a.requested_points != manifest_b.requested_points
    assert manifest_a.content_hash != manifest_b.content_hash


def test_manifest_bounds_are_canonicalized_as_plain_float_tuples():
    _, manifest = random_design([(-2, 2), (0, 1)], 10, seed=0)
    assert manifest.bounds == ((-2.0, 2.0), (0.0, 1.0))
    assert all(isinstance(v, float) for pair in manifest.bounds for v in pair)
    assert isinstance(manifest.seed, int)


def test_design_rejects_a_non_positive_point_count():
    with pytest.raises(ValueError):
        random_design(_BOUNDS_2D, 0, seed=0)


def test_manifest_rejects_malformed_fields():
    well_formed = {
        "kind": "random",
        "bounds": ((0.0, 1.0),),
        "seed": 0,
        "requested_points": 1,
        "actual_points": 1,
        "content_hash": "0" * 64,
    }
    manifest = TrainingDesignManifest(**well_formed)
    assert manifest.kind == "random"

    with pytest.raises(ValueError):
        TrainingDesignManifest(**{**well_formed, "kind": "not-a-design-kind"})
    with pytest.raises(ValueError):
        TrainingDesignManifest(**{**well_formed, "bounds": ()})
    with pytest.raises(ValueError):
        TrainingDesignManifest(**{**well_formed, "bounds": ((1.0, 0.0),)})  # hi <= lo
    with pytest.raises(ValueError):
        TrainingDesignManifest(**{**well_formed, "requested_points": 0})
    with pytest.raises(ValueError):
        TrainingDesignManifest(**{**well_formed, "actual_points": -1})
    with pytest.raises(ValueError):
        TrainingDesignManifest(**{**well_formed, "content_hash": "not-a-hex-digest"})


# ---------------------------------------------------------------------------
# Samplers plug directly into distill_forward's own sampler(n, rng) contract
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", sorted(_SAMPLER_FACTORIES))
def test_samplers_satisfy_the_distill_forward_sampler_contract(kind):
    sampler = _SAMPLER_FACTORIES[kind](_BOUNDS_2D)
    surrogate = distill_forward(_analytic_teacher, sampler, budget=32, seed=0)
    assert isinstance(surrogate, Surrogate)

    rng = np.random.default_rng(1)
    probe = list(rng.uniform([-1.5, -0.5], [1.5, 2.5], size=(10, 2)))
    report = surrogate.evaluate(probe)
    assert report["n"] == 10
    assert np.isfinite(report["mae"]).all()


def test_worked_example_surrogate_of_a_registered_fem_kernel():
    """mixle_pde.fem.solve_simplex_poisson (already registered/tested elsewhere in this package) used
    only as a realistic teacher forward: this module's samplers should plug into distill_forward for a
    genuine PDE solve, not just synthetic analytic functions. fem.py itself is untouched."""
    from mixle_pde.fem import solve_simplex_poisson
    from mixle_pde.mesh import box_simplex_mesh

    mesh = box_simplex_mesh((6, 6), lengths=(1.0, 1.0))

    def teacher(x):
        diffusion, magnitude = x
        u = solve_simplex_poisson(mesh, source=float(magnitude), diffusion=float(diffusion))
        return float(u.mean())

    bounds = [(0.5, 3.0), (0.5, 3.0)]
    surrogate = distill_forward(teacher, latin_hypercube_sampler(bounds), budget=24, seed=0)

    low_regime = (3.0, 0.5)  # low source, high diffusion -> small mean solution
    high_regime = (0.5, 3.0)  # high source, low diffusion -> large mean solution
    predicted_low = surrogate.predict(low_regime)
    predicted_high = surrogate.predict(high_regime)

    assert math.isfinite(predicted_low)
    assert math.isfinite(predicted_high)
    assert abs(predicted_high - teacher(high_regime)) < 0.05
    assert predicted_high > predicted_low  # the surrogate learned real input sensitivity, not a constant


# ---------------------------------------------------------------------------
# Adaptive / error-driven design
# ---------------------------------------------------------------------------


def test_adaptive_design_produces_the_declared_point_count_within_bounds_and_is_reproducible():
    result_a = adaptive_design(_analytic_teacher, _BOUNDS_2D, budget=32, seed=4, init_fraction=0.5, points_per_round=16)
    assert isinstance(result_a, AdaptiveDesignResult)
    assert result_a.points.shape == (32, 2)
    assert result_a.manifest.kind == "adaptive"
    assert result_a.manifest.requested_points == 32
    assert result_a.manifest.actual_points == 32
    assert 16 <= result_a.n_initial <= 32
    assert isinstance(result_a.bootstrap_surrogate, Surrogate)

    lo = np.array([b[0] for b in _BOUNDS_2D])
    hi = np.array([b[1] for b in _BOUNDS_2D])
    assert np.all(result_a.points >= lo)
    assert np.all(result_a.points <= hi)

    result_b = adaptive_design(_analytic_teacher, _BOUNDS_2D, budget=32, seed=4, init_fraction=0.5, points_per_round=16)
    assert result_a.manifest.content_hash == result_b.manifest.content_hash
    np.testing.assert_array_equal(result_a.points, result_b.points)

    result_c = adaptive_design(_analytic_teacher, _BOUNDS_2D, budget=32, seed=9, init_fraction=0.5, points_per_round=16)
    assert result_a.manifest.content_hash != result_c.manifest.content_hash


def test_adaptive_design_initial_batch_is_a_verbatim_prefix_of_the_full_design():
    result = adaptive_design(_analytic_teacher, _BOUNDS_2D, budget=40, seed=1, init_fraction=0.4, points_per_round=12)
    initial_batch = latin_hypercube_sampler(_BOUNDS_2D)(result.n_initial, np.random.default_rng(1))
    np.testing.assert_array_equal(result.points[: result.n_initial], initial_batch)


def test_adaptive_design_rejects_a_budget_below_the_distill_forward_floor():
    with pytest.raises(ValueError):
        adaptive_design(_analytic_teacher, _BOUNDS_2D, budget=8, seed=0)


def test_adaptive_design_held_out_error_improves_over_the_initial_batch_alone():
    """The real, checkable claim: refit one surrogate on the full adaptive design (initial batch plus
    error-concentrated refinement) and one on the initial batch alone, then measure both against an
    independent held-out set drawn uniformly over the whole domain. The refined surrogate's worst-case
    error should be markedly lower -- exactly the outcome "concentrate new points where error is
    worst" is supposed to buy, since the bump is the dominant source of worst-case error and a sparse
    initial batch rarely lands near it.
    """
    bounds = [(0.0, 1.0), (0.0, 1.0)]
    seed = 2

    result = adaptive_design(_bumpy_teacher, bounds, budget=64, seed=seed, init_fraction=0.3, points_per_round=16)
    initial_only = result.bootstrap_surrogate
    refined = distill_forward(_bumpy_teacher, _replay_sampler(result.points), budget=result.points.shape[0], seed=seed)

    rng = np.random.default_rng(999)
    holdout = list(rng.uniform([0.0, 0.0], [1.0, 1.0], size=(300, 2)))
    initial_report = initial_only.evaluate(holdout)
    refined_report = refined.evaluate(holdout)

    initial_max_error = initial_report["max_abs_error"][0]
    refined_max_error = refined_report["max_abs_error"][0]
    assert refined_max_error < 0.85 * initial_max_error, (
        "adaptive refinement should cut worst-case held-out error by at least 15% relative to the "
        f"initial batch alone: initial={initial_max_error!r} refined={refined_max_error!r}"
    )
