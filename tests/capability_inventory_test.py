"""MP-A1: freeze mixle-pde's own capability and limitation inventory.

Source: notes/mixle-pde-ai-native-multiphysics-work-plan.md section 8, MP-A1.

Acceptance criteria under test:

* "import-sweep plus registry test proves each advertised symbol exists" --
  :func:`test_every_public_symbol_is_importable` actually imports every registered module and asserts
  every symbol listed in ``public_symbols`` resolves via ``getattr``.
* "unsupported features are explicit and machine-readable" --
  :func:`test_non_solver_axes_are_explicit_not_applicable` and
  :func:`test_solver_axes_are_never_empty_or_none` check that every PDE-specific axis is always a
  populated, machine-readable value (``"not_applicable"`` for non-solvers, a real value for solvers)
  rather than ``None``/empty/absent.
* Registry-vs-disk drift is covered directly: every ``mixle_pde/*.py`` module is registered exactly
  once, and no registered module is missing from disk (:func:`test_registry_matches_modules_on_disk`).
* ``docs/capability-matrix.json`` is a checked-in, byte-identical dump of the registry
  (:func:`test_capability_matrix_json_is_up_to_date`), so the frozen artifact cannot silently drift
  from the source of truth.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

from mixle_pde.verification import capability_inventory as ci
from mixle_pde.verification.capability_inventory import (
    CAPABILITY_INVENTORY,
    SolverProfile,
    by_category,
    get_profile,
    modules,
    solver_profiles,
    to_capability_matrix,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
PACKAGE_DIR = REPO_ROOT / "mixle_pde"
MATRIX_PATH = REPO_ROOT / "docs" / "capability-matrix.json"

# mixle_pde.io and mixle_pde.geospatial are data-ingest/interchange subpackages (SEG-Y, LAS, InSAR,
# CRS reprojection, ...), not PDE/ODE solvers -- explicitly out of scope for a *solver* capability
# inventory, not an oversight. verification/ is this inventory itself.
OUT_OF_SCOPE_SUBPACKAGES = {"io", "geospatial", "verification"}


def _modules_on_disk() -> set[str]:
    names = set()
    for path in PACKAGE_DIR.glob("*.py"):
        if path.name == "__init__.py":
            continue
        if path.name.endswith("_test.py"):
            continue
        names.add(f"mixle_pde.{path.stem}")
    return names


def test_registry_matches_modules_on_disk():
    """Every top-level solver/utility module is registered exactly once; no stale entries either."""
    on_disk = _modules_on_disk()
    registered = set(modules())
    missing_from_registry = on_disk - registered
    stale_in_registry = registered - on_disk
    assert not missing_from_registry, f"modules on disk but not inventoried: {sorted(missing_from_registry)}"
    assert not stale_in_registry, f"inventoried modules no longer on disk: {sorted(stale_in_registry)}"


def test_registry_has_no_duplicate_modules():
    names = [profile.module for profile in CAPABILITY_INVENTORY]
    assert len(names) == len(set(names))


@pytest.mark.parametrize("profile", CAPABILITY_INVENTORY, ids=lambda p: p.module)
def test_every_public_symbol_is_importable(profile: SolverProfile):
    """The literal acceptance criterion: import-sweep every module, resolve every advertised symbol."""
    module = importlib.import_module(profile.module)
    missing = [name for name in profile.public_symbols if not hasattr(module, name)]
    assert not missing, f"{profile.module} advertises missing symbols: {missing}"


def test_public_symbols_are_non_empty_for_every_module():
    empty = [p.module for p in CAPABILITY_INVENTORY if not p.public_symbols]
    assert not empty, f"modules with zero advertised public symbols (should not be registered): {empty}"


def test_non_solver_axes_are_explicit_not_applicable():
    """Unsupported/inapplicable PDE axes must be the explicit sentinel, never silently blank."""
    for profile in CAPABILITY_INVENTORY:
        if profile.is_solver:
            continue
        assert profile.dimension == ("not_applicable",), profile.module
        assert profile.grid_type == "not_applicable", profile.module
        assert profile.boundary_conditions == ("not_applicable",), profile.module
        assert profile.time_regime == "not_applicable", profile.module


def test_solver_axes_are_never_empty_or_none():
    for profile in CAPABILITY_INVENTORY:
        assert profile.dimension, profile.module
        assert all(profile.dimension), profile.module
        assert profile.grid_type, profile.module
        assert profile.boundary_conditions, profile.module
        assert all(profile.boundary_conditions), profile.module
        assert profile.time_regime, profile.module
        assert profile.parallel_status, profile.module
        assert profile.verification_level, profile.module
        assert profile.category, profile.module
        assert profile.method, profile.module


def test_parallel_status_is_reported_for_every_module():
    """Repo-wide finding: no MPI/multiprocessing anywhere in mixle_pde -- with one named exception.

    mixle_pde.parallel_bayesian_execution (MP-I9) genuinely dispatches chains across OS processes via
    concurrent.futures.ProcessPoolExecutor, so it correctly reports "multi_process" rather than
    "single_process" -- carved out by name, not by weakening this invariant for every other module.
    """
    multi_process_modules = {"mixle_pde.parallel_bayesian_execution"}
    for profile in CAPABILITY_INVENTORY:
        expected = "multi_process" if profile.module in multi_process_modules else "single_process"
        assert profile.parallel_status == expected, profile.module


def test_get_profile_round_trips_and_rejects_unknown_module():
    profile = get_profile("mixle_pde.fem")
    assert profile.module == "mixle_pde.fem"
    with pytest.raises(KeyError):
        get_profile("mixle_pde.does_not_exist")


def test_solver_profiles_are_a_strict_subset_flagged_is_solver():
    solvers = solver_profiles()
    assert solvers
    assert all(p.is_solver for p in solvers)
    assert set(p.module for p in solvers) <= set(modules())
    assert len(solvers) < len(CAPABILITY_INVENTORY)


def test_by_category_partitions_the_registry():
    grouped = by_category()
    total = sum(len(profiles) for profiles in grouped.values())
    assert total == len(CAPABILITY_INVENTORY)
    for category, profiles in grouped.items():
        assert all(p.category == category for p in profiles)


@pytest.mark.parametrize(
    "module_name",
    [
        "mixle_pde.fem",
        "mixle_pde.gas_dynamics",
        "mixle_pde.geophysics",
        "mixle_pde.monitoring_design",
        "mixle_pde.posterior_query",
        "mixle_pde.geo_observations",
    ],
)
def test_task_named_example_modules_are_registered(module_name):
    """PDE-A1 explicitly names these modules as inventory targets; make sure they stay covered."""
    profile = get_profile(module_name)
    assert profile.public_symbols
    assert profile.method


def test_fem_is_flagged_as_a_dimension_generic_simplex_solver():
    profile = get_profile("mixle_pde.fem")
    assert profile.is_solver
    assert "solve_simplex_poisson" in profile.public_symbols
    assert "2D" in profile.dimension and "3D" in profile.dimension


def test_geophysics_forward_operators_are_not_a_discretized_pde_solver_family():
    """observations.py/potential_fields.py are sensitivity-kernel libraries, not BVP solvers."""
    assert get_profile("mixle_pde.observations").is_solver is False
    assert get_profile("mixle_pde.potential_fields").is_solver is False


def test_capability_matrix_shape_is_machine_readable_and_stable():
    matrix = to_capability_matrix()
    assert matrix["schema_version"] == 1
    assert len(matrix["modules"]) == len(CAPABILITY_INVENTORY)
    for entry in matrix["modules"]:
        assert isinstance(entry["dimension"], list)
        assert isinstance(entry["boundary_conditions"], list)
        assert isinstance(entry["public_symbols"], list)
        assert isinstance(entry["limitations"], list)
        json.dumps(entry)  # every entry must be JSON-serializable on its own


def test_capability_matrix_json_is_up_to_date():
    """docs/capability-matrix.json must be a frozen, checked-in dump of the live registry."""
    assert MATRIX_PATH.exists(), "run write_capability_matrix() and commit docs/capability-matrix.json"
    on_disk = json.loads(MATRIX_PATH.read_text())
    assert on_disk == to_capability_matrix(), (
        "docs/capability-matrix.json is stale relative to CAPABILITY_INVENTORY; "
        "regenerate with mixle_pde.verification.capability_inventory.write_capability_matrix(...)"
    )


def test_write_capability_matrix_round_trips(tmp_path):
    target = ci.write_capability_matrix(tmp_path / "nested" / "capability-matrix.json")
    assert target.exists()
    assert json.loads(target.read_text()) == to_capability_matrix()
