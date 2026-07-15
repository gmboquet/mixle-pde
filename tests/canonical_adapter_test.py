from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from mixle_pde.canonical_adapter import (
    SIM_LINEAR_SCHEMA,
    native_backend_manifest,
    solve_p1_poisson_canonical,
    solve_sim_linear_system,
)
from mixle_pde.ownership import migration_inventory, migration_inventory_digest


def linear_record() -> dict[str, object]:
    return {
        "schema": SIM_LINEAR_SCHEMA,
        "id": "two-by-two",
        "coefficient_domain": "rational",
        "source_problem_digest": "a" * 64,
        "unknowns": ["x", "y"],
        "matrix": [["2/1", "-1/1"], ["-1/1", "2/1"]],
        "rhs": ["1/1", "0/1"],
    }


def square_mesh(size: int) -> tuple[np.ndarray, np.ndarray]:
    coordinates = np.linspace(0, 1, size)
    grid_x, grid_y = np.meshgrid(coordinates, coordinates, indexing="ij")
    nodes = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    triangles = []
    for row in range(size - 1):
        for column in range(size - 1):
            lower = row * size + column
            triangles.extend(((lower, lower + 1, lower + size + 1), (lower, lower + size + 1, lower + size)))
    return nodes, np.asarray(triangles, dtype=int)


def test_portable_sim_record_executes_with_bound_residual_and_capability_receipt() -> None:
    result = solve_sim_linear_system(linear_record())
    np.testing.assert_allclose(result.values, (2 / 3, 1 / 3))
    assert result.receipt.outcome == "approximate"
    assert result.receipt.residual_norm < 1e-14
    assert result.receipt.backend_capability_digest == native_backend_manifest()[0].identity
    assert result.receipt.evidence_strength == "checked-residual"


def test_portable_adapter_rejects_unsupported_malformed_and_singular_records() -> None:
    with pytest.raises(ValueError, match="unsupported simulation record"):
        solve_sim_linear_system({**linear_record(), "schema": "invented/v9"})
    with pytest.raises(ValueError, match="dimensions do not agree"):
        solve_sim_linear_system({**linear_record(), "rhs": ["1/1"]})
    with pytest.raises(ValueError, match="singular"):
        solve_sim_linear_system({**linear_record(), "matrix": [["1/1", "1/1"], ["1/1", "1/1"]]})


def test_legacy_p1_poisson_wrapper_has_numerical_parity_and_canonical_receipts() -> None:
    nodes, triangles = square_mesh(7)
    result = solve_p1_poisson_canonical(nodes, triangles, 1.0, conductivity=2.0)
    assert result.legacy_parity_max_abs < 1e-11
    assert result.solve_receipt.residual_norm < 1e-12
    assert result.system_record["schema"] == SIM_LINEAR_SCHEMA
    assert result.system_record["conversion"]["mesh_digest"] == result.mesh_digest
    assert np.all(np.isfinite(result.solution))


def test_p1_wrapper_rejects_capability_excess_instead_of_guessing() -> None:
    nodes, triangles = square_mesh(3)
    with pytest.raises(ValueError, match="scalar source"):
        solve_p1_poisson_canonical(nodes, triangles, np.ones(len(nodes)))
    with pytest.raises(ValueError, match="invalid node"):
        solve_p1_poisson_canonical(nodes, triangles, 1.0, dirichlet={len(nodes): 0.0})


def test_migration_inventory_classifies_every_shipped_source_module() -> None:
    root = Path(__file__).parents[1] / "mixle_pde"
    source_modules = {f"mixle_pde.{path.stem}" for path in root.glob("*.py") if path.name != "__init__.py"}
    inventory = migration_inventory()
    assert {item.module for item in inventory} == source_modules
    assert all(item.final_owner.startswith("PRJ-") and item.rationale for item in inventory)
    assert migration_inventory_digest() == migration_inventory_digest()
    assert any(item.final_owner == "PRJ-SIM" and item.disposition == "adapt" for item in inventory)
    assert any(item.final_owner == "PRJ-CORE" and item.disposition == "migrate" for item in inventory)


def test_native_manifest_advertises_only_tested_canonical_slice() -> None:
    manifest = native_backend_manifest()
    assert len(manifest) == 1
    capability = manifest[0]
    assert capability.record_schemas == (SIM_LINEAR_SCHEMA,)
    assert capability.problem_kinds == ("rational_linear_system",)
    assert capability.discretizations == ("P1",)
    with pytest.raises(ValueError, match="unsupported backend maturity"):
        replace(capability, maturity="magical")


def test_required_documentation_paths_exist() -> None:
    root = Path(__file__).parents[1]
    manifest = json.loads((root / "docs/manifest.json").read_text())
    assert all((root / path).exists() for path in manifest["entries"])
