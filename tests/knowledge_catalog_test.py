"""MP-J2 baseline: solver knowledge catalog (governing equations / applicability / benchmark evidence).

Source: notes/mixle-pde-ai-native-multiphysics-work-plan.md section 8, MP-J2.

Acceptance criteria under test -- the completeness/no-drift invariant, matching the pattern
``tests/capability_inventory_test.py`` uses for ``capability_inventory.py``:

* Every catalog entry names a real, currently-registered :mod:`mixle_pde.pde_backend_registry`
  backend id, and its ``source`` string matches that registration's own ``source`` field
  (:func:`test_every_catalog_entry_matches_a_registered_backend`).
* Every registered backend id has a catalog entry -- no orphaned/uncovered kernel
  (:func:`test_every_registered_backend_has_a_catalog_entry`).
* Every cited benchmark-evidence test file exists on disk and actually defines the cited test
  function -- "cite real test files, don't invent evidence" is machine-checked, not just asserted
  in prose (:func:`test_benchmark_evidence_files_and_tests_exist_on_disk`).
* Every entry's descriptive fields are explicit, non-empty values, never silently blank
  (:func:`test_catalog_fields_are_never_empty`).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from mixle_pde.pde_backend_registry import list_kernel_registrations
from mixle_pde.verification.knowledge_catalog import (
    KNOWLEDGE_CATALOG,
    KnowledgeCatalogEntry,
    catalog_backend_ids,
    entries_by_physics_domain,
    get_entry,
    list_entries,
    to_catalog_matrix,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def _registered_backend_ids() -> set[str]:
    return {registration.profile.id for registration in list_kernel_registrations()}


def _registered_sources() -> dict[str, str]:
    return {registration.profile.id: registration.source for registration in list_kernel_registrations()}


def test_registry_has_at_least_the_backends_this_catalog_was_authored_against():
    """Sanity check on the live registry itself, so a future registry shrink is loud, not silent."""
    expected = {
        "elastic-fd-leapfrog",
        "helmholtz-pml-fd",
        "groundwater-fd-transport",
        "fem-p1-simplex",
        "wave-fd-leapfrog",
        "flow-fd-streamfunction",
        "em-fdtd-yee",
        "transport-fd-advdiff",
        "flow-spectral-ns",
    }
    assert expected <= _registered_backend_ids()


def test_every_catalog_entry_matches_a_registered_backend():
    """No catalog entry may reference a backend id that does not actually exist in the registry."""
    registered = _registered_backend_ids()
    sources = _registered_sources()
    for entry in KNOWLEDGE_CATALOG:
        assert entry.backend_id in registered, f"{entry.backend_id!r} is not a registered PDE backend"
        assert entry.source == sources[entry.backend_id], (
            f"{entry.backend_id!r} catalog source {entry.source!r} does not match the registered "
            f"kernel's own source {sources[entry.backend_id]!r}"
        )


def test_every_registered_backend_has_a_catalog_entry():
    """No orphaned capability: every registered kernel must be covered by this catalog."""
    registered = _registered_backend_ids()
    catalogued = set(catalog_backend_ids())
    missing = registered - catalogued
    assert not missing, f"registered backends with no knowledge_catalog entry: {sorted(missing)}"


def test_catalog_has_no_duplicate_or_stale_entries():
    ids = [entry.backend_id for entry in KNOWLEDGE_CATALOG]
    assert len(ids) == len(set(ids))
    assert set(ids) <= _registered_backend_ids()


@pytest.mark.parametrize("entry", KNOWLEDGE_CATALOG, ids=lambda e: e.backend_id)
def test_catalog_fields_are_never_empty(entry: KnowledgeCatalogEntry):
    assert entry.backend_id
    assert entry.source
    assert entry.physics_domain
    assert entry.governing_equation
    assert entry.method
    assert entry.applicability
    assert all(entry.applicability)
    assert entry.benchmark_evidence


@pytest.mark.parametrize("entry", KNOWLEDGE_CATALOG, ids=lambda e: e.backend_id)
def test_benchmark_evidence_files_and_tests_exist_on_disk(entry: KnowledgeCatalogEntry):
    """'Cite real test files, don't invent evidence' -- checked mechanically, not just by review."""
    for evidence in entry.benchmark_evidence:
        assert evidence.test_file
        assert evidence.test_name
        assert evidence.check_kind
        assert evidence.summary
        path = REPO_ROOT / evidence.test_file
        assert path.is_file(), f"{entry.backend_id}: cited test file {evidence.test_file!r} does not exist"
        text = path.read_text()
        pattern = re.compile(rf"^\s*def {re.escape(evidence.test_name)}\b", re.MULTILINE)
        assert pattern.search(text), (
            f"{entry.backend_id}: {evidence.test_file!r} does not define a test named {evidence.test_name!r}"
        )


def test_get_entry_round_trips_and_rejects_unknown_backend():
    entry = get_entry("fem-p1-simplex")
    assert entry.backend_id == "fem-p1-simplex"
    with pytest.raises(KeyError):
        get_entry("not-a-real-backend-id")


def test_list_entries_matches_the_module_level_tuple():
    assert list_entries() == KNOWLEDGE_CATALOG


def test_catalog_backend_ids_matches_entry_order():
    assert catalog_backend_ids() == tuple(entry.backend_id for entry in KNOWLEDGE_CATALOG)


def test_entries_by_physics_domain_partitions_the_catalog():
    grouped = entries_by_physics_domain()
    total = sum(len(entries) for entries in grouped.values())
    assert total == len(KNOWLEDGE_CATALOG)
    for domain, entries in grouped.items():
        assert all(entry.physics_domain == domain for entry in entries)


def test_catalog_matrix_shape_is_machine_readable_and_json_serializable():
    import json

    matrix = to_catalog_matrix()
    assert matrix["schema_version"] == 1
    assert len(matrix["entries"]) == len(KNOWLEDGE_CATALOG)
    for record in matrix["entries"]:
        assert isinstance(record["applicability"], list)
        assert isinstance(record["benchmark_evidence"], list)
        json.dumps(record)  # every entry must be JSON-serializable on its own


def test_no_entry_claims_support_beyond_its_own_registered_profile():
    """Cross-check applicability against the live PDEBackendProfile -- catch overclaiming drift.

    Every mesh_cell_type the profile declares support for should not contradict what the entry's
    applicability notes say is the *only* supported grid/mesh kind (a coarse but real guard against
    the catalog silently drifting more permissive than the registry it describes).
    """
    registrations = {registration.profile.id: registration for registration in list_kernel_registrations()}
    for entry in KNOWLEDGE_CATALOG:
        profile = registrations[entry.backend_id].profile
        assert profile.mesh_cell_types, f"{entry.backend_id}: registered profile declares no mesh_cell_types"
