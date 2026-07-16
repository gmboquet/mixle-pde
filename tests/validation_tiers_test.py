"""Tests for the MP-K5 typed validation-tier taxonomy (mixle_pde.verification.validation_tiers).

Covers the taxonomy's internal consistency (every ``ValidationTier`` has exactly one
``TierSemantics`` entry, with the honesty properties the module docstring claims), the
``ValidationEvidenceRecord``/registry/decorator mechanics (including the negative path: a failing
decorated call must not register evidence), that every ``STATIC_EVIDENCE_CATALOG`` citation actually
resolves to a real test on disk, and one live, non-hypothetical worked example: decorating and
running the already-merged ``manufactured_solution_refinement`` reference (MP-K3-style manufactured
solution, real numerical solve, not mocked) and confirming the query helpers report exactly the tier
that run earned.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from mixle_pde.multiphysics_reference import manufactured_solution_refinement
from mixle_pde.verification.validation_tiers import (
    STATIC_EVIDENCE_CATALOG,
    TIER_ORDER,
    TIER_SEMANTICS,
    ValidationEvidenceRecord,
    ValidationTier,
    clear_registry,
    evidence_for_capability,
    evidence_report,
    known_capabilities,
    registered_evidence,
    semantics_for,
    tiers_for_capability,
    validation_evidence,
)

# ---------------------------------------------------------------------------
# TierSemantics: completeness and the honesty properties the module docstring claims.
# ---------------------------------------------------------------------------


def test_tier_semantics_covers_every_tier_exactly_once():
    tiers = [entry.tier for entry in TIER_SEMANTICS]
    assert set(tiers) == set(ValidationTier)
    assert len(tiers) == len(ValidationTier)
    assert set(TIER_ORDER) == set(ValidationTier)
    assert len(TIER_ORDER) == len(ValidationTier)


def test_semantics_for_returns_matching_tier():
    for tier in ValidationTier:
        assert semantics_for(tier).tier is tier


def test_manufactured_tier_gives_exact_error_but_not_physical_validity():
    # The whole point of MMS: the comparison is exact, but the manufactured solution need not be
    # physically meaningful, so it cannot certify physical validity on its own.
    semantics = semantics_for(ValidationTier.MANUFACTURED)
    assert semantics.gives_exact_error is True
    assert semantics.establishes_physical_validity is False


def test_analytic_and_experimental_tiers_establish_physical_validity():
    assert semantics_for(ValidationTier.ANALYTIC).establishes_physical_validity is True
    assert semantics_for(ValidationTier.EXPERIMENTAL).establishes_physical_validity is True


def test_code_to_code_and_field_tiers_give_no_exact_error_or_validity_claim():
    for tier in (ValidationTier.CODE_TO_CODE, ValidationTier.FIELD):
        semantics = semantics_for(tier)
        assert semantics.gives_exact_error is False
        assert semantics.establishes_physical_validity is False


def test_field_tier_explicitly_names_absence_of_ground_truth():
    semantics = semantics_for(ValidationTier.FIELD)
    assert "no independent ground truth" in " ".join(semantics.typical_confounds).lower()


# ---------------------------------------------------------------------------
# ValidationEvidenceRecord: construction validation.
# ---------------------------------------------------------------------------


def test_validation_evidence_record_accepts_a_well_formed_record():
    record = ValidationEvidenceRecord(
        capability="demo::capability",
        tier=ValidationTier.ANALYTIC,
        source="tests/validation_tiers_test.py::test_validation_evidence_record_accepts_a_well_formed_record",
        summary="demonstration record",
        origin="static_catalog",
    )
    assert record.limitations == ()


@pytest.mark.parametrize("field_name", ["capability", "source", "summary"])
def test_validation_evidence_record_rejects_blank_required_strings(field_name):
    kwargs = {
        "capability": "demo::capability",
        "tier": ValidationTier.ANALYTIC,
        "source": "demo-source",
        "summary": "demo-summary",
        "origin": "static_catalog",
    }
    kwargs[field_name] = "   "
    with pytest.raises(ValueError):
        ValidationEvidenceRecord(**kwargs)


def test_validation_evidence_record_rejects_unknown_origin():
    with pytest.raises(ValueError):
        ValidationEvidenceRecord(
            capability="demo::capability",
            tier=ValidationTier.ANALYTIC,
            source="demo-source",
            summary="demo-summary",
            origin="made_up_origin",
        )


def test_validation_evidence_record_rejects_non_tier_member():
    with pytest.raises(TypeError):
        ValidationEvidenceRecord(
            capability="demo::capability",
            tier="manufactured",  # a raw string, not ValidationTier.MANUFACTURED
            source="demo-source",
            summary="demo-summary",
            origin="static_catalog",
        )


# ---------------------------------------------------------------------------
# STATIC_EVIDENCE_CATALOG: every citation resolves to a real test that exists on disk.
# ---------------------------------------------------------------------------


def test_static_evidence_catalog_is_nonempty_and_multi_tier():
    assert len(STATIC_EVIDENCE_CATALOG) >= 3
    assert {record.tier for record in STATIC_EVIDENCE_CATALOG} >= {
        ValidationTier.MANUFACTURED,
        ValidationTier.CODE_TO_CODE,
        ValidationTier.ANALYTIC,
    }
    assert all(record.origin == "static_catalog" for record in STATIC_EVIDENCE_CATALOG)


def test_static_evidence_catalog_sources_exist_on_disk():
    repo_root = Path(__file__).resolve().parent.parent
    for record in STATIC_EVIDENCE_CATALOG:
        file_part, separator, test_name = record.source.partition("::")
        assert separator == "::", f"{record.capability}: source {record.source!r} is not file::test"
        path = repo_root / file_part
        assert path.is_file(), f"{record.capability}: cited file {file_part} does not exist"
        contents = path.read_text()
        assert re.search(rf"^def {re.escape(test_name)}\(", contents, re.MULTILINE), (
            f"{record.capability}: cited test {test_name!r} not found in {file_part}"
        )


def test_known_capabilities_is_sorted_and_includes_the_static_catalog():
    names = known_capabilities()
    assert names == tuple(sorted(names))
    for record in STATIC_EVIDENCE_CATALOG:
        assert record.capability in names


# ---------------------------------------------------------------------------
# Query helpers: a capability can carry more than one tier of evidence at once.
# ---------------------------------------------------------------------------


def test_result_queries_capability_has_both_code_to_code_and_analytic_tier_evidence():
    capability = "mixle_pde.verification.result_queries (point/integral/extrema queries)"
    assert tiers_for_capability(capability) == frozenset({ValidationTier.CODE_TO_CODE, ValidationTier.ANALYTIC})

    report = evidence_report(capability)
    assert len(report.records) == 2
    assert {record.origin for record in report.records} == {"static_catalog"}
    # report.semantics() follows TIER_ORDER, not registration order.
    assert [entry.tier for entry in report.semantics()] == [ValidationTier.ANALYTIC, ValidationTier.CODE_TO_CODE]


def test_evidence_for_a_capability_with_nothing_recorded_is_honestly_empty():
    # mixle_pde.io.segy is a real ingest module (SEG-Y seismic, workstream B2) -- it consumes real
    # field data as an input but, per the MP-K5 ledger finding this ships to close, has zero
    # validation-tier evidence recorded anywhere yet. The query helpers must say so rather than
    # inventing a record.
    capability = "mixle_pde.io.segy"
    assert evidence_for_capability(capability) == ()
    assert tiers_for_capability(capability) == frozenset()
    report = evidence_report(capability)
    assert report.tiers == frozenset()
    assert report.records == ()
    assert report.semantics() == ()


# ---------------------------------------------------------------------------
# The `validation_evidence` decorator: registers only after a genuinely successful call.
#
# Each test below clears the dynamic registry on entry and exit and never inspects state left by
# another test item -- the registry is in-process, and this suite runs under pytest-xdist
# load-balanced distribution (tests/conftest.py), so two independently collected test items are not
# guaranteed to share a worker process.
# ---------------------------------------------------------------------------


def test_validation_evidence_decorator_registers_only_after_a_successful_call():
    clear_registry()
    capability = "tests.validation_tiers_test::dummy_pass"
    try:

        @validation_evidence(tier=ValidationTier.ANALYTIC, capability=capability, summary="dummy passing check")
        def _dummy_pass():
            return 42

        assert _dummy_pass() == 42

        records = evidence_for_capability(capability)
        assert len(records) == 1
        assert records[0].tier is ValidationTier.ANALYTIC
        assert records[0].origin == "dynamic_run"
        assert records[0].source.endswith("_dummy_pass")
        assert records[0] in registered_evidence()
    finally:
        clear_registry()


def test_validation_evidence_decorator_does_not_register_on_a_raising_call():
    clear_registry()
    capability = "tests.validation_tiers_test::dummy_fail"
    try:

        @validation_evidence(tier=ValidationTier.ANALYTIC, capability=capability, summary="dummy failing check")
        def _dummy_fail():
            raise ValueError("deliberate failure")

        with pytest.raises(ValueError, match="deliberate failure"):
            _dummy_fail()

        assert evidence_for_capability(capability) == ()
        assert registered_evidence() == ()
    finally:
        clear_registry()


# ---------------------------------------------------------------------------
# Worked, non-hypothetical example: tag a real, already-merged repo capability live.
# ---------------------------------------------------------------------------


def test_manufactured_solution_refinement_is_tagged_as_real_manufactured_tier_evidence():
    """Live proof, not a static claim: decorate and run an already-merged repo capability.

    ``manufactured_solution_refinement`` (``mixle_pde/multiphysics_reference.py``) is real,
    already-merged MP-K3-style method-of-manufactured-solutions code -- not a mock, not a
    hypothetical example. This test wraps it with ``validation_evidence``, calls it, and then
    confirms the registry/query helpers report exactly the tier that real run earned. The check and
    the registration both happen inside this one test function (see the isolation note above) so the
    result does not depend on xdist's worker assignment.
    """
    clear_registry()
    capability = "mixle_pde.multiphysics_reference (composite-heat P1 FEM reference, live run)"
    try:

        @validation_evidence(
            tier=ValidationTier.MANUFACTURED,
            capability=capability,
            summary=(
                "Live in-process re-run of manufactured_solution_refinement() (u(x)=x(1-x) against "
                "-u''=2 on [0,1]) -- a real, already-merged MP-K3-style manufactured-solution check, "
                "not a static citation."
            ),
        )
        def _run_manufactured_solution_refinement():
            return manufactured_solution_refinement()

        refinement = _run_manufactured_solution_refinement()
        # Same bound tests/multiphysics_reference_test.py already asserts for this exact call: P1
        # FEM's theoretical second-order L2 convergence, observed for real, not assumed.
        assert len(refinement.observed_orders) >= 1
        assert min(refinement.observed_orders) >= 1.9

        report = evidence_report(capability)
        assert report.tiers == frozenset({ValidationTier.MANUFACTURED})
        assert len(report.records) == 1
        record = report.records[0]
        assert record.origin == "dynamic_run"
        assert record.source.endswith("_run_manufactured_solution_refinement")
        assert semantics_for(ValidationTier.MANUFACTURED).gives_exact_error is True
    finally:
        clear_registry()
