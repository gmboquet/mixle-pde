"""Tests for the MP-J5 structured knowledge handoff (mixle_pde.verification.knowledge_handoff).

``test_three_context_handoff_drafts_resolves_and_audits_without_flattening_the_model`` is the accept-bar
scenario itself: model A drafts an inverse case with a real, deliberately-missing required field
(gravity inversion needs grid.cell_volumes -- mixle_pde.tools.run_inversion's own real ValueError names
this), model B resolves the gap and actually runs a real mixle_pde.tools.run_inversion, and model C reads
the resulting posterior back through mixle_pde.tools.query_posterior (a real decision quantity) and
audits a real, distilled mixle_pde.surrogate.Surrogate's own calibration report -- every step reads its
input back from a content hash via KnowledgeStore, never from a Python object a previous step is still
holding, and every numeric claim is checked against the real tool's actual return value.
"""

from __future__ import annotations

import json
import math

import pytest

from mixle_pde import tools
from mixle_pde.artifact_store import ArtifactStore
from mixle_pde.surrogate import Surrogate, distill_forward
from mixle_pde.verification.knowledge_handoff import (
    KnowledgeBundle,
    KnowledgeItem,
    KnowledgeKind,
    KnowledgeStore,
    compute_delta,
)


@pytest.fixture
def store(tmp_path) -> KnowledgeStore:
    return KnowledgeStore(ArtifactStore(tmp_path / "objects"))


# ---------------------------------------------------------------------------
# KnowledgeItem / revision chaining
# ---------------------------------------------------------------------------
def test_put_item_assigns_sequential_revisions_and_chains_parents(store):
    first = store.put_item(KnowledgeKind.DATASET, {"a": 1}, logical_id="d", produced_by="model_a")
    second = store.put_item(KnowledgeKind.DATASET, {"a": 1, "b": 2}, logical_id="d", produced_by="model_b")
    assert first.revision == 1
    assert second.revision == 2
    assert first.content_hash != second.content_hash  # different payload, different digest
    assert second.parents == (first.content_hash,)
    assert store.history("d") == (first.content_hash, second.content_hash)


def test_put_item_dedupes_an_explicitly_supplied_parent_equal_to_the_previous_revision(store):
    first = store.put_item(KnowledgeKind.DATASET, {"a": 1}, logical_id="d", produced_by="model_a")
    second = store.put_item(
        KnowledgeKind.DATASET, {"a": 2}, logical_id="d", produced_by="model_b", derived_from=(first.content_hash,)
    )
    assert second.parents == (first.content_hash,)  # not duplicated


def test_get_item_payload_round_trips(store):
    item = store.put_item(
        KnowledgeKind.MODEL_SPEC, {"nested": [1, 2, {"x": "y"}]}, logical_id="m", produced_by="model_a"
    )
    assert store.get_item_payload(item) == {"nested": [1, 2, {"x": "y"}]}


def test_identical_payload_under_different_logical_ids_shares_a_content_hash_but_not_a_revision_chain(store):
    """ArtifactStore is pure content addressing: identical bytes always land on the identical digest --
    this module's logical_id/revision layer sits *on top of* that, it does not defeat it. Each logical_id
    still gets its own independent revision history, even though both histories currently point at the
    same underlying content hash."""
    a = store.put_item(KnowledgeKind.ASSUMPTION, {"x": 1}, logical_id="assume_a", produced_by="model_a")
    b = store.put_item(KnowledgeKind.ASSUMPTION, {"x": 1}, logical_id="assume_b", produced_by="model_a")
    assert a.content_hash == b.content_hash
    assert a.revision == 1 and b.revision == 1
    assert store.history("assume_a") == (a.content_hash,)
    assert store.history("assume_b") == (b.content_hash,)


# ---------------------------------------------------------------------------
# KnowledgeBundle
# ---------------------------------------------------------------------------
def test_bundle_rejects_duplicate_logical_ids():
    item = KnowledgeItem(
        kind=KnowledgeKind.DATASET,
        logical_id="d",
        content_hash="x" * 64,
        revision=1,
        parents=(),
        produced_by="m",
        metadata={},
    )
    with pytest.raises(ValueError):
        KnowledgeBundle(bundle_id="b", items=(item, item))


def test_save_and_load_bundle_round_trips(store):
    item1 = store.put_item(
        KnowledgeKind.MODEL_SPEC, {"kind": "elastic"}, logical_id="model_spec", produced_by="model_a"
    )
    item2 = store.put_item(
        KnowledgeKind.UNRESOLVED_QUANTITY, {"missing": "vp"}, logical_id="gap_vp", produced_by="model_a"
    )
    bundle_hash = store.save_bundle("draft_v1", [item1, item2])

    reloaded = store.load_bundle(bundle_hash)
    assert reloaded.bundle_id == "draft_v1"
    assert reloaded.get("model_spec") == item1
    assert reloaded.get("gap_vp") == item2
    assert reloaded.by_kind(KnowledgeKind.UNRESOLVED_QUANTITY) == (item2,)
    assert reloaded.get("does_not_exist") is None


def test_load_bundle_from_a_second_independent_knowledge_store_over_the_same_backing_store(tmp_path):
    """The actual cross-context handoff: a *different* KnowledgeStore instance (representing a different
    model context, with an empty in-memory revision index of its own) reconstructs the full bundle from
    nothing but the content hash and the shared backing ArtifactStore."""
    backing = ArtifactStore(tmp_path / "objects")
    store_a = KnowledgeStore(backing)
    item = store_a.put_item(KnowledgeKind.DATASET, {"grid": "spec"}, logical_id="dataset", produced_by="model_a")
    bundle_hash = store_a.save_bundle("draft_v1", [item])

    store_b = KnowledgeStore(backing)  # model B's own context: never touched store_a's Python objects
    assert store_b.history("dataset") == ()  # genuinely empty -- no shared in-memory state
    bundle = store_b.load_bundle(bundle_hash)
    assert bundle.get("dataset").content_hash == item.content_hash
    assert store_b.get_item_payload(bundle.get("dataset")) == {"grid": "spec"}


# ---------------------------------------------------------------------------
# compute_delta
# ---------------------------------------------------------------------------
def test_compute_delta_reports_added_removed_and_modified(store):
    a1 = store.put_item(KnowledgeKind.DATASET, {"v": 1}, logical_id="dataset", produced_by="model_a")
    gap = store.put_item(
        KnowledgeKind.UNRESOLVED_QUANTITY, {"missing": "cell_volumes"}, logical_id="gap", produced_by="model_a"
    )
    base = KnowledgeBundle(bundle_id="draft", items=(a1, gap))

    a2 = store.put_item(
        KnowledgeKind.DATASET, {"v": 1, "cell_volumes": [1.0]}, logical_id="dataset", produced_by="model_b"
    )
    posterior = store.put_item(
        KnowledgeKind.POSTERIOR_ARTIFACT, {"posterior_ref": "p"}, logical_id="posterior", produced_by="model_b"
    )
    target = KnowledgeBundle(bundle_id="resolved", items=(a2, posterior))

    delta = compute_delta(base, target)
    assert delta.added == (posterior,)
    assert delta.removed == ("gap",)
    assert delta.modified == ((a1, a2),)
    assert not delta.is_empty


def test_compute_delta_is_empty_for_identical_bundles(store):
    item = store.put_item(KnowledgeKind.FIELD, {"v": 1}, logical_id="f", produced_by="model_a")
    bundle = KnowledgeBundle(bundle_id="b", items=(item,))
    delta = compute_delta(bundle, bundle)
    assert delta.is_empty


# ---------------------------------------------------------------------------
# Three-context handoff: model A drafts, model B resolves + runs, model C audits.
# ---------------------------------------------------------------------------
def _gravity_dataset_bundle(*, with_cell_volumes: bool) -> dict:
    bundle = {
        "grid": {
            "coordinates": [[0.0, 0.0, -10.0], [10.0, 0.0, -10.0], [0.0, 10.0, -10.0], [10.0, 10.0, -10.0]],
            "spacing": [10.0, 10.0, 10.0],
            "units": "kg/m3",
            "property_name": "density",
        },
        "observations": [
            {
                "location": [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [5.0, 5.0, 0.0]],
                "value": [0.02, 0.015, 0.03],
                "noise_cov": [1e-4, 1e-4, 1e-4],
                "units": "mGal",
            }
        ],
    }
    if with_cell_volumes:
        bundle["grid"]["cell_volumes"] = [1000.0, 1000.0, 1000.0, 1000.0]
    return bundle


def _analytic_surrogate_teacher(x):
    a, b = x
    return math.sin(a) + 0.5 * b * b


def _uniform_sampler(lo, hi):
    def sampler(n, rng):
        return rng.uniform(lo, hi, size=(n, 2))

    return sampler


def test_three_context_handoff_drafts_resolves_and_audits_without_flattening_the_model(tmp_path):
    backing = ArtifactStore(tmp_path / "objects")

    # --- Model A: drafts a multiphysics inverse case, with a real, honestly-flagged gap ---
    store_a = KnowledgeStore(backing)
    incomplete_dataset = _gravity_dataset_bundle(with_cell_volumes=False)
    dataset_item_v1 = store_a.put_item(
        KnowledgeKind.DATASET, incomplete_dataset, logical_id="dataset", produced_by="model_a_draft"
    )
    gap_item = store_a.put_item(
        KnowledgeKind.UNRESOLVED_QUANTITY,
        {"field": "grid.cell_volumes", "reason": "required by run_inversion for modality=gravity"},
        logical_id="gap_cell_volumes",
        produced_by="model_a_draft",
        derived_from=(dataset_item_v1.content_hash,),
    )
    draft_hash = store_a.save_bundle("draft_v1", [dataset_item_v1, gap_item])

    # Confirm the flagged gap is real: mixle_pde.tools.run_inversion actually rejects the v1 dataset.
    dataset_path_v1 = tmp_path / "dataset_v1.json"
    dataset_path_v1.write_text(json.dumps(incomplete_dataset))
    with pytest.raises(ValueError, match="cell_volumes"):
        tools.run_inversion(str(dataset_path_v1), "gravity", "smooth")

    # --- Model B: independently reloads the draft, resolves the gap, and actually runs the inversion ---
    store_b = KnowledgeStore(backing)
    draft_bundle = store_b.load_bundle(draft_hash)
    assert store_b.history("dataset") == ()  # model B starts with no in-memory state of its own
    flagged_gap = draft_bundle.get("gap_cell_volumes")
    assert store_b.get_item_payload(flagged_gap)["field"] == "grid.cell_volumes"

    resolved_dataset = _gravity_dataset_bundle(with_cell_volumes=True)
    dataset_item_v2 = store_b.put_item(
        KnowledgeKind.DATASET,
        resolved_dataset,
        logical_id="dataset",
        produced_by="model_b_gap_resolution",
        derived_from=(draft_bundle.get("dataset").content_hash, flagged_gap.content_hash),
    )
    # revision numbering is per-KnowledgeStore-instance (see the class docstring: the in-memory
    # _revisions index is same-process convenience, not a second source of truth), so model B's own
    # fresh instance legitimately starts counting "dataset" revisions at 1 again here -- the real,
    # global lineage is carried by `parents`/content hashes, not by this locally-numbered counter.
    assert dataset_item_v2.revision == 1
    assert flagged_gap.content_hash in dataset_item_v2.parents

    dataset_path_v2 = tmp_path / "dataset_v2.json"
    dataset_path_v2.write_text(json.dumps(resolved_dataset))
    inversion_result = tools.run_inversion(str(dataset_path_v2), "gravity", "smooth")
    posterior_item = store_b.put_item(
        KnowledgeKind.POSTERIOR_ARTIFACT,
        {"posterior_ref": inversion_result["posterior_ref"], "diagnostics": inversion_result["diagnostics"]},
        logical_id="posterior",
        produced_by="model_b_gap_resolution",
        derived_from=(dataset_item_v2.content_hash,),
    )
    resolved_hash = store_b.save_bundle("resolved_v2", [dataset_item_v2, posterior_item], derived_from=(draft_hash,))

    # --- Model C: independently reloads the resolved bundle, audits the posterior and a surrogate ---
    store_c = KnowledgeStore(backing)
    resolved_bundle = store_c.load_bundle(resolved_hash)
    posterior_payload = store_c.get_item_payload(resolved_bundle.get("posterior"))
    region_mass = tools.query_posterior(
        posterior_payload["posterior_ref"], "region_mass", {"region_index": [0, 1], "seed": 0}
    )
    assert "value" in region_mass and math.isfinite(region_mass["value"])

    surrogate: Surrogate = distill_forward(
        _analytic_surrogate_teacher, _uniform_sampler(-2.0, 2.0), budget=80, seed=0, holdout=0.3
    )
    surrogate_report = surrogate.report()
    surrogate_item = store_c.put_item(
        KnowledgeKind.SURROGATE_ARTIFACT, surrogate_report, logical_id="surrogate", produced_by="model_c_audit"
    )
    audit_report_item = store_c.put_item(
        KnowledgeKind.VERIFICATION_REPORT,
        {"posterior_region_mass": region_mass["value"], "surrogate_imprecise": surrogate_report["imprecise"]},
        logical_id="audit_report",
        produced_by="model_c_audit",
        derived_from=(resolved_bundle.get("posterior").content_hash, surrogate_item.content_hash),
    )
    audited_hash = store_c.save_bundle(
        "audited_v3",
        [resolved_bundle.get("dataset"), resolved_bundle.get("posterior"), surrogate_item, audit_report_item],
        derived_from=(resolved_hash,),
    )

    # Real assertions, not merely "no exception": the surrogate genuinely calibrated (qhat <= tol
    # everywhere, same as tests/test_e6_surrogate.py::test_surrogate_is_not_globally_imprecise), and the
    # deltas between drafts trace exactly the handoff that happened.
    assert surrogate_report["imprecise"] is False

    draft_to_resolved = compute_delta(draft_bundle, store_c.load_bundle(resolved_hash))
    assert {item.logical_id for item in draft_to_resolved.added} == {"posterior"}
    assert draft_to_resolved.removed == ("gap_cell_volumes",)
    assert {old.logical_id for old, _new in draft_to_resolved.modified} == {"dataset"}

    resolved_to_audited = compute_delta(store_c.load_bundle(resolved_hash), store_c.load_bundle(audited_hash))
    assert {item.logical_id for item in resolved_to_audited.added} == {"surrogate", "audit_report"}
    assert not resolved_to_audited.removed
    assert not resolved_to_audited.modified
