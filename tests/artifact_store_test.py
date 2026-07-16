"""MP-K1: content-addressed local artifact store -- round trip, corruption detection, lineage query.

Source: docs/reconciliation/mp-task-ledger.md, MP-K1 -- "no put/get/blob-storage/lineage-query *store*
implementation exists anywhere" despite several sibling content-addressing digest *primitives*. This
suite is the acceptance evidence for mixle_pde.artifact_store.ArtifactStore closing that gap:

* :func:`test_put_get_round_trips_content_and_metadata` -- the basic put/get contract, including that
  the returned digest is exactly ``hashlib.sha256(content).hexdigest()`` (no invented hash scheme).
* :func:`test_get_detects_on_disk_corruption` -- flips a byte directly in the on-disk object file and
  asserts ``get`` raises a typed :class:`~mixle_pde.artifact_store.ArtifactIntegrityError` rather than
  returning the corrupted bytes.
* :func:`test_three_generation_lineage_chain` -- a put-time ``parents=`` chain three generations deep,
  queried both directions (``parents_of``/``children_of``), including a branching sibling to prove
  ``children_of`` aggregates every child rather than only the most recent one.
"""

from __future__ import annotations

import hashlib

import pytest

from mixle_pde.artifact_store import (
    ArtifactIntegrityError,
    ArtifactNotFoundError,
    ArtifactStore,
    digest_of,
)


def test_digest_of_matches_plain_hashlib_sha256() -> None:
    # The module docstring's central claim: the digest rule is exactly sha256 hex of raw bytes,
    # not a new or modified hash scheme layered on top of it.
    content = b"mixle-pde artifact store MP-K1"
    assert digest_of(content) == hashlib.sha256(content).hexdigest()


def test_put_get_round_trips_content_and_metadata(tmp_path) -> None:
    store = ArtifactStore(tmp_path / "store")
    content = b"scenario result: pressure field, 64x64 grid"
    metadata = {"schema": "mixle_pde.sim_result/v1", "units": "Pa", "grid": {"nx": 64, "ny": 64}}

    digest = store.put(content, metadata)

    assert digest == hashlib.sha256(content).hexdigest()
    assert store.exists(digest)
    assert store.get(digest) == content
    assert store.metadata(digest) == metadata
    assert store.parents_of(digest) == ()
    assert store.children_of(digest) == ()


def test_put_is_content_addressed_and_idempotent(tmp_path) -> None:
    store = ArtifactStore(tmp_path / "store")
    content = b"identical bytes, put twice"

    first = store.put(content, {"label": "first"})
    second = store.put(content, {"label": "second-should-not-overwrite"})

    assert first == second
    # First call's metadata wins; content addressing means a digest names exactly one object.
    assert store.metadata(first) == {"label": "first"}
    assert store.get(first) == content


def test_get_unknown_digest_raises_not_found(tmp_path) -> None:
    store = ArtifactStore(tmp_path / "store")
    missing = "0" * 64
    with pytest.raises(ArtifactNotFoundError):
        store.get(missing)


def test_get_rejects_malformed_digest(tmp_path) -> None:
    store = ArtifactStore(tmp_path / "store")
    for bad in ("not-hex", "abc", "F" * 64, "a" * 63):
        with pytest.raises(ValueError):
            store.get(bad)


def test_get_detects_on_disk_corruption(tmp_path) -> None:
    root = tmp_path / "store"
    store = ArtifactStore(root)
    content = b"a result later corrupted on disk"
    digest = store.put(content, {})

    # Reach directly into the documented on-disk layout (objects/<aa>/<rest>) and corrupt the bytes
    # exactly as real disk corruption or an out-of-band edit would -- not through the store API.
    object_path = root / "objects" / digest[:2] / digest[2:]
    assert object_path.read_bytes() == content
    object_path.write_bytes(b"corrupted bytes that do not match the digest")

    with pytest.raises(ArtifactIntegrityError) as excinfo:
        store.get(digest)
    assert excinfo.value.digest == digest
    assert excinfo.value.actual_digest != digest
    assert excinfo.value.actual_digest == digest_of(object_path.read_bytes())


def test_put_rejects_dangling_parent(tmp_path) -> None:
    store = ArtifactStore(tmp_path / "store")
    dangling_parent = "1" * 64
    with pytest.raises(ArtifactNotFoundError):
        store.put(b"child content", {}, parents=[dangling_parent])
    # Rejected puts must not partially land.
    assert not store.exists(digest_of(b"child content"))


def test_parents_of_and_children_of_unknown_digest_raise_not_found(tmp_path) -> None:
    store = ArtifactStore(tmp_path / "store")
    missing = "2" * 64
    with pytest.raises(ArtifactNotFoundError):
        store.parents_of(missing)
    with pytest.raises(ArtifactNotFoundError):
        store.children_of(missing)


def test_three_generation_lineage_chain(tmp_path) -> None:
    store = ArtifactStore(tmp_path / "store")

    # Generation 1: a root observation artifact with no parents.
    gen1 = store.put(b"gen1: raw field observation", {"stage": "observation"})

    # Generation 2: two independent derivations from gen1 (a branch), to prove children_of
    # aggregates every child rather than remembering only the latest one.
    gen2_a = store.put(b"gen2: calibrated field", {"stage": "calibration"}, parents=[gen1])
    gen2_b = store.put(b"gen2: coarse preview", {"stage": "preview"}, parents=[gen1])

    # Generation 3: an inversion result descending from the calibrated branch only.
    gen3 = store.put(b"gen3: inversion posterior", {"stage": "inversion"}, parents=[gen2_a])

    # Downward (parents_of): each generation points at exactly its own direct parent(s).
    assert store.parents_of(gen1) == ()
    assert store.parents_of(gen2_a) == (gen1,)
    assert store.parents_of(gen2_b) == (gen1,)
    assert store.parents_of(gen3) == (gen2_a,)

    # Upward (children_of): gen1 fans out to both gen2 branches; only the calibrated branch
    # continues to gen3; the leaf has no children.
    assert store.children_of(gen1) == tuple(sorted((gen2_a, gen2_b)))
    assert store.children_of(gen2_a) == (gen3,)
    assert store.children_of(gen2_b) == ()
    assert store.children_of(gen3) == ()


def test_put_unions_new_parents_into_existing_lineage_on_repeat_put(tmp_path) -> None:
    store = ArtifactStore(tmp_path / "store")
    parent_one = store.put(b"parent one", {})
    parent_two = store.put(b"parent two", {})
    child_content = b"shared child, put twice with different parents"

    first = store.put(child_content, {}, parents=[parent_one])
    second = store.put(child_content, {}, parents=[parent_two])

    assert first == second
    assert store.parents_of(first) == tuple(sorted((parent_one, parent_two)))
    assert store.children_of(parent_one) == (first,)
    assert store.children_of(parent_two) == (first,)
