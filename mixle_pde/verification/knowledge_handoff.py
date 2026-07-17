"""Structured knowledge handoff: typed, content-addressed, revisioned knowledge items/bundles/deltas
across model contexts (MP-J5).

MP-J5's full description: "Represent model specs, datasets, observations, meshes, fields, diagnostics,
verification reports, posterior artifacts, surrogate artifacts, assumptions, conflicts, and unresolved
quantities as IC-13 knowledge items/bundles/deltas. Preserve revisions, typed relationships, partial
availability, and content hashes across model contexts." Accept bar: "model A drafts a multiphysics
inverse case, model B resolves a material/BC/data gap and runs it, and model C audits the posterior and
surrogate validity without flattening the model into text."

This module is a deliberately narrow baseline slice, matching this same MP-J/MP-K/MP-N family's
established convention (``knowledge_catalog.py``/MP-J2, ``diagnostic_ontology.py``/MP-J3,
``agent_loop.py``/MP-J4, ``job_governance.py``/MP-L4): it builds directly on
:class:`mixle_pde.artifact_store.ArtifactStore` (MP-K1, already landed) rather than reinventing
content-addressing, blob storage, or lineage tracking -- MP-K1's own ``put``/``get``/``parents_of``/
``children_of`` already give exactly the "content hashes" and one form of "typed relationships" (a
derivation edge) this card asks for. What this module adds *on top of* that already-real store:

* :class:`KnowledgeKind` -- a closed, twelve-member enum naming exactly the twelve item categories this
  card's own sentence lists (model spec, dataset, observation, mesh, field, diagnostic, verification
  report, posterior artifact, surrogate artifact, assumption, conflict, unresolved quantity). A caller
  cannot silently invent a thirteenth kind.
* :class:`KnowledgeItem` -- one typed, content-hashed record: which kind, a caller-chosen stable
  ``logical_id`` (the "same slot across revisions" name a raw content hash alone cannot give you --
  content addressing means two revisions of the same thing hash differently and share no name unless
  something tracks that), a 1-indexed ``revision`` counter, a ``parents`` tuple of the content hashes it
  was derived from, and a required ``produced_by`` provenance tag.
* :class:`KnowledgeStore` -- wraps one :class:`~mixle_pde.artifact_store.ArtifactStore`.
  :meth:`~KnowledgeStore.put_item` JSON-canonicalizes a payload, stores it, and automatically chains a new
  revision's ``parents`` to the previous revision under the same ``logical_id`` (so a revision history is
  reconstructable from real store lineage, not only from this object's own in-memory index).
  :meth:`~KnowledgeStore.save_bundle`/:meth:`~KnowledgeStore.load_bundle` persist/reload a
  :class:`KnowledgeBundle` (a named, ordered snapshot of items) as its own content-addressed manifest
  artifact -- the actual cross-context handoff mechanism: "model B" and "model C" reconstruct their view
  entirely from a bundle content hash plus the shared store, never from a live Python object "model A"
  is still holding, which is what "without flattening the model into text" means here: the handoff unit
  is a typed record tree, not a serialized natural-language description of one.
* :func:`compute_delta` -- a pure function comparing two bundles' ``logical_id -> content_hash`` maps into
  a typed :class:`KnowledgeDelta` (added/removed/modified), satisfying "deltas" and "preserve revisions"
  without any new storage mechanism.
* "Partial availability" is represented the same way this program represents every other kind of
  incompleteness (per its standing "unknown/timeout/resource_limit are valid typed outcomes, never a
  fabricated success" convention): a gap is a real, present
  :attr:`KnowledgeKind.UNRESOLVED_QUANTITY`/:attr:`KnowledgeKind.CONFLICT` item describing what is
  missing, never a silently absent field or an invented placeholder value.

Explicitly NOT attempted here (matching this family's "narrow, honest slice" convention): a
network/service-backed multi-process store (this module and MP-K1's own ``ArtifactStore`` are both
local-filesystem-backed; a real multi-context handoff would put the same backing store behind a shared
service, which is separate, unclaimed infrastructure work); relationship *kinds* beyond a plain derivation
edge (e.g. "resolves-gap-in" vs. "audits" as distinct typed edge labels -- ``parents`` here is one
undifferentiated lineage tuple, exactly what MP-K1's own ``parents``/``children_of`` already expose);
schema validation of a payload's *contents* against a kind-specific shape (a ``KnowledgeItem`` accepts any
JSON-serializable payload for its declared ``kind`` -- this module checks kind membership and JSON
round-tripping, not that a ``MODEL_SPEC`` payload is itself a well-formed model spec, which is a different,
already-owned validation concern -- e.g. :mod:`mixle_pde.problem_adapter` for a PDE study).

Dependency direction: this module imports only :mod:`mixle_pde.artifact_store` (MP-K1) at module level,
read-only (:class:`~mixle_pde.artifact_store.ArtifactStore` is never modified here). It has no opinion
about which capability *produced* an item's payload -- ``tests/knowledge_handoff_test.py`` demonstrates a
full three-context handoff using several already-real, already-merged capabilities
(:mod:`mixle_pde.tools`'s IC-3 inversion tools/MP-J1, :mod:`mixle_pde.surrogate`'s distillation/MP-N6) as
test-only dependencies, never as a production dependency of this module itself.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

from mixle_pde.artifact_store import ArtifactStore

__all__ = [
    "KnowledgeKind",
    "KnowledgeItem",
    "KnowledgeBundle",
    "KnowledgeDelta",
    "KnowledgeStore",
    "compute_delta",
]


class KnowledgeKind(Enum):
    """The twelve item categories MP-J5's own description names, verbatim -- a closed set."""

    MODEL_SPEC = "model_spec"
    DATASET = "dataset"
    OBSERVATION = "observation"
    MESH = "mesh"
    FIELD = "field"
    DIAGNOSTIC = "diagnostic"
    VERIFICATION_REPORT = "verification_report"
    POSTERIOR_ARTIFACT = "posterior_artifact"
    SURROGATE_ARTIFACT = "surrogate_artifact"
    ASSUMPTION = "assumption"
    CONFLICT = "conflict"
    UNRESOLVED_QUANTITY = "unresolved_quantity"


def _canonical_json_bytes(payload: Any) -> bytes:
    """The one payload encoding this module uses: deterministic (sorted keys, no incidental whitespace)
    so two callers that build the same logical payload always get the same content hash."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


@dataclass(frozen=True)
class KnowledgeItem:
    """One typed, content-hashed knowledge record.

    ``content_hash`` is the payload's address in the backing :class:`~mixle_pde.artifact_store.ArtifactStore`
    (fetch it with :meth:`KnowledgeStore.get_item_payload`), never the payload itself -- this dataclass
    carries no live payload, exactly the "content hashes, not text" handoff shape this card asks for.
    """

    kind: KnowledgeKind
    logical_id: str
    content_hash: str
    revision: int
    parents: tuple[str, ...]
    produced_by: str
    metadata: Mapping[str, Any]

    def to_record(self) -> dict[str, Any]:
        """A plain, JSON-serializable dict form -- what :class:`KnowledgeStore` persists inside a bundle
        manifest and reloads via :meth:`from_record`."""
        return {
            "kind": self.kind.value,
            "logical_id": self.logical_id,
            "content_hash": self.content_hash,
            "revision": self.revision,
            "parents": list(self.parents),
            "produced_by": self.produced_by,
            "metadata": dict(self.metadata),
        }

    @staticmethod
    def from_record(record: Mapping[str, Any]) -> KnowledgeItem:
        return KnowledgeItem(
            kind=KnowledgeKind(record["kind"]),
            logical_id=record["logical_id"],
            content_hash=record["content_hash"],
            revision=int(record["revision"]),
            parents=tuple(record["parents"]),
            produced_by=record["produced_by"],
            metadata=dict(record["metadata"]),
        )


@dataclass(frozen=True)
class KnowledgeBundle:
    """A named, ordered snapshot of :class:`KnowledgeItem` records -- one "handoff" between contexts.

    At most one item per ``logical_id`` -- a bundle is a *snapshot* (one revision per slot), not a full
    history; :meth:`KnowledgeStore.history` gives the full revision history for one ``logical_id`` when
    that is what a caller actually wants.
    """

    bundle_id: str
    items: tuple[KnowledgeItem, ...]

    def __post_init__(self) -> None:
        logical_ids = [item.logical_id for item in self.items]
        if len(logical_ids) != len(set(logical_ids)):
            raise ValueError(f"bundle {self.bundle_id!r} has more than one item for the same logical_id: {logical_ids}")

    def get(self, logical_id: str) -> KnowledgeItem | None:
        for item in self.items:
            if item.logical_id == logical_id:
                return item
        return None

    def by_kind(self, kind: KnowledgeKind) -> tuple[KnowledgeItem, ...]:
        return tuple(item for item in self.items if item.kind is kind)

    def as_index(self) -> dict[str, KnowledgeItem]:
        return {item.logical_id: item for item in self.items}


@dataclass(frozen=True)
class KnowledgeDelta:
    """What changed between two :class:`KnowledgeBundle` snapshots, by ``logical_id``."""

    base_bundle_id: str
    target_bundle_id: str
    added: tuple[KnowledgeItem, ...]
    removed: tuple[str, ...]
    modified: tuple[tuple[KnowledgeItem, KnowledgeItem], ...]

    @property
    def is_empty(self) -> bool:
        return not (self.added or self.removed or self.modified)


def compute_delta(base: KnowledgeBundle, target: KnowledgeBundle) -> KnowledgeDelta:
    """Compare two bundles' ``logical_id -> content_hash`` maps -- a pure function, no store access."""
    base_index = base.as_index()
    target_index = target.as_index()
    added = tuple(item for logical_id, item in target_index.items() if logical_id not in base_index)
    removed = tuple(logical_id for logical_id in base_index if logical_id not in target_index)
    modified = tuple(
        (base_index[logical_id], target_index[logical_id])
        for logical_id in base_index
        if logical_id in target_index and base_index[logical_id].content_hash != target_index[logical_id].content_hash
    )
    return KnowledgeDelta(
        base_bundle_id=base.bundle_id,
        target_bundle_id=target.bundle_id,
        added=added,
        removed=removed,
        modified=modified,
    )


class KnowledgeStore:
    """A typed knowledge-item/bundle layer over one :class:`~mixle_pde.artifact_store.ArtifactStore`.

    The ``_revisions`` index (``logical_id -> [content_hash, ...]``, oldest first) is this object's own
    in-memory bookkeeping, not a second source of truth: :class:`~mixle_pde.artifact_store.ArtifactStore`
    has no notion of "logical_id" at all (it is purely content-addressed), so *some* external index from a
    stable name to its current hash is unavoidable -- this is that index, kept only for this process's
    convenience. The *durable*, cross-context handoff path is :meth:`save_bundle`/:meth:`load_bundle`: a
    bundle manifest is itself a content-addressed artifact, so a different :class:`KnowledgeStore`
    instance (a different "model context") sharing the same backing store can reconstruct the full bundle
    from its hash alone, without ever touching this instance's in-memory index.
    """

    def __init__(self, artifact_store: ArtifactStore) -> None:
        self._store = artifact_store
        self._revisions: dict[str, list[str]] = {}

    def put_item(
        self,
        kind: KnowledgeKind,
        payload: Any,
        *,
        logical_id: str,
        produced_by: str,
        derived_from: Sequence[str] = (),
        metadata: Mapping[str, Any] | None = None,
    ) -> KnowledgeItem:
        """Store ``payload`` (any JSON-serializable value) as a new revision under ``logical_id``.

        The previous revision under the same ``logical_id`` (if any) is always folded into ``parents``
        (deduplicated against any explicitly-supplied ``derived_from``), so the revision chain is real
        :class:`~mixle_pde.artifact_store.ArtifactStore` lineage, inspectable via ``parents_of`` on the
        backing store, not only through this object's own index.
        """
        encoded = _canonical_json_bytes(payload)
        history = self._revisions.setdefault(logical_id, [])
        previous = (history[-1],) if history else ()
        parents = tuple(dict.fromkeys((*derived_from, *previous)))  # dedupe, preserve order
        content_hash = self._store.put(
            encoded, metadata={"logical_id": logical_id, "kind": kind.value}, parents=parents
        )
        revision = len(history) + 1
        history.append(content_hash)
        return KnowledgeItem(
            kind=kind,
            logical_id=logical_id,
            content_hash=content_hash,
            revision=revision,
            parents=parents,
            produced_by=produced_by,
            metadata=dict(metadata or {}),
        )

    def get_item_payload(self, item: KnowledgeItem) -> Any:
        """Read back and JSON-decode ``item``'s payload -- raises
        :class:`~mixle_pde.artifact_store.ArtifactIntegrityError` (via the backing store's own ``get``) if
        the stored bytes no longer hash to ``item.content_hash``, never a silently corrupted decode."""
        return json.loads(self._store.get(item.content_hash).decode("utf-8"))

    def history(self, logical_id: str) -> tuple[str, ...]:
        """Every content hash ever stored under ``logical_id`` through this instance, oldest first."""
        return tuple(self._revisions.get(logical_id, ()))

    def latest_hash(self, logical_id: str) -> str | None:
        history = self._revisions.get(logical_id)
        return history[-1] if history else None

    def save_bundle(self, bundle_id: str, items: Sequence[KnowledgeItem], *, derived_from: Sequence[str] = ()) -> str:
        """Persist a :class:`KnowledgeBundle` manifest (the list of item records) as one content-addressed
        artifact and return its digest -- the value a different context passes to :meth:`load_bundle`."""
        bundle = KnowledgeBundle(bundle_id=bundle_id, items=tuple(items))  # validates no duplicate logical_id
        manifest = {"bundle_id": bundle_id, "items": [item.to_record() for item in bundle.items]}
        encoded = _canonical_json_bytes(manifest)
        return self._store.put(
            encoded, metadata={"bundle_id": bundle_id, "kind": "bundle"}, parents=tuple(derived_from)
        )

    def load_bundle(self, content_hash: str) -> KnowledgeBundle:
        """Reconstruct a :class:`KnowledgeBundle` entirely from ``content_hash`` and the backing store --
        the cross-context handoff read path (see the class docstring)."""
        manifest = json.loads(self._store.get(content_hash).decode("utf-8"))
        items = tuple(KnowledgeItem.from_record(record) for record in manifest["items"])
        return KnowledgeBundle(bundle_id=manifest["bundle_id"], items=items)
