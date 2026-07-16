"""MP-K1: content-addressed local artifact store (put/get/lineage-query).

The M2 reconciliation ledger (``docs/reconciliation/mp-task-ledger.md``, row ``MP-K1``) records that
mixle-pde and mixle-discrete already carry strong content-addressing *primitives* --
``canonical_adapter.py``'s digest of a normalized linear-system record, ``io/artifacts.py``'s
``sha256_of_arrays``, ``ownership.py``'s ``migration_inventory_digest``, mixle-discrete's
``math_ir.semantic_digest`` -- but "no ``put``/``get``/blob-storage/lineage-query *store* implementation
exists anywhere". This module is that store.

Every digest is ``hashlib.sha256(content).hexdigest()`` of the raw content bytes: the exact hex-digest
convention every primitive named above already uses, applied directly to bytes rather than to a
JSON-normalized structure or a named-array collection. There is nothing to normalize here -- ``put``
takes raw bytes -- so this is that same convention's most primitive form, not a new hash scheme.

Layout under ``root`` (a two-character digest-prefix directory fan-out, the same shape
``.git/objects/`` uses)::

    objects/<aa>/<rest>            raw content bytes for digest "aa"+"rest"
    objects/<aa>/<rest>.json       ArtifactRecord sidecar: metadata, parents, size, created
    children/<aa>/<rest>/<child>   one empty marker file per direct child digest

Content addressing is over raw bytes only: identical content always resolves to the identical digest
regardless of what metadata or lineage it is put with. ``get`` re-hashes the bytes it reads back and
raises :class:`ArtifactIntegrityError` on any mismatch rather than silently returning them -- corruption
detection is a first-class outcome, not an afterthought.

Lineage is a caller-supplied edge list recorded at put-time (``put(..., parents=(...))``), not something
this module infers. ``parents_of``/``children_of`` are the query surface: ``parents_of`` is a direct
sidecar read; ``children_of`` is a reverse-index directory listing so it does not require scanning every
object in the store. Re-putting already-stored content unions any newly supplied parents into the
existing lineage (edges accumulate; they are never dropped) while keeping the first call's metadata.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")
_PREFIX_LEN = 2


def digest_of(content: bytes) -> str:
    """The store's one hashing rule: the sha256 hex digest of raw content bytes."""
    if not isinstance(content, (bytes, bytearray)):
        raise TypeError(f"content must be bytes, got {type(content).__name__}")
    return hashlib.sha256(bytes(content)).hexdigest()


class ArtifactNotFoundError(KeyError):
    """Raised when a well-formed digest has no corresponding object in the store."""

    def __init__(self, digest: str) -> None:
        self.digest = digest
        super().__init__(f"no artifact stored for digest {digest}")


class ArtifactIntegrityError(ValueError):
    """Raised when bytes read back from disk no longer hash to the digest that named them.

    ``get`` always re-hashes before returning content; this error means the on-disk object was
    corrupted, truncated, or tampered with after ``put`` wrote it. It is never raised in place of a
    silent, corrupted return value.
    """

    def __init__(self, digest: str, actual_digest: str) -> None:
        self.digest = digest
        self.actual_digest = actual_digest
        super().__init__(
            f"artifact {digest} failed integrity verification: stored bytes hash to {actual_digest} instead"
        )


@dataclass(frozen=True)
class ArtifactRecord:
    """The metadata sidecar ``put`` writes alongside an object's content."""

    digest: str
    metadata: Mapping[str, Any]
    parents: tuple[str, ...]
    size: int
    created: str


def _validate_digest(digest: str) -> str:
    if not isinstance(digest, str) or not _DIGEST_RE.fullmatch(digest):
        raise ValueError(f"digest must be a 64-character lowercase hex sha256 string, got {digest!r}")
    return digest


def _atomic_write(path: str, data: bytes) -> None:
    """Write ``data`` to ``path`` via a same-directory temp file + ``os.replace`` so a reader never
    observes a partially written object."""
    tmp_path = f"{path}.tmp.{os.getpid()}.{time.monotonic_ns()}"
    with open(tmp_path, "wb") as handle:
        handle.write(data)
    os.replace(tmp_path, path)


class ArtifactStore:
    """Local filesystem-backed content-addressed artifact store. Not a distributed system: one
    directory tree, one process's view of it at a time."""

    def __init__(self, root: str | os.PathLike[str]) -> None:
        self.root = str(root)
        self._objects_dir = os.path.join(self.root, "objects")
        self._children_dir = os.path.join(self.root, "children")
        os.makedirs(self._objects_dir, exist_ok=True)
        os.makedirs(self._children_dir, exist_ok=True)

    # -- put / get --------------------------------------------------------------------------------

    def put(self, content: bytes, metadata: Mapping[str, Any] | None = None, *, parents: Sequence[str] = ()) -> str:
        """Store ``content``, returning its sha256 digest. ``parents`` records lineage edges from each
        already-stored parent digest to this content's digest; every parent must already exist in the
        store (:class:`ArtifactNotFoundError` otherwise) -- lineage never references content that was
        never put. Re-putting identical bytes is idempotent for the content itself and additive for
        lineage: any parents not already recorded are unioned in."""
        if not isinstance(content, (bytes, bytearray)):
            raise TypeError(f"content must be bytes, got {type(content).__name__}")
        content = bytes(content)
        digest = digest_of(content)
        parent_digests = tuple(_validate_digest(parent) for parent in parents)
        for parent in parent_digests:
            if not self.exists(parent):
                raise ArtifactNotFoundError(parent)

        object_path = self._object_path(digest)
        os.makedirs(os.path.dirname(object_path), exist_ok=True)

        if os.path.exists(object_path):
            existing = self._read_record(digest)
            merged_parents = tuple(sorted(set(existing.parents) | set(parent_digests)))
            if merged_parents != existing.parents:
                self._write_record(
                    ArtifactRecord(
                        digest=digest,
                        metadata=existing.metadata,
                        parents=merged_parents,
                        size=existing.size,
                        created=existing.created,
                    )
                )
        else:
            _atomic_write(object_path, content)
            self._write_record(
                ArtifactRecord(
                    digest=digest,
                    metadata=dict(metadata or {}),
                    parents=tuple(sorted(set(parent_digests))),
                    size=len(content),
                    created=datetime.now(timezone.utc).isoformat(),
                )
            )

        for parent in parent_digests:
            self._link_child(parent=parent, child=digest)
        return digest

    def get(self, digest: str) -> bytes:
        """Read back ``digest``'s content, re-hashing it before returning it. Raises
        :class:`ArtifactIntegrityError` -- never a silently corrupted result -- if the bytes on disk no
        longer hash to ``digest``."""
        digest = _validate_digest(digest)
        object_path = self._object_path(digest)
        if not os.path.exists(object_path):
            raise ArtifactNotFoundError(digest)
        with open(object_path, "rb") as handle:
            content = handle.read()
        actual = digest_of(content)
        if actual != digest:
            raise ArtifactIntegrityError(digest, actual)
        return content

    def metadata(self, digest: str) -> Mapping[str, Any]:
        """The metadata dict recorded by the ``put`` call that first stored ``digest``."""
        return self._read_record(digest).metadata

    def exists(self, digest: str) -> bool:
        digest = _validate_digest(digest)
        return os.path.exists(self._object_path(digest))

    # -- lineage query ------------------------------------------------------------------------------

    def parents_of(self, digest: str) -> tuple[str, ...]:
        """Direct parent digests recorded across every ``put`` call for ``digest``, sorted."""
        return self._read_record(digest).parents

    def children_of(self, digest: str) -> tuple[str, ...]:
        """Direct child digests -- objects put with ``digest`` named in their ``parents`` -- sorted."""
        digest = _validate_digest(digest)
        if not self.exists(digest):
            raise ArtifactNotFoundError(digest)
        children_dir = self._children_index_dir(digest)
        if not os.path.isdir(children_dir):
            return ()
        return tuple(sorted(os.listdir(children_dir)))

    # -- path / record internals ---------------------------------------------------------------------

    def _object_path(self, digest: str) -> str:
        return os.path.join(self._objects_dir, digest[:_PREFIX_LEN], digest[_PREFIX_LEN:])

    def _record_path(self, digest: str) -> str:
        return self._object_path(digest) + ".json"

    def _children_index_dir(self, digest: str) -> str:
        return os.path.join(self._children_dir, digest[:_PREFIX_LEN], digest[_PREFIX_LEN:])

    def _read_record(self, digest: str) -> ArtifactRecord:
        digest = _validate_digest(digest)
        record_path = self._record_path(digest)
        if not os.path.exists(record_path):
            raise ArtifactNotFoundError(digest)
        with open(record_path) as handle:
            payload = json.load(handle)
        return ArtifactRecord(
            digest=payload["digest"],
            metadata=payload["metadata"],
            parents=tuple(payload["parents"]),
            size=payload["size"],
            created=payload["created"],
        )

    def _write_record(self, record: ArtifactRecord) -> None:
        payload = {
            "digest": record.digest,
            "metadata": record.metadata,
            "parents": list(record.parents),
            "size": record.size,
            "created": record.created,
        }
        encoded = json.dumps(payload, indent=2, sort_keys=True, default=str).encode()
        _atomic_write(self._record_path(record.digest), encoded)

    def _link_child(self, *, parent: str, child: str) -> None:
        children_dir = self._children_index_dir(parent)
        os.makedirs(children_dir, exist_ok=True)
        marker_path = os.path.join(children_dir, child)
        if not os.path.exists(marker_path):
            with open(marker_path, "w"):
                pass


__all__ = [
    "ArtifactIntegrityError",
    "ArtifactNotFoundError",
    "ArtifactRecord",
    "ArtifactStore",
    "digest_of",
]
