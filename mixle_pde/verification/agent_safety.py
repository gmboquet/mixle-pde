"""Agent safety and authority-boundary guards for the bounded tool surface (MP-J6).

MP-J6's full description: "Enforce workspace/object-store scope, input allowlists, artifact size limits,
process isolation, network policy, secret redaction, commercial-license policy, and approval gates for
costly/external runs." Accept bar: "adversarial specs cannot execute code, escape paths, fetch arbitrary
URLs, exhaust resources beyond quota, or mutate promoted artifacts."

This module is a deliberately narrow baseline slice, matching this same MP-J family's established
convention (``knowledge_catalog.py``/MP-J2, ``diagnostic_ontology.py``/MP-J3, ``agent_loop.py``/MP-J4,
``knowledge_handoff.py``/MP-J5): a set of real, adversarially-tested guards composed directly with this
repo's own already-real tool/store/job primitives, not a fabricated sandbox. Mapped against the accept
bar's five named guarantees:

* **"escape paths"** -- :func:`resolve_within_workspace` resolves a caller-supplied path against an
  allowed root (via ``os.path.realpath``, which also collapses ``..`` components and follows symlinks to
  their real target) and raises :class:`PathEscapeError` for anything that lands outside it. Directly
  relevant to :mod:`mixle_pde.tools` (MP-J1): ``run_inversion``/``forward_model`` take caller-supplied
  ``dataset_ref``/``geometry_ref``/``config["artifact_path"]`` file paths with *no* scoping today (read
  directly from ``tools.py`` -- confirmed, not assumed) -- this guard is the reusable primitive a caller
  of those tools should wrap around every such path, though this module does not itself edit
  ``tools.py`` (see "Dependency direction" below).
* **"execute code"** -- :func:`assert_json_safe_payload` requires a payload to round-trip through
  ``json.dumps`` -- a live Python callable, class, or module instance always fails that round trip, so
  this is a structural (not merely policy) guarantee that a tool-call payload cannot smuggle a callable
  through the free-form ``params``/``config`` dict every IC-3 tool accepts.
* **"fetch arbitrary URLs"** -- :func:`assert_no_network_hosts` recursively rejects any URL-shaped string
  (``scheme://...``) anywhere in a payload. Grounded in a real, checked fact about this repo's own tool
  surface, not merely a generic policy: ``tests/agent_safety_test.py`` sweeps
  :data:`mixle_pde.tools.PHYSICS_TOOL_SCHEMAS` and confirms no declared property is documented as a URL,
  so rejecting one outright never breaks a legitimate call.
* **"exhaust resources beyond quota"** -- :class:`ApprovalPolicy`/:func:`submit_with_approval_gate` sit
  directly in front of :class:`mixle_pde.job_governance.JobRegistry` (MP-L4, already landed): a
  :class:`~mixle_pde.job_governance.ResourceBudget` exceeding policy thresholds raises
  :class:`ApprovalRequiredError` unless the caller explicitly passes ``approved=True`` -- reusing
  MP-L4's own real budget/registry, not a new resource model.
* **"mutate promoted artifacts"** -- :class:`PromotionRegistry` sits in front of
  :class:`mixle_pde.artifact_store.ArtifactStore` (MP-K1, already landed): once a name is promoted to a
  digest, repointing that name to a *different* digest raises :class:`PromotedArtifactImmutableError`
  unless the caller explicitly passes ``allow_repromotion=True``.

Two more guards round out the accept bar's supporting list (input allowlists, artifact size limits, secret
redaction): :func:`validate_against_allowlist` (checked directly against the real, frozen
``tools.PHYSICS_TOOL_SCHEMAS``, not a synthetic schema), :func:`bounded_put` (a size-checked wrapper
around :meth:`~mixle_pde.artifact_store.ArtifactStore.put`), and :func:`redact_secrets`.

Explicitly NOT attempted here (matching this family's "narrow, honest slice" convention rather than
fabricating coverage): **process isolation** -- this repo has no sandboxed executor anywhere;
``mixle_pde/job_governance.py``'s own docstring states it "never starts a thread, process, or MPI rank",
and no other module does either (confirmed by a repo-wide grep for ``subprocess``/``multiprocessing``
call sites reachable from the tool surface) -- a real guard here would need an actual sandbox to sit in
front of, which does not exist to compose with. **Commercial-license policy** -- no license/entitlement
catalog exists anywhere in this repo to enforce against (confirmed by a repo-wide grep for
"license"/"entitlement" outside this module and ``pyproject.toml``'s own MIT declaration) -- fabricating
one here would be inventing the very capability this task is supposed to guard, not guarding it.

Dependency direction: this module imports :mod:`mixle_pde.artifact_store` (MP-K1) and
:mod:`mixle_pde.job_governance` (MP-L4) at module level, read-only -- neither is modified here. It does
not import or modify :mod:`mixle_pde.tools` (MP-J1): ``tests/agent_safety_test.py`` imports
``tools.PHYSICS_TOOL_SCHEMAS`` read-only, purely to check :func:`validate_against_allowlist` and
:func:`assert_no_network_hosts` against a real, frozen schema rather than a synthetic one invented for
this module's own tests.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from mixle_pde.artifact_store import ArtifactNotFoundError, ArtifactStore
from mixle_pde.job_governance import JobRecord, JobRegistry, ResourceBudget

__all__ = [
    "PathEscapeError",
    "resolve_within_workspace",
    "validate_against_allowlist",
    "ArtifactTooLargeError",
    "bounded_put",
    "UnsafePayloadError",
    "assert_json_safe_payload",
    "NetworkPolicyViolationError",
    "assert_no_network_hosts",
    "redact_secrets",
    "ApprovalPolicy",
    "ApprovalRequiredError",
    "submit_with_approval_gate",
    "PromotedArtifactImmutableError",
    "PromotionRegistry",
]


# ---------------------------------------------------------------------------
# Workspace / object-store scope -- path-escape guard
# ---------------------------------------------------------------------------
class PathEscapeError(ValueError):
    """Raised when a caller-supplied path resolves outside its declared workspace root."""

    def __init__(self, path: str, root: str) -> None:
        self.path = path
        self.root = root
        super().__init__(f"path {path!r} resolves outside workspace root {root!r}")


def resolve_within_workspace(root: str | os.PathLike[str], path: str | os.PathLike[str]) -> str:
    """Resolve ``path`` against ``root`` and raise :class:`PathEscapeError` if it lands outside it.

    ``os.path.realpath`` both collapses ``..``/``.`` components and follows symlinks to their real
    target, so a relative ``../../etc/passwd`` traversal and a symlink planted inside the workspace that
    points outside it are both caught the same way -- neither is a special case. An absolute ``path``
    outside ``root`` is rejected identically to a relative one that escapes via traversal; a caller that
    wants to allow a specific absolute location must include it under ``root`` (e.g. a bind mount),
    never by special-casing this function.
    """
    root_real = os.path.realpath(root)
    candidate = path if os.path.isabs(path) else os.path.join(root_real, path)
    candidate_real = os.path.realpath(candidate)
    if candidate_real != root_real and not candidate_real.startswith(root_real + os.sep):
        raise PathEscapeError(str(path), root_real)
    return candidate_real


# ---------------------------------------------------------------------------
# Input allowlist -- checked against a real JSON-schema-shaped tool contract
# ---------------------------------------------------------------------------
def validate_against_allowlist(payload: Mapping[str, Any], schema: Mapping[str, Any]) -> tuple[str, ...]:
    """Check ``payload``'s top-level keys against a JSON-Schema-shaped ``schema`` (the same
    ``{"properties": {...}, "required": [...]}`` shape :data:`mixle_pde.tools.PHYSICS_TOOL_SCHEMAS`
    already uses for every real IC-3 tool).

    Returns a tuple of violation strings -- empty means compliant -- rather than raising immediately, so
    a caller can report every gap at once, mirroring
    :attr:`mixle_pde.problem_adapter.PDECompatibilityReport.unsupported_features`'s own "list every
    unsupported feature" convention instead of failing on the first one found.
    """
    properties = schema.get("properties", {})
    required = schema.get("required", ())
    violations = [
        f"key {key!r} is not declared in schema properties {sorted(properties)}"
        for key in payload
        if key not in properties
    ]
    violations += [f"required key {key!r} is missing" for key in required if key not in payload]
    return tuple(violations)


# ---------------------------------------------------------------------------
# Artifact size limits -- composes with the real MP-K1 ArtifactStore
# ---------------------------------------------------------------------------
class ArtifactTooLargeError(ValueError):
    def __init__(self, size: int, max_bytes: int) -> None:
        self.size = size
        self.max_bytes = max_bytes
        super().__init__(f"artifact content is {size} bytes, exceeds the {max_bytes}-byte limit")


def bounded_put(
    store: ArtifactStore,
    content: bytes,
    *,
    max_bytes: int,
    metadata: Mapping[str, Any] | None = None,
    parents: Sequence[str] = (),
) -> str:
    """:meth:`~mixle_pde.artifact_store.ArtifactStore.put`, but rejecting content over ``max_bytes``
    *before* it is ever written to disk."""
    if len(content) > max_bytes:
        raise ArtifactTooLargeError(len(content), max_bytes)
    return store.put(content, metadata=metadata, parents=parents)


# ---------------------------------------------------------------------------
# "Cannot execute code" -- a structural (JSON-round-trip), not merely policy, guarantee
# ---------------------------------------------------------------------------
class UnsafePayloadError(ValueError):
    pass


def assert_json_safe_payload(payload: Any) -> None:
    """Raise :class:`UnsafePayloadError` unless ``payload`` round-trips through ``json.dumps``.

    A live Python callable, class, module, or arbitrary object always fails that round trip (``TypeError:
    Object of type ... is not JSON serializable``), so requiring every tool-call ``params``/``config``
    payload to pass this check is a structural guarantee against smuggling executable code through a
    free-form dict, not merely a policy that could be argued around.
    """
    try:
        json.dumps(payload)
    except (TypeError, ValueError) as exc:
        raise UnsafePayloadError(
            f"payload is not JSON-serializable, so it cannot be ruled out as a smuggled live code object/callable: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Network policy -- no tool in this repo's real surface declares a URL-shaped parameter
# ---------------------------------------------------------------------------
_URL_SCHEME_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.\-]*://")


class NetworkPolicyViolationError(ValueError):
    pass


def _find_urls(value: Any, path: str = "$") -> Iterator[str]:
    if isinstance(value, str):
        if _URL_SCHEME_RE.match(value):
            yield f"{path}: {value!r}"
    elif isinstance(value, Mapping):
        for key, sub_value in value.items():
            yield from _find_urls(sub_value, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, sub_value in enumerate(value):
            yield from _find_urls(sub_value, f"{path}[{index}]")


def assert_no_network_hosts(payload: Any) -> None:
    """Recursively reject any URL-shaped string (``scheme://...``) anywhere in ``payload``.

    This repo's own tool surface has no legitimate use for one: every :data:`mixle_pde.tools.PHYSICS_TOOL_SCHEMAS`
    property is a content-hashed ref, an enum, or a numeric/object config knob, never a URL (checked
    directly by ``tests/agent_safety_test.py``, not assumed) -- so this default-deny policy costs no
    legitimate caller anything today.
    """
    violations = tuple(_find_urls(payload))
    if violations:
        raise NetworkPolicyViolationError(
            f"payload contains URL-shaped value(s); network access is denied by policy: {violations}"
        )


# ---------------------------------------------------------------------------
# Secret redaction
# ---------------------------------------------------------------------------
_SECRET_KEY_RE = re.compile(r"(api[_-]?key|secret|token|password|passwd|credential)", re.IGNORECASE)


def redact_secrets(payload: Any) -> Any:
    """Recursively replace the value of any mapping key whose *name* looks secret-shaped
    (``api_key``/``secret``/``token``/``password``/``passwd``/``credential``, case-insensitive) with a
    fixed ``"<redacted>"`` marker.

    Only a mapping *key* is ever judged -- a bare string value is never redacted on its own, since there
    is no key name to identify it by; that is a narrower, honestly-scoped guarantee than a general
    secret-*shaped-value* scanner (e.g. an AWS-key-shaped string under an innocuous key name would pass
    through), which this baseline does not attempt.
    """
    if isinstance(payload, Mapping):
        return {
            key: ("<redacted>" if _SECRET_KEY_RE.search(str(key)) else redact_secrets(value))
            for key, value in payload.items()
        }
    if isinstance(payload, list):
        return [redact_secrets(value) for value in payload]
    if isinstance(payload, tuple):
        return tuple(redact_secrets(value) for value in payload)
    return payload


# ---------------------------------------------------------------------------
# Approval gates for costly/external runs -- composes with the real MP-L4 JobRegistry
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ApprovalPolicy:
    """Resource ceilings a :class:`~mixle_pde.job_governance.ResourceBudget` may not exceed without
    explicit human approval."""

    max_wall_clock_seconds: float = 3600.0
    max_cpu_cores: float = 8.0
    max_memory_mb: float = 16_384.0
    max_gpu_count: int = 0

    def violations(self, budget: ResourceBudget) -> tuple[str, ...]:
        """Every policy dimension ``budget`` exceeds -- empty means the budget needs no approval."""
        checks = (
            ("wall_clock_seconds", budget.wall_clock_seconds, self.max_wall_clock_seconds),
            ("cpu_cores", budget.cpu_cores, self.max_cpu_cores),
            ("memory_mb", budget.memory_mb, self.max_memory_mb),
            ("gpu_count", budget.gpu_count, self.max_gpu_count),
        )
        return tuple(
            f"{name} {actual} exceeds policy ceiling {ceiling}" for name, actual, ceiling in checks if actual > ceiling
        )


class ApprovalRequiredError(RuntimeError):
    def __init__(self, reasons: tuple[str, ...]) -> None:
        self.reasons = reasons
        super().__init__(f"job requires explicit human approval before submission: {'; '.join(reasons)}")


def submit_with_approval_gate(
    registry: JobRegistry,
    *,
    budget: ResourceBudget,
    policy: ApprovalPolicy,
    approved: bool = False,
    retry_limit: int = 0,
    metadata: Mapping[str, Any] | None = None,
    job_id: str | None = None,
) -> JobRecord:
    """:meth:`~mixle_pde.job_governance.JobRegistry.submit`, gated by ``policy``: a ``budget`` that
    exceeds any policy ceiling raises :class:`ApprovalRequiredError` unless the caller passes
    ``approved=True`` -- the caller-facing signal that a human (not this function) decided the costly/
    external run is authorized. This function never grants approval itself; it only enforces that
    someone must have.
    """
    reasons = policy.violations(budget)
    if reasons and not approved:
        raise ApprovalRequiredError(reasons)
    return registry.submit(budget=budget, retry_limit=retry_limit, metadata=dict(metadata or {}), job_id=job_id)


# ---------------------------------------------------------------------------
# Promoted-artifact immutability -- composes with the real MP-K1 ArtifactStore
# ---------------------------------------------------------------------------
class PromotedArtifactImmutableError(ValueError):
    def __init__(self, name: str, existing_digest: str, attempted_digest: str) -> None:
        self.name = name
        self.existing_digest = existing_digest
        self.attempted_digest = attempted_digest
        super().__init__(
            f"promoted name {name!r} is already bound to {existing_digest}; cannot rebind it to "
            f"{attempted_digest} without allow_repromotion=True"
        )


class PromotionRegistry:
    """An in-process ``name -> content digest`` promotion map (single-owner/dict-backed, the same explicit
    non-goal :class:`~mixle_pde.job_governance.JobRegistry` states for real concurrent/distributed use).

    Once :meth:`promote` binds ``name`` to a digest, a second call with a *different* digest raises
    :class:`PromotedArtifactImmutableError` unless ``allow_repromotion=True`` -- the mechanical guarantee
    behind MP-J6's "cannot mutate promoted artifacts" accept-bar clause. Re-promoting the *same* digest is
    always a no-op, never an error (promotion is idempotent for identical content).
    """

    def __init__(self) -> None:
        self._promotions: dict[str, str] = {}

    def promote(
        self, name: str, digest: str, *, store: ArtifactStore | None = None, allow_repromotion: bool = False
    ) -> None:
        """Bind ``name`` to ``digest``. When ``store`` is supplied, ``digest`` must already exist in it
        (:class:`~mixle_pde.artifact_store.ArtifactNotFoundError` otherwise) -- a promoted name can never
        point at content nothing actually stored."""
        if store is not None and not store.exists(digest):
            raise ArtifactNotFoundError(digest)
        existing = self._promotions.get(name)
        if existing is not None and existing != digest and not allow_repromotion:
            raise PromotedArtifactImmutableError(name, existing, digest)
        self._promotions[name] = digest

    def resolve(self, name: str) -> str:
        try:
            return self._promotions[name]
        except KeyError:
            raise KeyError(f"no promoted artifact registered under name {name!r}") from None

    def is_promoted(self, name: str) -> bool:
        return name in self._promotions
