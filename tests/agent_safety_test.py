"""Tests for the MP-J6 agent safety / authority-boundary guards (mixle_pde.verification.agent_safety).

Every adversarial case in this file constructs a genuine attack shape (a real ``../`` traversal, a real
on-disk symlink escaping its root, a real oversized byte string, a real un-JSON-serializable callable, a
real URL-shaped string, a real over-policy ResourceBudget) and checks the guard actually stops it -- not
merely that the function exists. ``validate_against_allowlist``/``assert_no_network_hosts`` are checked
directly against :data:`mixle_pde.tools.PHYSICS_TOOL_SCHEMAS`, the real, frozen IC-3 tool contract, not a
synthetic schema invented for this test file.
"""

from __future__ import annotations

import os

import pytest

from mixle_pde import tools
from mixle_pde.artifact_store import ArtifactNotFoundError, ArtifactStore
from mixle_pde.job_governance import JobRegistry, JobStatus, ResourceBudget
from mixle_pde.verification.agent_safety import (
    ApprovalPolicy,
    ApprovalRequiredError,
    ArtifactTooLargeError,
    NetworkPolicyViolationError,
    PathEscapeError,
    PromotedArtifactImmutableError,
    PromotionRegistry,
    UnsafePayloadError,
    assert_json_safe_payload,
    assert_no_network_hosts,
    bounded_put,
    redact_secrets,
    resolve_within_workspace,
    submit_with_approval_gate,
    validate_against_allowlist,
)


# ---------------------------------------------------------------------------
# resolve_within_workspace -- path escape
# ---------------------------------------------------------------------------
def test_resolves_a_plain_relative_path_inside_the_workspace(tmp_path):
    (tmp_path / "sub").mkdir()
    resolved = resolve_within_workspace(tmp_path, "sub/file.json")
    assert resolved == os.path.realpath(tmp_path / "sub" / "file.json")


def test_rejects_dot_dot_traversal_outside_the_workspace(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with pytest.raises(PathEscapeError):
        resolve_within_workspace(workspace, "../../etc/passwd")


def test_rejects_an_absolute_path_outside_the_workspace(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside" / "secret.txt"
    with pytest.raises(PathEscapeError):
        resolve_within_workspace(workspace, str(outside))


def test_accepts_an_absolute_path_that_is_genuinely_inside_the_workspace(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    inside = workspace / "data" / "a.json"
    resolved = resolve_within_workspace(workspace, str(inside))
    assert resolved == os.path.realpath(inside)


def test_rejects_a_symlink_that_points_outside_the_workspace(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside_target = tmp_path / "outside.txt"
    outside_target.write_text("secret")
    symlink = workspace / "innocuous.json"
    symlink.symlink_to(outside_target)
    with pytest.raises(PathEscapeError):
        resolve_within_workspace(workspace, "innocuous.json")


# ---------------------------------------------------------------------------
# validate_against_allowlist -- checked against the real, frozen IC-3 schema
# ---------------------------------------------------------------------------
def test_allowlist_accepts_a_compliant_run_inversion_payload():
    schema = tools.PHYSICS_TOOL_SCHEMAS["run_inversion"]
    payload = {"dataset_ref": "d.json", "modality": "gravity", "prior": "smooth"}
    assert validate_against_allowlist(payload, schema) == ()


def test_allowlist_flags_an_undeclared_extra_key():
    schema = tools.PHYSICS_TOOL_SCHEMAS["run_inversion"]
    payload = {"dataset_ref": "d.json", "modality": "gravity", "prior": "smooth", "exec": "rm -rf /"}
    violations = validate_against_allowlist(payload, schema)
    assert any("exec" in v for v in violations)


def test_allowlist_flags_a_missing_required_key():
    schema = tools.PHYSICS_TOOL_SCHEMAS["run_inversion"]
    payload = {"dataset_ref": "d.json", "modality": "gravity"}  # missing required "prior"
    violations = validate_against_allowlist(payload, schema)
    assert any("prior" in v for v in violations)


@pytest.mark.parametrize("tool_name", list(tools.PHYSICS_TOOL_SCHEMAS))
def test_every_real_tool_schema_has_a_properties_and_required_shape_this_guard_understands(tool_name):
    schema = tools.PHYSICS_TOOL_SCHEMAS[tool_name]
    assert "properties" in schema
    assert isinstance(schema["required"], list)


# ---------------------------------------------------------------------------
# bounded_put -- artifact size limit, composed with the real MP-K1 ArtifactStore
# ---------------------------------------------------------------------------
def test_bounded_put_stores_content_under_the_limit(tmp_path):
    store = ArtifactStore(tmp_path / "objects")
    digest = bounded_put(store, b"small content", max_bytes=1024)
    assert store.exists(digest)
    assert store.get(digest) == b"small content"


def test_bounded_put_rejects_oversized_content_before_writing_anything(tmp_path):
    store = ArtifactStore(tmp_path / "objects")
    content = b"x" * 2048
    with pytest.raises(ArtifactTooLargeError):
        bounded_put(store, content, max_bytes=1024)
    import hashlib

    would_be_digest = hashlib.sha256(content).hexdigest()
    assert not store.exists(would_be_digest)


# ---------------------------------------------------------------------------
# assert_json_safe_payload -- structural "cannot execute code" guarantee
# ---------------------------------------------------------------------------
def test_json_safe_payload_accepts_plain_data():
    assert_json_safe_payload({"a": 1, "b": [1, 2.5, "x", None, True]})


def test_json_safe_payload_rejects_a_smuggled_callable():
    with pytest.raises(UnsafePayloadError):
        assert_json_safe_payload({"config": {"hook": lambda: None}})


def test_json_safe_payload_rejects_a_smuggled_class_instance():
    class NotJson:
        pass

    with pytest.raises(UnsafePayloadError):
        assert_json_safe_payload({"value": NotJson()})


# ---------------------------------------------------------------------------
# assert_no_network_hosts -- no real tool payload legitimately needs a URL
# ---------------------------------------------------------------------------
def test_network_policy_accepts_plain_data():
    assert_no_network_hosts({"dataset_ref": "d.json", "modality": "gravity", "prior": "smooth"})


@pytest.mark.parametrize(
    "payload",
    [
        {"dataset_ref": "http://evil.example.com/steal"},
        {"config": {"webhook": "https://attacker.example/exfiltrate"}},
        {"nested": ["fine", {"deep": "ftp://internal.example/creds"}]},
    ],
)
def test_network_policy_rejects_url_shaped_strings_anywhere_in_the_payload(payload):
    with pytest.raises(NetworkPolicyViolationError):
        assert_no_network_hosts(payload)


@pytest.mark.parametrize("tool_name", list(tools.PHYSICS_TOOL_SCHEMAS))
def test_no_real_tool_schema_declares_a_url_shaped_property(tool_name):
    """Grounds the module docstring's claim: no property in the real, frozen IC-3 schema is documented as
    a URL, so the default-deny network policy costs no legitimate caller of this repo's own tools
    anything."""
    schema = tools.PHYSICS_TOOL_SCHEMAS[tool_name]
    for prop_name, prop_schema in schema["properties"].items():
        description = str(prop_schema.get("description", ""))
        assert "url" not in prop_name.lower()
        assert "://" not in description


# ---------------------------------------------------------------------------
# redact_secrets
# ---------------------------------------------------------------------------
def test_redact_secrets_redacts_by_key_name_recursively():
    payload = {
        "api_key": "sk-live-abc123",
        "nested": {"password": "hunter2", "safe": "value"},
        "list": [{"token": "abc"}, {"safe": "ok"}],
    }
    redacted = redact_secrets(payload)
    assert redacted["api_key"] == "<redacted>"
    assert redacted["nested"]["password"] == "<redacted>"
    assert redacted["nested"]["safe"] == "value"
    assert redacted["list"][0]["token"] == "<redacted>"
    assert redacted["list"][1]["safe"] == "ok"


def test_redact_secrets_is_honest_about_only_judging_key_names_not_values():
    """Documented, narrow limitation: a secret-shaped *value* under an innocuous key name is not caught by
    this baseline -- confirms the module docstring's own honesty note rather than silently overclaiming."""
    payload = {"comment": "sk-live-abc123-this-looks-like-a-real-key"}
    redacted = redact_secrets(payload)
    assert redacted["comment"] == "sk-live-abc123-this-looks-like-a-real-key"


# ---------------------------------------------------------------------------
# ApprovalPolicy / submit_with_approval_gate -- composes with the real MP-L4 JobRegistry
# ---------------------------------------------------------------------------
def test_submit_within_policy_needs_no_approval():
    registry = JobRegistry()
    policy = ApprovalPolicy(max_gpu_count=0, max_cpu_cores=8.0, max_memory_mb=16_384.0, max_wall_clock_seconds=3600.0)
    budget = ResourceBudget(cpu_cores=2.0, memory_mb=1024.0, wall_clock_seconds=60.0, gpu_count=0)
    record = submit_with_approval_gate(registry, budget=budget, policy=policy)
    assert record.status is JobStatus.QUEUED
    assert registry.inspect(record.id).budget == budget


def test_submit_over_policy_without_approval_raises():
    registry = JobRegistry()
    policy = ApprovalPolicy(max_gpu_count=0)
    budget = ResourceBudget(gpu_count=4)
    with pytest.raises(ApprovalRequiredError) as excinfo:
        submit_with_approval_gate(registry, budget=budget, policy=policy)
    assert "gpu_count" in str(excinfo.value)
    assert registry.list_jobs() == []  # nothing was submitted


def test_submit_over_policy_with_explicit_approval_succeeds():
    registry = JobRegistry()
    policy = ApprovalPolicy(max_gpu_count=0)
    budget = ResourceBudget(gpu_count=4)
    record = submit_with_approval_gate(registry, budget=budget, policy=policy, approved=True)
    assert record.status is JobStatus.QUEUED
    assert registry.inspect(record.id).budget.gpu_count == 4


# ---------------------------------------------------------------------------
# PromotionRegistry -- composes with the real MP-K1 ArtifactStore
# ---------------------------------------------------------------------------
def test_promote_then_resolve_round_trips():
    registry = PromotionRegistry()
    registry.promote("production/model-v1", "a" * 64)
    assert registry.resolve("production/model-v1") == "a" * 64
    assert registry.is_promoted("production/model-v1")


def test_repromoting_the_identical_digest_is_a_no_op():
    registry = PromotionRegistry()
    registry.promote("production/model-v1", "a" * 64)
    registry.promote("production/model-v1", "a" * 64)  # same digest again -- never an error
    assert registry.resolve("production/model-v1") == "a" * 64


def test_repromoting_a_different_digest_without_allow_repromotion_is_rejected():
    registry = PromotionRegistry()
    registry.promote("production/model-v1", "a" * 64)
    with pytest.raises(PromotedArtifactImmutableError):
        registry.promote("production/model-v1", "b" * 64)
    assert registry.resolve("production/model-v1") == "a" * 64  # unchanged


def test_repromoting_with_explicit_allow_repromotion_succeeds():
    registry = PromotionRegistry()
    registry.promote("production/model-v1", "a" * 64)
    registry.promote("production/model-v1", "b" * 64, allow_repromotion=True)
    assert registry.resolve("production/model-v1") == "b" * 64


def test_resolve_of_an_unpromoted_name_raises_key_error():
    registry = PromotionRegistry()
    with pytest.raises(KeyError):
        registry.resolve("never-promoted")


def test_promote_checks_the_digest_actually_exists_in_the_given_store(tmp_path):
    store = ArtifactStore(tmp_path / "objects")
    registry = PromotionRegistry()
    real_digest = store.put(b"model bytes")
    registry.promote("production/model-v1", real_digest, store=store)
    assert registry.resolve("production/model-v1") == real_digest

    with pytest.raises(ArtifactNotFoundError):
        registry.promote("production/model-v2", "f" * 64, store=store)
