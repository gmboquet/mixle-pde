---
id: GOV-AGENTS-PDE
schema_version: 1.0.0
document_version: 1.0.0
status: active
owner_project: PRJ-PDE
effective_at: 2026-07-15T00:00:00Z
reviewed_at: 2026-07-15T00:00:00Z
review_due: 2026-08-15T00:00:00Z
review_interval_days: 31
---

# Mixle PDE instructions

Mixle PDE is the installed integrated profile, legacy facade, migration layer, and curated specialist numerical-kernel
collection. Canonical values/inference belong to Core, physical meaning to Physics, numerical representation to Sim,
general mathematics/verification to Discrete, data fulfillment to Data, and reasoning to Inquiry.

Before material work, follow `/Users/grantboquet/mixle/status/AGENTS.md`, resolve the active release, and inspect
source/tests. Preserve these invariants:

- existing public imports and verified solver behavior remain compatible until an explicit deprecation gate passes;
- every module has a disposition and final semantic owner; compatibility does not create a second authority;
- adapters consume public versioned records and never import another package's private implementation;
- backend manifests advertise only executable features with named evidence and limitations;
- unsupported schema, discretization, mesh, evidence, or inverse behavior fails explicitly without degradation;
- execution receipts bind the exact source record, backend capability, conversions, residuals, and limitations;
- do not remove a legacy surface until callers, parity, migration docs, and the evidence window are complete.

See `docs/architecture.md`, `docs/contracts.md`, `docs/scientific-validity.md`, and `docs/releases.md`.
