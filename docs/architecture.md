# Architecture

The package has four roles: stable legacy facade, integrated install profile, curated/reference numerical kernels, and
canonical migration adapters. `ownership.py` exhaustively assigns every source module a disposition/final owner.
`canonical_adapter.py` implements the bounded portable Sim finite-system execution boundary and first P1 parity path.
`problem_adapter.py` negotiates broader Math problem features without selecting unsupported behavior.
`capabilities.py` retains the evidence-backed specialist readiness catalog.

Canonical dependency direction is Core values plus Physics theory → Sim lowering → Discrete planning/verification →
PDE specialist/native execution adapter → Sim reconstruction. PDE adapters operate on public records and never define
upstream identities.
