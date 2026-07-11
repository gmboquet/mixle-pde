# Changelog

## 0.7.0 - 2026-07-11

`mixle-pde` is versioned 0.7.0 to track the mixle 0.7.0 family release. This is the
package's first public-release-readiness pass: build, dependency, test, lint, and
documentation gates were run and verified against the exact `release/0.7.0` commit
(see `release-checklists/0.7.0.md` for full evidence).

### Added

- `mixle>=0.7.0` declared as the package's dependency floor (previously unpinned).
- `torch>=2.0` declared as a core (not test-only) dependency: `mixle_pde/ops.py`
  imports torch unconditionally, so every solver and inverse callback already
  required it at runtime.

### Changed

- CI installs `mixle` from PyPI at the declared floor instead of `git+...@main`,
  now that `mixle` is published.
- README and CONTRIBUTING no longer claim that installing `mixle` alone brings
  torch/numba; `mixle-pde` now declares its own runtime dependencies directly.

### Fixed

- `mixle_pde.capabilities.run_required_modeling_checks()` (and
  `assert_required_modeling()`, `readiness_report()`) raised `TypeError` for the
  default required-capability set: one verification scenario's `tolerance` dict
  held a list value (`biostrat_age_range`) where every other scenario, and the
  `ScenarioResult` type, expect scalar floats. Split into
  `biostrat_age_min`/`biostrat_age_max`.
- `run_synthetic_3d_geochem_geophysics_inversion`'s gravity observation was
  tagged with a noise covariance but never actually had noise realized against
  it, leaving the geophysical-only posterior already near-exact and making the
  "assay update improves the estimate" comparison a coin flip dominated by
  resampling noise rather than signal. Noise is now drawn at the declared
  covariance.
- Two test/scenario assertions in the linear-dynamics 4D assimilation path
  compared smoothed uncertainty at the observed time against the smoothed
  uncertainty at the start of the window; with amplifying (doubling) dynamics
  an RTS smoother legitimately ends up *tighter* at the start than at the
  observation, so the comparison doesn't hold in general. Reworked to compare
  against the flat-prior uncertainty instead, which is what the tests actually
  intended to check.
- A `SpatioTemporalGaussianPrior` test constructed a `FieldGaussianPrior` with
  `marginal_precision=0.0` to isolate temporal-coupling terms; `0.0` fails the
  class's own "proper prior" validation (`marginal_precision > 0`, in place
  since the class was introduced). Uses a numerically negligible `1e-12`
  instead.
- Ruff import-order/format violations and one unused import across
  `mixle_pde/__init__.py`, `earth_scenarios.py`, `capabilities.py`, `fem.py`,
  `field_assimilation.py`, `gas_dynamics.py`, `geo_observations.py`,
  `latent.py`, `posterior_query.py`, `sample_update.py`, and
  `tests/sample_update_test.py` (the `lint` CI job was failing on this commit).
- Broken README link to a pre-Sphinx-migration doc path
  (`docs/0.6.x-field-modeling.md` -> `docs/field-modeling.rst`).

### Known limitations

- New PDE feature development is paused for this release; see
  `docs/release-readiness.rst` for the numerical-honesty and scope-freeze gates
  this pass followed.

## 0.6.3 PDE modeling branch - 2026-07-08

`mixle-pde` currently declares package version `0.1.0`; this entry tracks the
0.6.x PDE modeling capability branch participating in the Mixle 0.6.3 family
release work.

### Added

- Sphinx docs for installation, overview, package map, modeling workflows,
  field-modeling guide, API overview, validation, troubleshooting, release
  notes, and security/data handling.
- Generated API documentation for public `mixle_pde` modules.

### Changed

- Documentation now covers solver families, field inversion, field priors,
  posterior query/calibration, geophysical observations, and package
  boundaries with core Mixle.

### Fixed

- `docs/_build` is ignored for local builds.

### Removed

- No documented public API removal in this pass.

### Known limitations

- Public release still needs clean build/install evidence, solver-specific
  known-answer tests, optional-dependency behavior checks, and family
  integration verification against the final core package.
