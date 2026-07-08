# Changelog

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
