# Changelog

## 0.8.0 (development)

- Added exhaustive legacy-module disposition and final-owner inventory.
- Added a versioned native canonical backend manifest constrained to executable evidence.
- Added portable `mixle.sim.finite-linear-system/v1` validation, native execution, and residual receipts.
- Added a receipt-bearing P1 Poisson compatibility wrapper with legacy numerical parity.
- Preserved existing solver APIs while documenting cross-project semantic ownership and migration gates.

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
