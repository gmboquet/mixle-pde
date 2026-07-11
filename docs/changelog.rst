Changelog
=========

This changelog records documentation-visible release changes for
``mixle-pde``. See the root ``CHANGELOG.md`` for the full commit-level record.

0.7.0 - 2026-07-11
-------------------

The package's first public-release-readiness pass. ``mixle>=0.7.0`` is now the
declared dependency floor (previously unpinned), and ``torch>=2.0`` is
declared as a core dependency rather than test-only: every solver and inverse
callback runs through the ``ops`` torch backend, so it was already required at
runtime. Verified that the PDE/ODE solver and inversion surface imports and
runs against ``mixle==0.7.0`` (the plugin depends on ``mixle.ppl`` and
``mixle.inference``); no runtime API changes were required.

This pass also found and fixed four deterministic test/scenario bugs surfaced
by pinning the dependency floor: a ``TypeError`` in the default
modeling-readiness checks, an unrealized noise covariance that made a
synthetic-inversion comparison a coin flip, an RTS-smoother uncertainty
assertion with the comparison direction backwards under amplifying dynamics,
and a prior constructed with an invalid (zero) marginal precision.

See :doc:`release-notes` for scope, validation evidence, and known risks.

Added
~~~~~

* Sphinx manual for PDE/ODE solvers, field modeling, observation and inversion
  contracts, solver selection, validation, and troubleshooting.
* field-modeling guide converted into Sphinx-native docs.
* Mesh-backed ``Field3D`` and time-indexed ``Field4D`` latent object contracts.
* Fossil/palynology assemblage likelihoods with detection and reworking
  uncertainty.
* Forward-operator capability reports for Jacobian, finite-difference,
  differentiable, and adjoint metadata.
* Sparse spatiotemporal Gaussian prior over ``Field4D`` objects.
* Moving-domain simplex mesh primitives for 3D deformation and 4D space-time
  geometry.
* Full-time-axis 4D posterior arrays, intervals, samples, and interpolated
  query slices.
* Zero-dimensional reactive gas/engine-cylinder combustion simulator kernel.
* Generated API reference for public PDE, geophysics, field-inversion, and
  assimilation modules.
* Release-readiness checklist for manufactured solutions, inverse recovery,
  posterior checks, numerical honesty, and optional-dependency evidence.

Changed
~~~~~~~

* Docs separate linear-Gaussian, nonlinear inversion, assimilation, and
  posterior-query surfaces so examples do not overstate solver guarantees.
* The docs tree is Sphinx/reStructuredText only.

Known limitations
~~~~~~~~~~~~~~~~~~

New PDE feature development is paused for this release.
