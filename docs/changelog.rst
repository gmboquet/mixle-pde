Changelog
=========

This changelog records documentation-visible release changes for
``mixle-pde``.

Unreleased
----------

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
* Small-reference Random-Walk Metropolis inversion and empirical
  ``PosteriorFieldSamples3D`` artifacts for nonlinear/non-Gaussian posterior
  validation.
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

Release Gate
~~~~~~~~~~~~

A public release is not complete until solver/inverse tests, synthetic
recovery checks, optional-dependency behavior, strict Sphinx docs, packaging
checks, and the coordinated family manifest all refer to the same commit.
