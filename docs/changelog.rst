Changelog
=========

This changelog records documentation-visible release changes for
``mixle-pde``.

0.7.0 - 2026-07-10
------------------

Version bumped to 0.7.0 to track the mixle 0.7.0 family release. Verified that
the PDE/ODE solver and inversion surface imports and runs against
``mixle==0.7.0`` (the plugin depends on ``mixle.ppl`` and ``mixle.inference``);
no runtime API changes were required.

See :doc:`release-notes` for scope, validation evidence, and known risks.

Development status: new PDE feature development is paused. Documentation
updates should clarify the current surface, validation expectations, and known
limits rather than introducing new runtime claims.

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

Release Gate
~~~~~~~~~~~~

A public release is not complete until solver/inverse tests, synthetic
recovery checks, optional-dependency behavior, strict Sphinx docs, packaging
checks, and the coordinated family manifest all refer to the same commit.

Reviewer Notes
~~~~~~~~~~~~~~

Documentation should preserve the difference between a small solver example, a
field-inversion workflow, and a scientific claim. When a page introduces a new
physics module, it should name the governing assumptions, input units,
boundary or mesh requirements, validation evidence, and limitations needed to
interpret the result.

Maintenance Notes
~~~~~~~~~~~~~~~~~

While development is paused, prefer documentation corrections, API reference
coverage, and validation instructions over new examples. A new example should
only land when it exercises already-reviewed code and includes the corresponding
numerical evidence.
