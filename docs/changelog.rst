Changelog
=========

This changelog records documentation-visible release changes for
``mixle-pde``.

0.6.3 Release Branch
--------------------

See :doc:`release-0-6-3` for scope, validation evidence, and known risks.

Added
~~~~~

* Sphinx manual for PDE/ODE solvers, field modeling, observation and inversion
  contracts, solver selection, validation, and troubleshooting.
* 0.6.x field-modeling guide converted into Sphinx-native docs.
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
