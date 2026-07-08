Release Notes
=============

``mixle-pde`` is the mechanistic modeling and inverse-problem package for the
0.6.3 family. Its documentation now covers solver families, field modeling,
posterior workflows, validation expectations, and package boundaries.

Included
--------

* Sphinx manual with modeling workflows, package map, API overview, validation,
  and troubleshooting pages.
* 3D/4D field-modeling guide for the release branch.
* Moving-domain simplex meshes with 3D pipe/cylinder deformation and 4D
  space-time extrusion.
* 4D posterior artifacts with full-time mean, marginal uncertainty, credible
  intervals, marginal samples, and interpolated time slices.
* Zero-dimensional reactive gas/engine-cylinder combustion kernel with fuel
  depletion, heat release, pressure rise, and prescribed piston volume.
* Generated API pages for public modules.
* Documentation extra in package metadata.
* ``docs/_build`` ignore rule for local builds.

Validation Evidence
-------------------

Record:

* focused tests for touched solver/inverse modules;
* synthetic recovery or known-answer checks;
* posterior predictive or held-out checks for field modeling changes;
* optional dependency behavior for Torch, SciPy, and sparse paths;
* ``python -m sphinx -W -b html docs docs/_build/html``.

Known Risks
-----------

* Linear-Gaussian inversion and nonlinear Gauss-Newton paths must not be
  presented as interchangeable.
* Ensemble 4D assimilation is a reference Gaussian-summary path, not a full
  production particle/MCMC smoother.
* Geoscience likelihood helpers are evidence surfaces, not complete geologic
  process simulators.
* Moving meshes are geometry support; they do not yet provide ALE transport,
  fluid-structure interaction, or adaptive remeshing.
* The combustion kernel is a chamber model, not turbulent CFD, detonation, or
  detailed chemical kinetics.
