Release Notes
=============

``mixle-pde`` is the mechanistic modeling and inverse-problem package for the
Mixle 0.7.0 family release. This page summarizes what ships in this release,
the validation evidence behind it, and its known risks.

Included
--------

* Sphinx manual with modeling workflows, package map, API overview, validation,
  and troubleshooting pages.
* 3D/4D field-modeling guide.
* Moving-domain simplex meshes with 3D pipe/cylinder deformation and 4D
  space-time extrusion.
* Mesh-backed ``Field3D`` and time-indexed ``Field4D`` latent object contracts.
* Fossil/palynology assemblage likelihoods with detection probability,
  reworking/background mixture, and overdispersion support.
* Forward-operator capability reports that distinguish fixed Jacobians,
  finite-difference fallbacks, differentiable paths, and true adjoint support.
* Sparse spatiotemporal Gaussian prior over ``Field4D`` objects.
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
* optional-loader behavior for the ``mixle_pde.env_data`` geophysical extras;
* ``make -C docs html SPHINXOPTS="-W --keep-going"``.

Known Risks
-----------

* Linear-Gaussian inversion and nonlinear Gauss-Newton paths must not be
  presented as interchangeable.
* Ensemble 4D assimilation is a reference Gaussian-summary path, not a full
  production particle/MCMC smoother.
* Geoscience likelihood helpers are evidence surfaces, not complete geologic
  process simulators.
* Moving meshes provide geometry support only. ALE transport, fluid-structure
  interaction, and adaptive remeshing remain outside this helper.
* The combustion kernel is a chamber model, not turbulent CFD, detonation, or
  detailed chemical kinetics.
