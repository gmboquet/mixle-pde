Overview
========

``mixle-pde`` owns the mechanistic modeling layer of the Mixle ecosystem. It
provides forward solvers, inverse-problem helpers, geophysical operators,
field priors, posterior-query utilities, posterior-calibration diagnostics,
mesh helpers, and readiness checks for modeling capabilities.

Package Boundaries
------------------

The package owns:

* PDE and ODE solver kernels.
* PDE-constrained latent-state models.
* Forward geophysical operators and inversion helpers.
* Gaussian field priors, cross-property priors, and posterior checks.
* Gauss-Newton and linear-Gaussian inversion workflows.
* 4D field assimilation.
* Geoscience observation likelihoods, posterior-query utilities, and posterior
  calibration diagnostics.
* Mesh construction and modeling-readiness checks.

The package does not own gateway serving, typed knowledge contracts, notebooks,
or the core Mixle probability catalog. It plugs into those surfaces when a
workflow needs physics-aware modeling.

Integration With Mixle Core
---------------------------

Importing ``mixle_pde`` registers PDE-specific composition hooks with Mixle
without requiring the core package to depend on this repository. Local docs and
tests need the core package importable, either from an installed distribution
or from the sibling workspace path.
