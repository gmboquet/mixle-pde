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

Typical Workflow Shape
----------------------

A reviewable modeling workflow should name each stage explicitly:

* construct a grid, mesh, or low-dimensional field representation;
* choose a forward operator and boundary/source assumptions;
* attach observations with a stated noise model;
* choose a prior or regularization policy;
* run a forward, inverse, or assimilation routine;
* summarize the posterior or fitted field; and
* record validation evidence and known limitations.

The package provides small numerical kernels and contract objects for those
steps, but it does not make a scientific conclusion by itself. A downstream
demo or notebook must still state the data source, units, calibration
standard, and interpretation limits.

Numerical Expectations
----------------------

Solver examples should prefer explicit tolerances, deterministic seeds where
sampling is involved, and documented grid or mesh assumptions. Inverse
workflows should report misfit, posterior uncertainty, and any regularization
or prior that materially affects the result.
