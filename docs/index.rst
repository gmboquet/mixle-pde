mixle-pde
=========

``mixle-pde`` is Mixle's mechanistic-modeling package: PDE and ODE solvers,
field-inversion utilities, geophysical forward operators, field priors,
assimilation routines, and PDE-constrained latent models. Splitting it out
keeps physics kernels out of core Mixle, while giving downstream applications
one clean integration point into Mixle's probabilistic-programming surface.

Start Here
----------

Start with :doc:`quickstart` for a small capability and observation-likelihood
check. Use :doc:`modeling-workflows` for field-to-posterior data flow and
:doc:`observation-and-inversion-contract` for the concrete observation,
forward-operator, inversion, and assimilation surface. Use
:doc:`solver-selection-and-inversion-guide` when choosing a solver or inverse
route.

The package is intentionally split into two layers. The low-level solver
modules expose compact numerical kernels for heat, wave, flow, elasticity,
potential-field, electromagnetic, and geophysical propagation examples. The
field-inversion layer wraps those kernels in Mixle-style contracts for
observations, Gaussian priors, posterior summaries, MCMC reference checks,
Gauss-Newton updates, sampled-posterior updates, synthetic earth scenarios,
and assimilation reports.

Treat solver examples as building blocks, not finished workflows: a
defensible result records the operator, boundary conditions, mesh or grid
assumptions, noise model, and validation evidence behind it, not just the
code that produced a number.

.. toctree::
   :caption: Start Here
   :hidden:
   :maxdepth: 2

   installation
   quickstart
   overview
   package-map
   modeling-workflows
   observation-and-inversion-contract
   solver-selection-and-inversion-guide
   release-readiness
   release-notes
   changelog
   security-and-data
   field-modeling
   api-overview
   validation
   troubleshooting

.. toctree::
   :caption: Reference
   :hidden:
   :maxdepth: 2

   api/modules

See :doc:`release-readiness` for the release gates this package's numerical
claims are held to.
