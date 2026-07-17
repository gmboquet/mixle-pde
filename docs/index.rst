mixle-pde
=========

``mixle-pde`` is the PDE, ODE, field-inversion, and mechanistic simulation
package for Mixle. It keeps physics kernels, inverse-problem utilities,
geophysical forward operators, field priors, assimilation routines, and
PDE-constrained latent models outside the core package while preserving a
clean integration point with Mixle's probabilistic-programming surface.

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

Treat solver examples as building blocks: production workflows should record
the operator, boundary conditions, mesh or grid assumptions, noise model, and
validation evidence that make an inverse result defensible. For release
review, the API reference should cover every tracked module because missing
automodule pages hide public numerical surface area from reviewers.

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

Release Review Standard
-----------------------

Review PDE documentation by the numerical claim being made. A forward solver
needs a stability, manufactured-solution, or reference-problem check. An
inverse workflow needs recovery, posterior, or held-out evidence. A geoscience
example needs data classification, units, coordinate assumptions, and
limitations. The API reference names the surface; the workflow pages explain
what evidence makes results defensible.
