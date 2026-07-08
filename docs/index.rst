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
:doc:`solver-selection-and-inversion-guide` when choosing a solver or inverse
route.

.. toctree::
   :caption: Start Here
   :hidden:
   :maxdepth: 2

   installation
   quickstart
   overview
   package-map
   modeling-workflows
   solver-selection-and-inversion-guide
   README
   release-0-6-3
   changelog
   security-and-data
   0.6.x-field-modeling
   api-overview
   validation
   troubleshooting

.. toctree::
   :caption: Reference
   :hidden:
   :maxdepth: 2

   api/modules
