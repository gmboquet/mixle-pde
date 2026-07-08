API Overview
============

The generated API reference is available in :doc:`api/modules`. This page maps
the main modeling areas.

PDE Models And Solvers
----------------------

* ``mixle_pde.pde``, ``mixle_pde.pde_solve``, ``mixle_pde.dynamics``, and
  ``mixle_pde.ops`` provide the core PDE-state-space registration, solver
  infrastructure, dynamics operators, and operator helpers.

* ``mixle_pde.heat``, ``mixle_pde.wave``, ``mixle_pde.flow``,
  ``mixle_pde.elastic``, ``mixle_pde.maxwell``, and related modules provide
  forward solver families for thermal, wave, fluid, elastic, EM, acoustic, and
  multiphysics problems.

Field Modeling
--------------

``mixle_pde.latent``
    Static field and posterior-field containers.

``mixle_pde.observations`` and ``mixle_pde.geo_observations``
    Forward operators, Gaussian observation likelihoods, geochemical assays,
    biostrat constraints, and related geoscience observation models.

* ``mixle_pde.field_inversion``, ``mixle_pde.field_gauss_newton``,
  ``mixle_pde.field_assimilation``, and ``mixle_pde.field_priors`` provide
  linear-Gaussian inversion, Gauss-Newton inversion, 4D assimilation, depth
  weights, and cross-property priors.

Geoscience And Mechanistic Helpers
----------------------------------

* ``mixle_pde.geophysics``, ``mixle_pde.potential_fields``,
  ``mixle_pde.rock_physics``, ``mixle_pde.basin``, and
  ``mixle_pde.propagation_inverse`` provide potential-field, rock-physics,
  basin, and propagation inverse utilities.

``mixle_pde.posterior_query``, ``mixle_pde.posterior_calibration``, and ``mixle_pde.mesh``
    Posterior summaries, sections, region queries, low-rank compression,
    calibration diagnostics, and simplex mesh helpers.

Readiness
---------

``mixle_pde.capabilities``
    Modeling capability catalog, dependency checks, and verification scenarios.
