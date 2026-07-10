API Overview
============

The generated API reference is available in :doc:`api/modules`. This page maps
the main modeling areas.

PDE Models and Solvers
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
    biostrat constraints, and related geoscience observation models. See
    :doc:`observation-and-inversion-contract` for runnable usage.

* ``mixle_pde.field_inversion``, ``mixle_pde.field_gauss_newton``,
  ``mixle_pde.field_assimilation``, and ``mixle_pde.field_priors`` provide
  linear-Gaussian inversion, Gauss-Newton inversion, 4D assimilation, depth
  weights, and cross-property priors.

Geoscience and Mechanistic Helpers
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

Inverse-Problem Surface
-----------------------

``mixle_pde.inverse``, ``mixle_pde.field_inversion``, and ``mixle_pde.propagation_inverse``
    Inverse-problem entry points. Use these when a workflow needs to infer
    field parameters or source properties from observations rather than only
    simulate a forward system.

``mixle_pde.field_mcmc`` and ``mixle_pde.sample_update``
    Reference sampling and sampled-posterior update helpers. These modules are
    useful for validating approximate updates against slower but more explicit
    posterior checks.

``mixle_pde.misfit`` and ``mixle_pde.posterior_calibration``
    Misfit functions and calibration diagnostics. Review these modules when a
    result claims posterior fit, coverage, or calibrated uncertainty.

Solver Families
---------------

Specialized solver modules include ``flow3d``, ``wave3d``, ``wave_pml``,
``helmholtz_pml``, ``em_diffusion_3d``, ``poroelastic``, ``viscoelastic``,
``normal_modes``, ``guided_wave``, ``ray_scattering``, ``refractivity``,
``two_phase``, and ``smoluchowski``. They should be treated as focused
mechanistic examples unless a workflow records mesh, boundary, coefficient,
source, and stability assumptions.

Review Notes
------------

The API reference should be used with the workflow guides. A module page proves
that the public names are documented; the workflow pages explain which
combination of observation model, forward operator, prior, inversion method,
and validation evidence makes a result reviewable.
