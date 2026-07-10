Package Map
===========

``mixle_pde.dynamics`` / ``mixle_pde.ops`` / ``mixle_pde.pde`` / ``mixle_pde.pde_solve``
    PDE-state-space registration, backend-agnostic operator helpers, dynamics
    operators, sparse-solve tracking, and solver infrastructure.

``mixle_pde._operator``
    Typed callback contracts for the ``ops`` namespace and user-supplied
    ``forward``, ``observe``, and right-hand-side functions. This is the
    documentation-level contract that keeps PDE callbacks consistent with the
    core probabilistic-programming surface.

Forward solver families
    ``heat``, ``wave``, ``wave3d``, ``wave_pml``, ``flow``, ``flow3d``,
    ``elastic``, ``elastic_aniso``, ``maxwell``, ``gas_dynamics``,
    ``schrodinger``, ``spectral_flow``, ``fem``, ``plate``, ``beam``, and
    related solver modules.

Field inversion and assimilation
    ``latent``, ``observations``, ``field_inversion``,
    ``field_gauss_newton``, ``field_mcmc``, ``field_assimilation``,
    ``field_priors``, ``posterior_query``, ``posterior_calibration``,
    ``sample_update``, and ``earth_scenarios``. Use exact linear-Gaussian or
    sparse-precision routes when their assumptions hold; use Gauss-Newton,
    ensemble, variational, MCMC references, or sampled-posterior importance
    updates when the observation operator or likelihood requires it.

Geoscience and potential fields
    ``geophysics``, ``geo_observations``, ``potential_fields``,
    ``rock_physics``, ``basin``, ``migration``, ``propagation_inverse``, and
    ``reasoning``.

Propagation and environment helpers
    ``attenuation``, ``boundaries``, ``env_data``, ``parabolic_equation``,
    ``ray_scattering``, ``normal_modes``, ``refractivity``, and
    ``sound_speed``.

Capability and mesh support
    ``capabilities`` and ``mesh`` provide readiness checks, verification
    scenarios, and simplex mesh utilities. The capability checks are intended
    for release review and demo gating, not as a replacement for problem-level
    numerical validation.

Documentation Boundaries
------------------------

This package map is navigational, not a claim that every solver family has the
same maturity level. Solver pages and validation evidence should state the
assumptions, numerical method, supported dimensions, boundary handling, and
expected diagnostics for the specific workflow being used.

Adding Modules
--------------

When adding a solver or inversion helper, update the package map, API overview,
workflow guide, and validation expectations together. A new public module
should say whether it is a forward solver, inverse helper, posterior diagnostic,
mesh utility, geoscience operator, or capability check. That classification
keeps the API reference navigable as the package grows.
