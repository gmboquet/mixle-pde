Package Map
===========

``mixle_pde.dynamics`` / ``mixle_pde.ops`` / ``mixle_pde.pde`` / ``mixle_pde.pde_solve``
    PDE-state-space registration, backend-agnostic operator helpers, dynamics
    operators, sparse-solve tracking, and solver infrastructure.

Forward solver families
    ``heat``, ``wave``, ``wave3d``, ``wave_pml``, ``flow``, ``flow3d``,
    ``elastic``, ``elastic_aniso``, ``maxwell``, ``gas_dynamics``,
    ``schrodinger``, ``spectral_flow``, ``fem``, ``plate``, ``beam``, and
    related solver modules.

Field inversion and assimilation
    ``latent``, ``observations``, ``field_inversion``,
    ``field_gauss_newton``, ``field_mcmc``, ``field_assimilation``,
    ``field_priors``, and ``posterior_query``.

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
    scenarios, and simplex mesh utilities.
