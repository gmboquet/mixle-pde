Solver Selection and Inversion Guide
====================================

``mixle-pde`` provides mechanistic forward solvers, inverse-problem utilities,
field priors, posterior summaries, and geoscience operators. This guide maps
common modeling goals to the package surface and the validation evidence that
should accompany them.

Forward Solver Choice
---------------------

Choose the solver family from the physics and numerical contract:

``heat`` and diffusion-like systems
    Use for parabolic smoothing, thermal diffusion, and simple scalar field
    propagation where stability depends on time step and grid spacing.

``wave``, ``wave3d``, and PML variants
    Use for acoustic or wave propagation examples. Record boundary conditions,
    source wavelet, receiver layout, and CFL or stability settings.

``flow`` and ``flow3d``
    Use for transport and fluid-style examples. Document conservation checks,
    boundary fluxes, and any numerical diffusion assumptions.

``gas_dynamics``
    Use for compressible shock-tube references and zero-dimensional reactive
    chamber studies. The combustion helper models fuel depletion, heat
    release, pressure rise, and prescribed piston/cylinder volume, not
    turbulent flame propagation or detailed chemistry.

``elastic`` and ``elastic_aniso``
    Use when vector displacement or anisotropic material behavior matters.
    Record material parameter bounds and symmetry assumptions.

``potential_fields`` and geophysical helpers
    Use for gravity, magnetic, or related potential-field examples where the
    forward operator connects subsurface parameters to observations.

``fem``, ``mesh``, ``plate``, and ``beam``
    Use for mesh-based or structural examples. Record mesh quality,
    constraints, and manufactured-solution checks where possible.

Inverse Workflow
----------------

An inverse problem should be documented as a dataflow:

1. Define the latent field or parameter vector.
2. Define the forward operator.
3. Attach observations with units and uncertainty.
4. Choose priors or regularization.
5. Run inversion or assimilation.
6. Query posterior summaries or calibrated predictions.
7. Validate against synthetic truth, held-out observations, or physical
   invariants.

The docs and examples should name which of those steps are real and which are
mocked or synthetic.

Uncertainty and Calibration
---------------------------

PDE workflows often fail quietly when uncertainty is treated as decoration.
A convincing example states:

* observation noise model;
* prior family and parameter bounds;
* posterior query target;
* calibration or coverage diagnostic;
* held-out observation behavior;
* failure mode for inconsistent observations.

Posterior summaries should state whether they are approximate, sampled,
linearized, ensemble-based, or analytic.

Numerical Stability Checks
--------------------------

A solver or inversion change should include at least one stability check:

* zero-state behavior;
* constant-state behavior;
* bounded-input and bounded-output behavior;
* manufactured solution or analytic comparison;
* convergence trend under grid refinement;
* conservation or monotonicity invariant where the physics requires it;
* graceful handling of singular, empty, or ill-conditioned inputs.

Optional Backends
-----------------

NumPy, SciPy, and Torch are base dependencies as of 0.7.0: every solver and
inverse callback runs through the Torch-backed ``ops`` namespace, so
importing any solver module requires Torch. GPU execution is a Torch device
choice, not a separate optional path. The genuinely optional pieces are the
``mixle_pde.env_data`` geophysical loaders (GEBCO, WOA/Argo, DEM, ERA5),
which import their heavier backends lazily and fail with a clear error
naming what is missing.
