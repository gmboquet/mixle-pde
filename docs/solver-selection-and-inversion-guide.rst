Solver Selection And Inversion Guide
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

Uncertainty And Calibration
---------------------------

PDE workflows often fail quietly when uncertainty is treated as decoration.
For release-quality examples, record:

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

SciPy and sparse linear algebra are part of the normal package path. Torch,
GPU, or other accelerated paths should remain optional unless the package
metadata and validation matrix explicitly make them required. Importing the
package without optional accelerators should not fail.
