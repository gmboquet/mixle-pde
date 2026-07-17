Field Modeling Capabilities
===========================

The field-modeling surface adds posterior modeling for gridded physical fields.
It complements forward solvers with typed field artifacts, observation
contracts, linear and nonlinear inversion paths, 4D assimilation, geoscience
likelihoods, posterior query utilities, calibration diagnostics, mesh helpers,
and readiness checks.

Package Boundary
----------------

``mixle-pde`` owns PDE/ODE solvers, simulator kernels, inverse-problem
utilities, mechanistic verification, and geophysical forward operators. Core
``mixle`` remains PDE-free and consumes these results through ordinary
artifacts, receipts, and model interfaces.

Capability Map
--------------

.. list-table::
   :header-rows: 1

   * - Area
     - Public surface
     - Purpose
   * - Fields
     - ``Field3D``, ``Field4D``, ``PosteriorField3D``
     - Grid or simplex-mesh geometry, time axes, units, bounds, posterior
       covariance, samples, intervals, and slices.
   * - Observations
     - ``Observation``, ``ForwardOperator``, ``ForwardOperatorRegistry``
     - Common measurement contract and registry-based prediction path.
   * - Static inversion
     - ``FieldGaussianPrior``, ``linear_gaussian_invert``,
       ``sparse_linear_gaussian_invert``
     - Exact dense or sparse linear-Gaussian posteriors for identity-transform
       fields.
   * - 4D priors
     - ``SpatioTemporalGaussianPrior``
     - Sparse spatial plus random-walk temporal precision over ``Field4D``.
   * - Bounded inversion
     - ``gauss_newton_invert``, ``GaussNewtonReport``
     - MAP plus Laplace covariance for bounded or nonlinear-transform fields.
   * - 4D assimilation
     - ``PosteriorField4D``, ``assimilate_4d``, ``assimilate_4d_ensemble``
     - Kalman/RTS smoothing for linear observations and ensemble filtering for
       nonlinear evolving fields.
   * - Geoscience likelihoods
     - assay, assemblage, biostratigraphy, geochronology, correlation, and
       facies helpers
     - Evidence surfaces for domain observations, not full geologic process
       simulators.
   * - Posterior queries
     - ``marginal_at_points``, ``section``, ``region_summary``,
       ``derived_quantity``
     - Query and compact posterior field artifacts.
   * - 3D/4D meshes
     - ``SimplexMesh``, ``MovingSimplexMesh``, ``moving_mesh``,
       ``pipe_radial_deformation``
     - Static and moving simplicial geometry for 3D domains and 4D
       space-time extrusion.
   * - Readiness
     - ``readiness_report``, ``run_required_modeling_checks``
     - Cheap verification scenarios for applications and CI.

Static Inversion
----------------

Use ``Field3D`` to name one physical property on a grid. Observations carry
geometry, values, noise, units, time, and provenance. A registry resolves each
observation kind to a forward operator.

``Field3D`` can also bind to a 3D simplex mesh through ``Field3D.from_mesh``.
``Field4D`` adds a strictly increasing time axis and optional moving-mesh
geometry so a posterior can describe a time-varying object rather than a loose
list of arrays.

.. code-block:: python

   import numpy as np

   from mixle_pde import (
       Field3D,
       FieldGaussianPrior,
       ForwardOperatorRegistry,
       Observation,
       borehole_forward_operator,
       linear_gaussian_invert,
   )

   coords = np.array(
       [
           [0.0, 0.0, 0.0],
           [1.0, 0.0, 0.0],
           [0.0, 1.0, 0.0],
       ]
   )
   grid = Field3D(coords, spacing=1.0, units="kg/m^3", property_name="density")

   obs = Observation(
       kind="borehole",
       location=np.array([[0.0, 0.0, 0.0]]),
       value=np.array([120.0]),
       noise_cov=np.array([25.0]),
       units="kg/m^3",
       provenance={"well": "A-01"},
   )

   registry = ForwardOperatorRegistry()
   registry.register(borehole_forward_operator())

   prior = FieldGaussianPrior(mean=0.0, smoothness_precision=1.0, marginal_precision=0.01)
   posterior = linear_gaussian_invert(grid, [obs], registry, prior)

``linear_gaussian_invert`` is exact when every observation operator is fixed
linear in the field and ``Field3D.bounds is None``. It rejects bounded fields
and nonlinear operators instead of silently linearizing them.

Bounded Fields
--------------

Physical properties often have bounds, such as positive susceptibility or
porosity in ``[0, 1]``. ``Field3D`` maps bounded values into an unconstrained
space with log or logit transforms. ``gauss_newton_invert`` solves for a MAP
estimate in that unconstrained space and returns a Laplace covariance there.

Every posterior sample and interval endpoint maps back through the field
transform, so bounded inversions should not report physically impossible
values.

4D Assimilation
---------------

``assimilate_4d`` adds time through a random-walk linear-Gaussian state model.
It uses the same ``Field3D``, ``Observation``, and ``ForwardOperatorRegistry``
contracts as static inversion. Unobserved times still receive posteriors from
the prior, process model, and neighboring observations.

``assimilate_4d_ensemble`` is the nonlinear reference path. It pushes ensemble
members through registered forward operators and returns Gaussian summaries
estimated from the ensemble at each time.

``PosteriorField4D`` exposes whole-time-axis arrays through ``mean_array``,
``marginal_std``, ``credible_interval()``, and ``sample()``. Samples are
per-time marginal draws because the stored artifact keeps one covariance per
time slice rather than a full cross-time covariance.

Moving Meshes
-------------

Use ``moving_mesh`` when a domain has fixed connectivity but time-varying node
coordinates, such as a deforming pipe, piston/cylinder volume, or evolving
geologic block model. The result can be interpolated at arbitrary times,
checked for volume changes and inverted cells, or extruded into a 4D
space-time simplex mesh.

``pipe_radial_deformation`` supplies a simple cylindrical displacement law for
radial and axial strain. It is geometry plumbing for moving-domain solvers; it
does not replace ALE transport, fluid-structure coupling, combustion chemistry,
or adaptive remeshing.

Posterior Queries and Calibration
---------------------------------

Use posterior query helpers after inversion or assimilation for point
marginals, sections, region summaries, linear derived quantities, low-rank
summaries, diagonal summaries, and ensemble exports.

Use calibration diagnostics before promotion:

* synthetic-truth coverage when available;
* held-out observation checks;
* uncertainty inflation away from sensitive observations;
* identifiability diagnostics for weakly constrained batches.

Validation Expectations
-----------------------

Field-modeling changes should include:

* known-answer likelihood or synthetic recovery tests;
* failure tests for unsupported exact-inversion inputs;
* posterior predictive or held-out fit evidence;
* clear documentation naming whether the posterior is exact, sparse-exact,
  linearized, ensemble-based, or sampled;
* strict Sphinx builds with warnings treated as errors.
