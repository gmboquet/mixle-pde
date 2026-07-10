Observation and Inversion Contract
==================================

``mixle-pde`` connects physical fields to observations through one explicit
contract: an ``Observation`` declares geometry, value, noise, time, units, and
provenance; a ``ForwardOperator`` maps a latent field to predicted
observations; a ``ForwardOperatorRegistry`` resolves observation kind to
operator. Inversion and assimilation code use that registry instead of
branching on every measurement type.

Define a Field
--------------

.. code-block:: python

   import numpy as np
   from mixle_pde.latent import Field3D

   coordinates = np.array(
       [
           [0.0, 0.0, -10.0],
           [1.0, 0.0, -10.0],
           [0.0, 1.0, -10.0],
           [1.0, 1.0, -10.0],
       ]
   )

   grid = Field3D(
       coordinates=coordinates,
       spacing=1.0,
       units="kg/m^3",
       property_name="density_contrast",
       bounds=None,
   )

The exact linear-Gaussian inversion route requires ``bounds=None`` because the
posterior is Gaussian in the field's own coordinate system. Bounded fields use
an unconstrained transform and should go through a nonlinear route such as the
Gauss-Newton path.

Register Observations
---------------------

.. code-block:: python

   from mixle_pde.observations import (
       ForwardOperatorRegistry,
       Observation,
       borehole_forward_operator,
   )

   registry = ForwardOperatorRegistry()
   registry.register(borehole_forward_operator())

   observed = Observation(
       kind="borehole",
       location=coordinates[[0, 3]],
       value=np.array([10.0, 14.0]),
       noise_cov=np.array([0.25, 0.25]),
       units="kg/m^3",
       provenance={"survey": "synthetic-doc-example"},
   )

``noise_cov`` may be a diagonal variance vector with shape ``(n,)`` or a full
``(n, n)`` covariance matrix. The observation validates location shape, value
shape, positive diagonal variances, and full-covariance symmetry.

``registry.capability_report()`` records whether each operator exposes a fixed
Jacobian, state-dependent local Jacobian, finite-difference fallback,
differentiable path, or true adjoint. This metadata is part of the inversion
contract; examples should not imply an adjoint exists when a finite-difference
fallback is being used.

Score a Field
-------------

.. code-block:: python

   values = np.array([10.0, 11.0, 13.0, 14.0])
   log_likelihood = registry.log_likelihood(grid, values, observed)
   assert np.isfinite(log_likelihood)

Every observation kind shares the same Gaussian likelihood calculation. The
operator supplies predictions; the observation supplies the noise model.

Invert a Static Field
---------------------

.. code-block:: python

   from mixle_pde.field_inversion import FieldGaussianPrior, linear_gaussian_invert

   prior = FieldGaussianPrior(
       mean=0.0,
       smoothness_precision=0.1,
       marginal_precision=0.01,
       length_scale=1.0,
       neighbors=2,
   )

   posterior = linear_gaussian_invert(grid, [observed], registry, prior)

   assert posterior.mean.shape == (grid.n,)
   assert posterior.cov.shape == (grid.n, grid.n)
   assert np.allclose(posterior.map, posterior.mean)

``linear_gaussian_invert`` checks two important conditions:

* every observation kind has a fixed Jacobian and is linear in the field;
* the field uses the identity transform.

If either condition is false, the function raises a clear error rather than
silently returning a linearized posterior.

Sparse Inversion
----------------

Use ``sparse_linear_gaussian_invert`` for larger grids where materializing the
dense covariance is not acceptable. It assembles the same posterior precision
with SciPy sparse matrices, solves for the posterior mean, and stores sparse
precision factors. Marginal variances and linear derived quantities are
recovered through sparse solves.

Posterior Predictive Checks
---------------------------

After fitting, evaluate held-out observations:

.. code-block:: python

   from mixle_pde.field_inversion import posterior_predictive_check

   held_out = Observation(
       kind="borehole",
       location=coordinates[[1, 2]],
       value=np.array([11.0, 13.0]),
       noise_cov=np.array([0.25, 0.25]),
   )

   check = posterior_predictive_check(posterior, registry, [held_out], alpha=0.1)
   assert 0.0 <= check.coverage <= 1.0

Release-quality examples should report held-out fit, not only posterior mean
or MAP fields.

Assimilate a 4D Field
---------------------

``assimilate_4d`` uses the same observation and registry contract for an
evolving field. ``observations_by_time`` is parallel to ``times``; an entry may
be empty, and the smoother still returns a posterior slice for that time.

.. code-block:: python

   from mixle_pde.field_assimilation import assimilate_4d

   times = np.array([0.0, 1.0, 2.0])
   observations_by_time = [
       [Observation("borehole", coordinates[[0]], np.array([10.0]), np.array([0.25]), time=0.0)],
       [],
       [Observation("borehole", coordinates[[3]], np.array([14.0]), np.array([0.25]), time=2.0)],
   ]

   posterior4d = assimilate_4d(
       grid,
       times,
       observations_by_time,
       registry,
       prior,
       process_var=1.0,
   )

   slice_t1 = posterior4d.at_time(1.0)
   assert slice_t1.mean.shape == (grid.n,)

Use ``assimilate_4d_ensemble`` when observations are nonlinear and fixed
Jacobians are not available. The ensemble route returns Gaussian summaries
from an ensemble approximation and should be described as approximate evidence.

Validation Checklist
--------------------

When adding an observation or inversion route, include:

* observation shape and noise-validation tests;
* a known-answer likelihood or synthetic recovery test;
* a failure test for unregistered observation kinds;
* a failure test for nonlinear operators passed to exact linear inversion;
* posterior predictive or held-out fit evidence;
* clear docs naming whether the posterior is exact, sparse-exact,
  linearized, ensemble-based, or sampled.
