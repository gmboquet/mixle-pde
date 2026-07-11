Modeling Workflows
==================

``mixle-pde`` makes mechanistic models usable from probabilistic and
application workflows without moving physics kernels into core ``mixle``.
This page maps the workflow from field definition to posterior validation.

Define the Field
----------------

Start with the physical quantity and geometry:

* property name and units;
* grid or mesh;
* bounds or transform;
* prior smoothness assumptions;
* whether the field is static, time-indexed, or coupled to another property.

Use ``Field3D`` and ``PosteriorField3D`` for gridded field posteriors. Use the
mesh helpers when geometry needs to be explicit rather than implied by array
shape.

Choose Observations
-------------------

Use the observation registry when a measurement can be represented as a
forward operator plus Gaussian noise. Built-in operators cover direct samples,
potential fields, DC resistivity, magnetotellurics, and related geophysical
surfaces.

Keep nonlinear observations out of exact linear-Gaussian inversion. Use the
Gauss-Newton path or a dedicated inverse routine when the forward map is
nonlinear.

The concrete ``Observation`` and ``ForwardOperatorRegistry`` contract is shown
in :doc:`observation-and-inversion-contract`.

Choose the Inference Route
--------------------------

``linear_gaussian_invert``
    Exact posterior for linear observations and Gaussian priors.

``gauss_newton_invert``
    Bounded or transformed fields and local Laplace covariance.

``assimilate_4d``
    Linear time-indexed fields with Kalman/RTS smoothing.

``assimilate_4d_ensemble``
    Nonlinear ensemble reference path for evolving fields.

``Differential``
    Differentiable forward callbacks integrated with the PPL surface.

Validate the Posterior
----------------------

Do not stop at a MAP field. Record:

* held-out observation fit;
* posterior predictive checks;
* coverage against synthetic truth when available;
* uncertainty inflation away from data;
* identifiability flags when observations are too sparse;
* physical sanity checks such as conservation, sign, monotonicity, or stability.

Query and Export
----------------

Use posterior query helpers for:

* points and sections;
* region summaries and total mass;
* derived linear quantities;
* low-rank or diagonal summaries;
* ensemble samples.

Applications should export summaries with units, provenance, and limitations
rather than raw arrays alone.

Boundaries
----------

``mixle-pde`` owns physics kernels and inverse utilities. It does not own demo
fixtures, gateway serving, mobile UI state, or core probability primitives.
Those surfaces belong in ``mixle-demos``, ``mixle-mlops``, ``mixle-ios``, and
``mixle`` respectively.
