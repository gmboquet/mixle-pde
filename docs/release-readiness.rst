Release Readiness
=================

``mixle-pde`` is release-ready only when solver behavior, inverse-problem
workflows, and field-modeling claims are backed by numerical checks. Import
coverage alone is not enough for a physics package.

Supported Environment
---------------------

The package metadata declares Python 3.10 and newer with core ``mixle``,
NumPy, and SciPy as base dependencies. Optional Torch, sparse, or accelerator
paths should remain guarded and should record their tested versions when used.

Because new development is paused, release readiness should focus on preserving
and accurately documenting the current surface rather than expanding scope.

Solver and Inversion Gates
--------------------------

For changed PDE, ODE, geophysics, or inversion behavior, record at least one
of the following evidence types:

* manufactured-solution or analytic check;
* synthetic inverse recovery;
* posterior predictive or held-out observation check;
* conservation, stability, or boundary-condition check;
* uncertainty calibration or coverage check; or
* comparison against a known small reference problem.

Numerical Honesty Gates
-----------------------

Docs should not present linear-Gaussian updates, nonlinear Gauss-Newton
iterations, ensemble summaries, and full posterior samplers as interchangeable.
Each workflow page should name the approximation, failure modes, and optional
dependencies used by the example.

Documentation Gates
-------------------

The solver selection guide, modeling workflow, observation/inversion contract,
field-modeling guide, and API reference should match the shipped modules.
Build Sphinx with warnings as errors and from a clean archive before release.

Blocking Conditions
-------------------

Block release when a solver has no numerical check, an inverse workflow lacks a
synthetic or reference recovery path, optional dependencies change results
without documentation, or generated artifacts omit units, coordinates,
provenance, or limitations. A physics package can import cleanly while still
being scientifically under-documented.

Scope Freeze Gate
-----------------

Do not add a new solver, inverse workflow, or geoscience claim during the docs
release pass unless the accompanying validation evidence already exists. If a
page references a capability that is only planned, label it as planned or remove
it from the public release surface.
