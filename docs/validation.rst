Validation
==========

The test suite covers forward solvers, inverse solvers, field modeling,
geoscience observation likelihoods, mesh helpers, posterior queries,
posterior calibration diagnostics, and modeling capability checks.

Focused field-modeling validation:

.. code-block:: console

   PYTHONPATH=../mixle python -m pytest \
       tests/latent_test.py \
       tests/observations_test.py \
       tests/field_inversion_test.py \
       tests/field_gauss_newton_test.py \
       tests/field_assimilation_test.py \
       tests/field_priors_test.py \
       tests/geo_observations_test.py \
       tests/posterior_query_test.py \
       tests/posterior_calibration_test.py \
       tests/mesh_test.py \
       tests/capabilities_test.py

The local workspace needs the core Mixle package on ``PYTHONPATH`` unless
``mixle`` is installed into the active environment.

For this docs pass, validation language should preserve existing behavior. Do
not add new solver claims without the matching test command and expected
evidence.

Run the full suite from the package root with:

.. code-block:: console

   python -m pytest

Strict Documentation Gate
-------------------------

.. code-block:: console

   make -C docs html SPHINXOPTS="-W --keep-going"

Clean-Archive Documentation Gate
--------------------------------

Before public release, also build the docs from tracked files only:

.. code-block:: console

   tmp=$(mktemp -d)
   git archive HEAD | tar -x -C "$tmp"
   PYTHONPATH="$tmp:${MIXLE_CORE_CHECKOUT:?set MIXLE_CORE_CHECKOUT to a core mixle checkout}" \
     make -C "$tmp/docs" html SPHINXOPTS="-W --keep-going"

Use an installed core ``mixle`` package instead of the workspace path when
validating published artifacts.

Modeling Review
---------------

A modeling change should include more than import coverage. Record which of
these were exercised:

* analytic or manufactured-solution check;
* synthetic inverse recovery;
* posterior predictive check;
* held-out observation fit;
* uncertainty calibration or coverage;
* stability check for zero, constant, or bounded states;
* optional dependency behavior for SciPy, Torch, or sparse backends.

Evidence Notes
--------------

For inverse and posterior workflows, keep the observation model, noise
assumptions, prior or regularization policy, grid/mesh shape, units, and
validation metric with the test record. Numerical evidence is much easier to
review when the modeling assumptions are recorded beside the command that
produced the result.

Paused-Development Review
-------------------------

When reviewing documentation during the pause, classify each change as
clarification, correction, API-reference coverage, or validation guidance. Any
change that sounds like a new capability should be backed by committed tests or
rewritten as a limitation.
