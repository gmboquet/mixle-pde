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

Evidence Notes
--------------

For inverse and posterior workflows, keep the observation model, noise
assumptions, prior or regularization policy, grid/mesh shape, units, and
validation metric with the test record. Numerical evidence is much easier to
review when the modeling assumptions are recorded beside the command that
produced the result. See :doc:`release-readiness` for which evidence types a
modeling change needs before release.
