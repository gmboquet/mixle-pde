Validation
==========

The test suite covers forward solvers, inverse solvers, field modeling,
geoscience observation likelihoods, mesh helpers, posterior queries,
posterior calibration diagnostics, and modeling capability checks.

Focused field-modeling validation:

.. code-block:: console

   PYTHONPATH=/Users/grantboquet/mixle/mixle python -m pytest \
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
