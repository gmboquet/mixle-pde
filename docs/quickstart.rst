Quickstart
==========

This quickstart exercises ``mixle-pde`` without requiring a large simulation.
It checks the modeling capability catalog and runs a small observation
likelihood through the shared field/observation contracts.

Install for Local Development
-----------------------------

From the repository root:

.. code-block:: console

   python -m pip install -e ".[docs]"
   python -m pytest tests/capabilities_test.py

Local tests and docs also need the core ``mixle`` package importable, either
from an installed distribution or from the sibling workspace checkout.

Inspect Modeling Capabilities
-----------------------------

.. code-block:: python

   from mixle_pde.capabilities import capability_catalog

   capabilities = capability_catalog()
   for capability in capabilities:
       print(capability.id, capability.available)

The catalog reports dependency availability and the deterministic readiness
scenarios associated with each modeling family. Use it to distinguish
implemented capability from future release plans.

Create an Observation Likelihood
--------------------------------

.. code-block:: python

   import numpy as np

   from mixle_pde.observations import Observation, gaussian_log_likelihood

   obs = Observation(
       kind="direct",
       location=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
       value=np.array([1.0, 1.5]),
       noise_cov=np.array([0.1, 0.1]),
       units="synthetic",
   )

   logp = gaussian_log_likelihood(obs, predicted=np.array([0.9, 1.4]))
   assert np.isfinite(logp)

This proves the typed observation contract and likelihood path without
constructing a full inversion.

Run a Focused Test Slice
-------------------------

For field-modeling changes, start with:

.. code-block:: console

   python -m pytest \
     tests/latent_test.py \
     tests/observations_test.py \
     tests/field_inversion_test.py \
     tests/posterior_query_test.py \
     tests/capabilities_test.py

Then add the solver-specific tests for the module you changed. See
:doc:`validation` for the full test matrix and :doc:`release-readiness` for
the evidence a release claim needs.

Next Steps
----------

Read :doc:`modeling-workflows` for the full field-to-posterior path and
:doc:`solver-selection-and-inversion-guide` for choosing solvers and inverse
methods.
