Troubleshooting
===============

Core ``mixle`` Import Fails
---------------------------

Install core ``mixle`` into the environment or set ``PYTHONPATH`` to the local
core checkout for the specific validation command. Confirm the interpreter sees
the intended package:

.. code-block:: console

   python - <<'PY'
   import mixle
   import mixle_pde

   print(mixle.__file__)
   print(mixle_pde.__file__)
   PY

Torch or SciPy Is Missing
-------------------------

NumPy, SciPy, and Torch are base dependencies, not optional extras: every
solver and inverse callback runs through the Torch-backed ``ops`` namespace
(``mixle_pde/ops.py`` imports torch unconditionally). If either is missing,
the environment install is incomplete. Reinstall with
``python -m pip install -e .`` rather than adding the package by hand. The
geophysical data loaders in ``mixle_pde.env_data`` (GEBCO, WOA/Argo, DEM,
ERA5) are the only genuinely optional pieces; they raise a clear
``ImportError`` naming the missing backend when their heavier dependencies
are absent.

Linear Inversion Rejects an Observation
---------------------------------------

Check whether the forward operator is nonlinear. Nonlinear observations belong
on the Gauss-Newton or dedicated inverse path, not exact linear-Gaussian
inversion. Also check that the observation covariance is positive definite and
that observed values match the operator's expected shape.

Posterior Looks Overconfident
-----------------------------

Run held-out fit, posterior predictive checks, and uncertainty inflation
diagnostics. Sparse observations can produce attractive MAP fields with weak
identifiability. Compare MAP, posterior mean, and predictive residuals before
treating a smooth field as evidence that the inverse problem is well
constrained.

MCMC Acceptance Collapses
-------------------------

For ``field_mcmc`` references, check the proposal scale, transformed field
bounds, and prior precision. A near-zero acceptance rate usually means the
proposal is too wide or the posterior is sharply constrained; an acceptance
rate near one usually means the chain is moving too slowly to explore the
posterior.

Docs Build Fails
----------------

Run:

.. code-block:: console

   PYTHONPATH=../mixle \
     make -C docs html SPHINXOPTS="-W --keep-going"

If autodoc fails on an optional dependency, decide whether the dependency
belongs in the docs environment or whether the module should provide a clearer
import-time error for users without that backend.
