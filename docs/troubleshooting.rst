Troubleshooting
===============

Core ``mixle`` Import Fails
---------------------------

Install core ``mixle`` into the environment or set ``PYTHONPATH`` to the local
core checkout for validation. Do not rely on an accidental shell path when
recording release evidence.

Torch Or SciPy Is Missing
-------------------------

Some solvers and inverse paths require optional numerical dependencies. The
base package should fail clearly for missing optional dependencies. Validation
notes should say which extras were installed.

Linear Inversion Rejects An Observation
---------------------------------------

Check whether the forward operator is nonlinear. Nonlinear observations belong
on the Gauss-Newton or dedicated inverse path, not exact linear-Gaussian
inversion.

Posterior Looks Overconfident
-----------------------------

Run held-out fit, posterior predictive checks, and uncertainty inflation
diagnostics. Sparse observations can produce attractive MAP fields with weak
identifiability.

Docs Build Fails
----------------

Run:

.. code-block:: console

   python -m sphinx -W -b html docs docs/_build/html
