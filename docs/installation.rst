Installation
============

``mixle-pde`` installs into an environment that can import core ``mixle``.
Install core first when validating the package family from sibling checkouts,
or install the published core package before testing an isolated release
candidate.

Use installation checks to preserve the existing package surface. Do not treat a
local environment with unpublished solver experiments as release evidence.

From a checkout:

.. code-block:: console

   python -m pip install -e .
   python -m pip install -e ".[test]"
   python -m pip install -e ".[docs]"

When working from the sibling workspace without installing core Mixle, put core
on ``PYTHONPATH`` for the command being validated:

.. code-block:: console

   PYTHONPATH=../mixle python -m pytest

The optional dependency profile matters. Solver and inversion paths may use
NumPy, SciPy, Torch, or sparse linear algebra libraries depending on the module
being exercised. Release notes should record which extras were installed for
each validation run so that a missing optional backend is not mistaken for a
solver failure.

Smoke test the public import surface:

.. code-block:: console

   python - <<'PY'
   from mixle_pde import DiffusionOperator, PDE

   print(DiffusionOperator)
   print(PDE)
   PY

Build this documentation with warnings treated as failures:

.. code-block:: console

   PYTHONPATH=../mixle \
     make -C docs html SPHINXOPTS="-W --keep-going"

For release evidence, also run the package tests from an environment that does
not inherit the developer shell's ambient ``PYTHONPATH``. That catches missing
declared dependencies and stale imports before the package is published.

Optional Backend Notes
----------------------

Record whether SciPy sparse routines, Torch-backed paths, or other optional
backends were installed for each validation run. If an optional backend is not
available, examples should skip clearly or use the documented NumPy/SciPy
fallback. Do not treat a developer environment with extra solvers as proof that
the base install behaves the same way.

Frozen Surface Expectations
---------------------------

During the paused-development period, install documentation should describe the
current package and optional backends only. Add new dependency instructions only
when the corresponding code and validation evidence are committed.
