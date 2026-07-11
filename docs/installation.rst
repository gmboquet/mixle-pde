Installation
============

``mixle-pde`` requires core ``mixle`` to be importable. Install core first
when validating the package family from sibling checkouts; install the
published core package when testing an isolated release candidate.

From a checkout:

.. code-block:: console

   python -m pip install -e .
   python -m pip install -e ".[test]"
   python -m pip install -e ".[docs]"

When working from the sibling workspace without installing core Mixle, put core
on ``PYTHONPATH`` for the command being validated:

.. code-block:: console

   PYTHONPATH=../mixle python -m pytest

NumPy, SciPy, and Torch are base dependencies, not optional extras: every
solver and inverse callback runs through the Torch-backed ``ops`` namespace
(``mixle_pde/ops.py`` imports torch unconditionally), so a plain install
brings everything the solver and inversion paths need. The only genuinely
optional pieces are the geophysical data loaders in ``mixle_pde.env_data``
(GEBCO, WOA/Argo, DEM, ERA5), which import their heavier backends lazily and
raise a clear ``ImportError`` naming what is missing.

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
