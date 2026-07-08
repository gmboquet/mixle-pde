Installation
============

``mixle-pde`` installs into an environment that can import core ``mixle``.

From a checkout:

.. code-block:: console

   python -m pip install -e .
   python -m pip install -e ".[test]"
   python -m pip install -e ".[docs]"

When working from the sibling workspace without installing core Mixle, put core
on ``PYTHONPATH``:

.. code-block:: console

   PYTHONPATH=/Users/grantboquet/mixle/mixle python -m pytest

Smoke test:

.. code-block:: console

   python - <<'PY'
   from mixle_pde import DiffusionOperator, PDE

   print(DiffusionOperator)
   print(PDE)
   PY

Build this documentation:

.. code-block:: console

   PYTHONPATH=/Users/grantboquet/mixle/mixle make -C docs html
