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

A base install (``pip install -e .``, zero extras) must import ``mixle_pde`` and
run the ``mixle_pde.problem_adapter`` compatibility-boundary tests with no
optional backend present; ``tests/packaging_test.py`` pins this contract
statically (no module in the package imports a heavy backend at module scope)
and, when a network is available, by resolving each extra with ``pip install
--dry-run``.

mixle-pde defines capability-family installation extras on top of the
numpy/scipy/pyproj base:

======================================== ==========================================================
Extra                                    Unlocks
======================================== ==========================================================
``fem``                                  AMG (``pyamg``) and CHOLMOD (``scikit-sparse``) accelerators
                                          for large assembled FEM/simplex systems
``mesh``                                 AMG preconditioning (``pyamg``) for mesh-scale assembled
                                          linear systems
``mpi``                                  Forward-declared distributed mesh/solve transport
                                          (``mpi4py``); no backend is wired to it yet
``fvm``                                  Differentiable finite-volume forwards (``torch``) plus AMG
                                          scale-up (``pyamg``)
``coupling``                             Differentiable coupled/multiphysics forwards (``torch``)
``inverse``                              Differentiable adjoints, randomized-SVD low-rank UQ, and
                                          sparse-scale posterior solves (``torch``, ``scikit-learn``,
                                          ``pyamg``, ``scikit-sparse``)
``surrogate``                            Neural surrogate distillation and calibration (``torch``,
                                          ``scikit-learn``)
``all``                                  The union of the seven capability extras above
``raster`` / ``netcdf`` / ``grib`` /     Single-format geoscience data ingest
``segy`` / ``las`` / ``potfield``
``data``                                 Convenience bundle of every data-format backend above
======================================== ==========================================================

Commercial or optional adapters (for example the sibling ``mixle_mlops``
task-cascade integration ``mixle_pde.surrogate`` can optionally bind to) are
never declared as a dependency of any extra and are never required to install
or test mixle-pde.

Frozen Surface Expectations
---------------------------

During the paused-development period, install documentation should describe the
current package and optional backends only. Add new dependency instructions only
when the corresponding code and validation evidence are committed.
