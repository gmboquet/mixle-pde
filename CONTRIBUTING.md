# Contributing

Thanks for your interest in mixle-pde. This is a `mixle.ppl` plugin: it depends on
[mixle](https://github.com/gmboquet/mixle) and never the reverse.

## Development setup

mixle-pde depends directly on mixle, numpy, scipy, and torch (the `ops` backend every solver runs
through requires torch).

```bash
pip install -e ".[test,lint]"
```

## Tests

```bash
pytest                              # runs the suite (-n auto via pyproject)
pytest tests/wave3d_test.py -q      # a single file
```

Every solver ships with a test that checks it against an exact analytical solution (a normal-mode
frequency, a Poiseuille profile, a Young-Laplace jump, a decaying eigenmode). New solvers are expected
to do the same: agreement with a closed-form reference is the acceptance bar, not just "it runs".

## Style

`ruff` is the linter and formatter, pinned to the version in `[project.optional-dependencies].lint`.

```bash
ruff check .
ruff format .
```

CI runs both in a blocking `lint` job, so keep the tree clean.

## Writing a new solver

The forward solvers follow one convention (see `mixle_pde/wave3d.py` or `mixle_pde/flow3d.py` as
templates):

- a class `<Equation><Dim>` with `__init__(n, *, dt, spacing=None, ...)` holding the grid and any
  precomputed operators;
- a pure `step(state, ..., ops)` that advances one time step;
- all math goes through the backend-agnostic `ops` namespace (`mixle_pde/ops.py`) so the solver stays
  differentiable and never imports a tensor library directly. That differentiability is what lets a
  solver drop into the `Differential` inverse stack.

Re-export the headline class from `mixle_pde/__init__.py` and add it to `__all__`.

## Pull requests

Small, focused commits. Make sure `ruff check`, `ruff format --check`, and `pytest` all pass before
opening a PR.
