"""Pytest collection-wide fixtures.

mixle-pde runs its suite under pytest-xdist (``addopts = "-ra -n auto"`` in ``pyproject.toml``), so
each worker process executes many test files sequentially in one interpreter. Several tests call
``torch.set_default_dtype(torch.float64)`` with no cleanup; without isolation, whichever test last
sets that global leaves every later test on that same worker running under float64, regardless of
what the later test itself assumes. A plain float32 module then crashes with "mat1 and mat2 must
have the same dtype, but got Float and Double" -- a different victim set on every run, depending on
how xdist happened to distribute files to workers that run.
"""

import pytest


@pytest.fixture(autouse=True)
def _isolate_torch_global_state():
    """Snapshot and restore torch's process-global state around every test.

    Restores the default dtype (the confirmed, reproduced leak) and the global RNG state (the same
    class of hazard: an unseeded torch draw in one test would otherwise depend on which tests ran
    earlier in the same worker process). Also pins single-threaded, warn-only-deterministic execution
    before each test, since a prior test may have left multi-threaded or nondeterministic algorithms
    enabled.
    """
    try:
        import torch
    except ImportError:
        yield
        return

    default_dtype = torch.get_default_dtype()
    rng_state = torch.random.get_rng_state()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True, warn_only=True)
    try:
        yield
    finally:
        torch.set_default_dtype(default_dtype)
        torch.random.set_rng_state(rng_state)
