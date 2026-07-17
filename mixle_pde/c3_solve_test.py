"""Definition-of-Done test for C3 -- iterative sparse solvers + factorization caching.

Two things must hold once the dense-inverse covariance sites are replaced: (1) :func:`spd_solve` handles
a genuinely survey-scale SPD system -- a dense inverse would be infeasible -- within a modest memory
budget, and (2) the four call sites the work order names no longer call ``np.linalg.inv`` on that
covariance path.
"""

from __future__ import annotations

import importlib
import inspect
import tracemalloc

import numpy as np
import scipy.sparse as sp

from mixle_pde.linear_solve import make_factor_cache, spd_solve


def _laplacian_3d_spd(n: int, jitter: float = 1.0e-2):
    """A 3D 7-point graph-Laplacian SPD operator via Kronecker-sum assembly (no dense intermediate) --
    stands in for a prior-precision / Gauss-Newton normal-equations matrix at survey scale."""
    e = np.ones(n)
    lap1d = sp.diags([-e[:-1], 2.0 * e, -e[:-1]], offsets=[-1, 0, 1], format="csr")
    ident = sp.identity(n, format="csr")
    lap3d = (
        sp.kron(sp.kron(lap1d, ident), ident)
        + sp.kron(sp.kron(ident, lap1d), ident)
        + sp.kron(sp.kron(ident, ident), lap1d)
    )
    return (lap3d + jitter * sp.identity(n**3, format="csr")).tocsr()


def test_spd_solve_scales_to_125000_cells_within_memory_budget():
    n_side = 50
    a = _laplacian_3d_spd(n_side)
    assert a.shape == (125_000, 125_000)

    rng = np.random.default_rng(0)
    x_true = rng.standard_normal(a.shape[0])
    b = a @ x_true

    tracemalloc.start()
    cache = make_factor_cache()
    x = spd_solve(a, b, method="cg", precond="amg", tol=1e-10, factor_cache=cache)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    relative_residual = np.linalg.norm(a @ x - b) / np.linalg.norm(b)
    assert relative_residual < 1e-6
    # a dense np.linalg.inv here would need 125_000**2 floats (~125 GB); spd_solve must stay well under 1 GB.
    assert peak < 1_000_000_000


def test_spd_solve_reuses_the_cached_preconditioner():
    n_side = 20
    a = _laplacian_3d_spd(n_side)
    rng = np.random.default_rng(1)
    b1 = rng.standard_normal(a.shape[0])
    b2 = rng.standard_normal(a.shape[0])

    cache = make_factor_cache()
    spd_solve(a, b1, precond="amg", factor_cache=cache)
    n_entries_after_first = len(cache)
    spd_solve(a, b2, precond="amg", factor_cache=cache)
    # the second solve against the same `a` object must not add a new preconditioner cache entry.
    assert len(cache) == n_entries_after_first


_MODIFIED_SITES = [
    ("mixle_pde.field_inversion", "_noise_precision"),
    ("mixle_pde.field_inversion", "linear_gaussian_invert"),
    ("mixle_pde.field_gauss_newton", "gauss_newton_invert"),
    ("mixle_pde.field_priors", "joint_linear_gaussian_invert"),
]


def test_covariance_path_never_calls_dense_numpy_inverse():
    for module_name, func_name in _MODIFIED_SITES:
        module = importlib.import_module(module_name)
        source = inspect.getsource(getattr(module, func_name))
        assert "np.linalg.inv" not in source, f"{module_name}.{func_name} still calls np.linalg.inv"
