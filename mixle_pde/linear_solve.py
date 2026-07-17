"""Iterative sparse solvers + factorization caching for survey-scale linear-Gaussian inversion (C3).

The exact linear-Gaussian and Gauss-Newton paths (:mod:`mixle_pde.field_inversion`,
:mod:`mixle_pde.field_gauss_newton`, :mod:`mixle_pde.field_priors`) size their posterior covariance with
``np.linalg.inv`` on an ``(n, n)`` dense array. That is fine at grid scale but infeasible at survey scale
(a 50^3 = 125,000-cell normal-equations matrix would need a 125,000^2 dense array -- on the order of
100 GB). This module is the scale-up: an SPD sparse solve (:func:`spd_solve`, CG/MINRES with an algebraic
multigrid preconditioner, falling back to incomplete-LU when ``pyamg`` is unavailable), a Hutchinson
marginal-standard-deviation estimator that never forms ``diag(A^-1)`` explicitly
(:func:`marginal_std_cg`), and a factorization cache (:func:`make_factor_cache`) threaded through repeated
solves so the preconditioner / direct-solve fallback for a given operator is built once and reused --
across Hutchinson probes, across Gauss-Newton iterations that keep the same operator object, or across
repeated PDE solves with an unchanged coefficient field (see :func:`mixle_pde.pde_solve.sparse_solve`).

:func:`dense_spd_solve` is the small/medium-scale sibling: a Cholesky-factor solve that replaces
``np.linalg.inv`` on the *dense* covariance paths without changing their O(n^3) complexity class (those
functions are documented as materializing a dense covariance already, so the fix here is "never call
`inv`", not "make it sparse") -- Cholesky is the numerically appropriate factorization for a matrix known
to be SPD (a posterior precision or Gauss-Newton Hessian always is, up to the jitter that keeps it so),
and is roughly twice as cheap as forming a general inverse via LU.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.linalg import cho_factor, cho_solve
from scipy.sparse.linalg import LinearOperator, cg, minres, spilu, splu

__all__ = [
    "spd_solve",
    "marginal_std_cg",
    "make_factor_cache",
    "dense_spd_solve",
]


def make_factor_cache() -> dict:
    """A fresh factorization cache.

    Keyed by ``(id(A), nnz)`` (see :func:`_matrix_key`) so it is safe to share a single cache across many
    solves: a cache entry is only reused when the *same operator object* recurs (e.g. the same
    Gauss-Newton Hessian across Hutchinson probes, or the same PDE operator across time steps), never when
    a different matrix merely happens to look similar.
    """
    return {}


def _matrix_key(A) -> tuple[int, int]:
    nnz = int(A.nnz) if hasattr(A, "nnz") else int(np.asarray(A).size)
    return (id(A), nnz)


def _amg_preconditioner(A_csr):
    """A pyamg smoothed-aggregation V-cycle preconditioner, or ``None`` if pyamg is unavailable."""
    try:
        import pyamg
    except ImportError:
        return None
    ml = pyamg.smoothed_aggregation_solver(A_csr)
    return ml.aspreconditioner()


def _ilu_preconditioner(A_csr):
    """A drop-tolerance incomplete-LU ``LinearOperator`` -- the fallback when pyamg is unavailable."""
    ilu = spilu(A_csr.tocsc(), drop_tol=1.0e-5, fill_factor=10)
    n = A_csr.shape[0]
    return LinearOperator((n, n), matvec=ilu.solve, dtype=A_csr.dtype)


def _preconditioner(A_csr, precond: str | None, factor_cache: dict | None):
    """Build (or fetch from ``factor_cache``) the requested preconditioner for ``A_csr``."""
    if precond is None:
        return None
    key = ("precond", precond, *_matrix_key(A_csr)) if factor_cache is not None else None
    if factor_cache is not None and key in factor_cache:
        return factor_cache[key]
    if precond == "amg":
        m = _amg_preconditioner(A_csr)
        if m is None:
            m = _ilu_preconditioner(A_csr)
    elif precond == "ilu":
        m = _ilu_preconditioner(A_csr)
    else:
        raise ValueError(f"unknown precond {precond!r}; use 'amg', 'ilu', or None.")
    if factor_cache is not None:
        factor_cache[key] = m
    return m


def _direct_fallback(A_csr, b: np.ndarray, factor_cache: dict | None) -> np.ndarray:
    """An exact cached sparse-LU solve -- used when the iterative solve stalls, so a poorly-conditioned
    system still returns an accurate answer rather than a truncated CG/MINRES iterate."""
    key = ("lu", *_matrix_key(A_csr)) if factor_cache is not None else None
    lu = factor_cache.get(key) if factor_cache is not None else None
    if lu is None:
        lu = splu(A_csr.tocsc())
        if factor_cache is not None:
            factor_cache[key] = lu
    return np.asarray(lu.solve(b), dtype=float)


def _spd_solve_torch(A_csr, b: np.ndarray, *, tol: float, maxiter: int | None) -> np.ndarray:
    """A plain torch conjugate-gradient solve -- the optional GPU backend (CPU default).

    Moves the sparse operator to a ``torch.sparse_csr_tensor`` and runs CG with torch's own linear
    algebra, so a CUDA build solves on-device with no other code change. This backend is optional and
    untested at survey scale (see the module Non-goals); the CPU path (``backend="cpu"``, the default)
    is what :func:`spd_solve`'s Definition-of-Done exercises.
    """
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    a = A_csr.tocsr()
    a_t = torch.sparse_csr_tensor(
        torch.as_tensor(a.indptr, dtype=torch.int64),
        torch.as_tensor(a.indices, dtype=torch.int64),
        torch.as_tensor(a.data, dtype=torch.float64),
        size=a.shape,
    ).to(device)
    b_t = torch.as_tensor(b, dtype=torch.float64, device=device)
    n = a.shape[0]
    x = torch.zeros_like(b_t)
    r = b_t - a_t @ x
    p = r.clone()
    rs_old = torch.dot(r, r)
    b_norm = torch.linalg.norm(b_t).clamp_min(1.0e-300)
    for _ in range(maxiter or (2 * n)):
        ap = a_t @ p
        alpha = rs_old / torch.dot(p, ap).clamp_min(1.0e-300)
        x = x + alpha * p
        r = r - alpha * ap
        rs_new = torch.dot(r, r)
        if torch.sqrt(rs_new) / b_norm < tol:
            break
        p = r + (rs_new / rs_old) * p
        rs_old = rs_new
    return x.detach().cpu().numpy()


def spd_solve(
    A_csr,
    b: np.ndarray,
    *,
    method: str = "cg",
    precond: str | None = "amg",
    tol: float = 1.0e-8,
    factor_cache: dict | None = None,
    backend: str = "cpu",
    maxiter: int | None = None,
) -> np.ndarray:
    """Solve the SPD sparse system ``A_csr x = b`` without ever forming ``A^-1``.

    ``method`` is ``"cg"`` (conjugate gradient) or ``"minres"`` (for a symmetric-but-not-quite-positive
    system, e.g. a slightly indefinite Hessian near the jitter floor); ``precond`` is ``"amg"`` (pyamg
    smoothed-aggregation, falling back to incomplete-LU when pyamg is not installed), ``"ilu"``, or
    ``None`` for unpreconditioned CG. ``factor_cache`` (see :func:`make_factor_cache`) lets the
    preconditioner (and, if the iterative solve stalls, the direct-solve fallback factor) be built once
    and reused across repeated calls against the *same* ``A_csr`` object -- e.g. one call per Hutchinson
    probe in :func:`marginal_std_cg`. ``backend="torch"`` runs an optional GPU-capable CG instead (CPU is
    the default and the only backend exercised at scale; see the module Non-goals).

    If the iterative solve does not reach ``tol`` within ``maxiter`` iterations, falls back to an exact
    sparse LU factor solve rather than silently returning an under-converged iterate.
    """
    # avoid re-wrapping an already-CSR matrix: `factor_cache` keys off `id(A_csr)`, so preserving the
    # caller's object identity (rather than always allocating a fresh `csr_matrix` copy) is what lets a
    # repeated call against the same operator actually hit the cache.
    if not sp.isspmatrix_csr(A_csr):
        A_csr = sp.csr_matrix(A_csr)
    b_arr = np.asarray(b, dtype=float).reshape(-1)
    n = A_csr.shape[0]
    if A_csr.shape[0] != A_csr.shape[1]:
        raise ValueError(f"A_csr must be square, got {A_csr.shape}.")
    if b_arr.shape != (n,):
        raise ValueError(f"b must have shape ({n},), got {b_arr.shape}.")

    if backend == "torch":
        return _spd_solve_torch(A_csr, b_arr, tol=tol, maxiter=maxiter)
    if backend != "cpu":
        raise ValueError(f"unknown backend {backend!r}; use 'cpu' or 'torch'.")

    solver = {"cg": cg, "minres": minres}.get(method)
    if solver is None:
        raise ValueError(f"unknown method {method!r}; use 'cg' or 'minres'.")

    m = _preconditioner(A_csr, precond, factor_cache)
    x, info = solver(A_csr, b_arr, rtol=tol, maxiter=maxiter, M=m)
    if info != 0:
        x = _direct_fallback(A_csr, b_arr, factor_cache)
    return np.asarray(x, dtype=float)


def marginal_std_cg(
    A_csr,
    *,
    probes: int = 32,
    rng: np.random.Generator | None = None,
    factor_cache: dict | None = None,
    precond: str | None = "amg",
    tol: float = 1.0e-6,
) -> np.ndarray:
    """Hutchinson estimate of the marginal posterior standard deviation, i.e. ``sqrt(diag(A^-1))``.

    Draws ``probes`` Rademacher probe vectors ``z`` (iid +-1 entries), solves ``A x = z`` for each through
    :func:`spd_solve` (sharing one ``factor_cache`` across probes so the preconditioner is amortized over
    the whole batch), and averages ``z * x`` -- an unbiased estimator of ``diag(A^-1)`` that never
    materializes the ``(n, n)`` inverse. The square root matches this codebase's ``marginal_std =
    sqrt(marginal_variance)`` convention (e.g. :meth:`mixle_pde.field_assimilation.EnsemblePosterior.marginal_std`);
    the raw Hutchinson estimate is clipped at 0 first since it is noisy near-zero variances.
    """
    A_csr = sp.csr_matrix(A_csr)
    n = A_csr.shape[0]
    rng = rng if rng is not None else np.random.default_rng()
    if factor_cache is None:
        factor_cache = make_factor_cache()  # always share the preconditioner across this batch of probes

    accum = np.zeros(n, dtype=float)
    for _ in range(int(probes)):
        z = rng.integers(0, 2, size=n).astype(float) * 2.0 - 1.0
        x = spd_solve(A_csr, z, precond=precond, tol=tol, factor_cache=factor_cache)
        accum += z * x
    variance = np.clip(accum / float(probes), 0.0, None)
    return np.sqrt(variance)


def dense_spd_solve(A: np.ndarray, B: np.ndarray, *, factor_cache: dict | None = None) -> np.ndarray:
    """Solve ``A X = B`` for a dense SPD ``A`` via a cached Cholesky factor.

    The direct replacement for ``np.linalg.inv(A) @ B`` (equivalently ``np.linalg.inv(A)`` when
    ``B = np.eye(n)``) on the small/medium dense-covariance paths (:func:`mixle_pde.field_inversion.linear_gaussian_invert`,
    :func:`mixle_pde.field_gauss_newton.gauss_newton_invert`,
    :func:`mixle_pde.field_priors.joint_linear_gaussian_invert`). Those paths keep an explicit dense
    covariance by contract, so this does not change their complexity class -- it removes the one thing
    those sites should never do, which is call the general-purpose ``inv`` on a matrix already known to be
    SPD, when a Cholesky factor-and-solve is both cheaper and exploits that structure. When
    ``factor_cache`` is supplied the factor is cached by ``id(A)`` so re-solving against the same precision
    object for a different right-hand side is free after the first call.
    """
    key = ("chol", *_matrix_key(A)) if factor_cache is not None else None
    cached = factor_cache.get(key) if factor_cache is not None else None
    if cached is None:
        cached = cho_factor(A, lower=True)
        if factor_cache is not None:
            factor_cache[key] = cached
    return np.asarray(cho_solve(cached, B), dtype=float)
