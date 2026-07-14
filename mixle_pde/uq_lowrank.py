"""Matrix-free / low-rank Hessian UQ + selected inversion (workstream C2, inversion realism & scale).

Two independent survey-scale primitives that both avoid ever materializing a dense ``(n, n)`` array:

* :func:`randomized_lowrank_hessian` -- a randomized eigendecomposition of a symmetric PSD Hessian
  accessed *only* through a Hessian-vector product (HVP). This is the Bui-Thanh/Flath/Petra low-rank
  Laplace-approximation trick: the prior-preconditioned data-misfit Hessian ``H = P^{1/2} J^T R^-1 J
  P^{1/2}`` usually has a fast-decaying spectrum (data informs only a handful of directions in a
  survey-scale model), so its top-``k`` eigenpairs summarize essentially all of the data's information
  content. The Hessian itself is never assembled -- only ``k + oversample`` HVP evaluations are needed.

* :func:`takahashi_selected_inversion` -- the diagonal of a sparse SPD precision matrix's inverse
  (the posterior marginal variances), computed via the classic Takahashi/Erisman-Tinney recursion over
  the sparse Cholesky (LDL^T) factor's own nonzero pattern. Memory is ``O(nnz(L))``, not ``O(n^2)``: for
  a survey-scale precision (n ~ 10^5-10^6 cells) a dense inverse is tens of gigabytes and infeasible,
  while the factor's fill is typically a small multiple of ``n``.

Both are consumed by :class:`mixle_pde.latent.SparsePosteriorPrecision.marginal_variance` (selected
inversion, replacing a dense ``solve(eye)``) and by the optional low-rank attachment in
:func:`mixle_pde.field_inversion.sparse_linear_gaussian_invert` and
:func:`mixle_pde.field_gauss_newton.gauss_newton_hessian_hvp` (the matrix-free HVP the low-rank routine
consumes).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def randomized_lowrank_hessian(
    hvp: Callable[[np.ndarray], np.ndarray],
    n: int,
    k: int,
    rng: np.random.Generator,
    *,
    oversample: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Randomized eigendecomposition of a symmetric PSD Hessian, accessed only through ``hvp``.

    ``hvp(v) -> H v`` is the Hessian-vector product of the (typically prior-preconditioned) data-misfit
    Hessian; ``H`` is never assembled as an ``(n, n)`` array. Draws an ``n x (k + oversample)`` Gaussian
    sketch, orthonormalizes it, projects ``H`` onto that subspace (one more batch of HVPs), and
    eigendecomposes the resulting small dense matrix (Halko/Martinsson/Tropp Algorithm 5.3, the
    symmetric-matrix specialization of randomized SVD). Returns ``(U, s)`` with ``U`` shape ``(n, k)``
    (orthonormal, approximate top eigenvectors) and ``s`` shape ``(k,)`` (approximate eigenvalues,
    descending, clipped to be non-negative since ``H`` is PSD).

    Uses ``sklearn.utils.extmath.randomized_svd`` for the final small dense factorization when sklearn is
    importable (its randomized_svd only accepts array-likes, not a matrix-free operator, so the
    matrix-free sketch/projection is always done by hand here); falls back to ``scipy.linalg.eigh`` on
    the same small matrix otherwise. Either path gives numerically identical eigenpairs for a symmetric
    PSD input up to sign.
    """
    if n < 0:
        raise ValueError("n must be non-negative.")
    if k < 0:
        raise ValueError("k must be non-negative.")
    m = min(n, k + max(int(oversample), 0))
    if m == 0 or n == 0:
        return np.zeros((n, 0)), np.zeros(0)

    omega = rng.standard_normal((n, m))
    sketch = np.empty((n, m))
    for i in range(m):
        sketch[:, i] = np.asarray(hvp(omega[:, i]), dtype=float)

    basis, _ = np.linalg.qr(sketch)

    projected = np.empty((n, m))
    for i in range(m):
        projected[:, i] = np.asarray(hvp(basis[:, i]), dtype=float)

    small = basis.T @ projected
    small = 0.5 * (small + small.T)  # symmetrize away HVP round-off noise

    keep = min(k, m)
    try:
        from sklearn.utils.extmath import randomized_svd

        seed = int(rng.integers(0, 2**31 - 1))
        vecs, vals, _ = randomized_svd(small, n_components=keep, random_state=seed)
    except ImportError:
        from scipy.linalg import eigh

        eigvals, eigvecs = eigh(small)
        order = np.argsort(eigvals)[::-1][:keep]
        vals = np.clip(eigvals[order], 0.0, None)
        vecs = eigvecs[:, order]

    U = basis @ vecs[:, :keep]
    s = np.clip(np.asarray(vals[:keep], dtype=float), 0.0, None)
    if keep < k:
        U = np.hstack([U, np.zeros((n, k - keep))])
        s = np.concatenate([s, np.zeros(k - keep)])
    return U, s


def _ldlt_natural_order(precision_csc):
    """``precision = L D L^T`` (unit-lower ``L``, diagonal ``D``) via ``splu`` in natural, no-pivot mode.

    ``diag_pivot_thresh=0`` disables numerical row pivoting (safe for a genuinely SPD matrix -- diagonal
    pivots never need to be avoided) and ``permc_spec='NATURAL'`` disables column reordering, so both
    permutations come back as the identity and ``L``/``U`` sit directly in the caller's original index
    order (verified below, since the recursion that follows assumes column ``j``'s children are exactly
    the original row indices with a nonzero sub-diagonal entry in column ``j``).
    """
    import numpy as np
    from scipy.sparse.linalg import splu

    n = precision_csc.shape[0]
    lu = splu(precision_csc, diag_pivot_thresh=0.0, permc_spec="NATURAL")
    if not (np.array_equal(lu.perm_r, np.arange(n)) and np.array_equal(lu.perm_c, np.arange(n))):
        raise ValueError(
            "takahashi_selected_inversion requires a symmetric positive-definite `precision_csc` "
            "(natural-order LU needed a pivot outside the diagonal)."
        )
    L = lu.L.tocsc()
    D = np.asarray(lu.U.diagonal(), dtype=float)
    return L, D


def _column_children(L) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Per-column ``(rows > j, values)`` of the strictly-sub-diagonal nonzeros of unit-lower ``L``."""
    n = L.shape[0]
    indptr, indices, data = L.indptr, L.indices, L.data
    rows: list[np.ndarray] = [None] * n  # type: ignore[list-item]
    vals: list[np.ndarray] = [None] * n  # type: ignore[list-item]
    for j in range(n):
        start, end = indptr[j], indptr[j + 1]
        seg = indices[start:end]
        mask = seg > j
        rows[j] = seg[mask].astype(np.int64, copy=False)
        vals[j] = data[start:end][mask]
    return rows, vals


def _takahashi_recursion(
    n: int, children_rows: list[np.ndarray], children_vals: list[np.ndarray], D: np.ndarray
) -> np.ndarray:
    """Backward Takahashi/Erisman-Tinney recursion: ``diag(precision^-1)`` over ``L``'s fill pattern only.

    For column ``j`` (processed from ``n-1`` down to ``0``), the nonzero sub-diagonal rows of column
    ``j`` form a clique in the elimination graph (a standard fact about Cholesky/LDL^T fill), so every
    pairwise selected-inverse entry the recursion needs for those rows was already produced by an earlier
    (higher-index) column -- the algorithm never needs an entry it hasn't already computed, and it never
    touches an (i, k) pair outside `L`'s own sparsity pattern. Memory is one float per stored (i, k) pair
    in that pattern, i.e. ``O(nnz(L))``, independent of how the caller chooses to store it.
    """
    z_rows: list[dict] = [dict() for _ in range(n)]  # z_rows[i][k] = (precision^-1)[i, k], k <= i
    diag = np.empty(n)
    for j in range(n - 1, -1, -1):
        rows = children_rows[j]
        vals = children_vals[j]
        m = rows.size
        if m == 0:
            zjj = 1.0 / D[j]
        else:
            v = np.empty(m)
            for a in range(m):
                ia = int(rows[a])
                z_ia = z_rows[ia]
                acc = 0.0
                for b in range(m):
                    ib = int(rows[b])
                    acc += (z_ia.get(ib, 0.0) if ib <= ia else z_rows[ib].get(ia, 0.0)) * vals[b]
                v[a] = acc
            for a in range(m):
                z_rows[int(rows[a])][j] = -v[a]
            zjj = 1.0 / D[j] + float(vals @ v)
        z_rows[j][j] = zjj
        diag[j] = zjj
    return diag


def _selected_inversion_by_columns(precision_csc) -> np.ndarray:
    """Safety-net fallback: exact diagonal via one factor-solve per unit vector, O(n) memory (never the
    dense (n, n) inverse). Only used if the natural-order LDL^T extraction above can't be verified."""
    from scipy.sparse.linalg import splu

    n = precision_csc.shape[0]
    factor = splu(precision_csc)
    diag = np.empty(n)
    e = np.zeros(n)
    for i in range(n):
        e[i] = 1.0
        diag[i] = factor.solve(e)[i]
        e[i] = 0.0
    return diag


def takahashi_selected_inversion(precision_csc) -> np.ndarray:
    """``diag(precision^-1)`` for a sparse SPD ``precision``, never forming the dense inverse.

    Runs the Takahashi/Erisman-Tinney selected-inversion recursion over the sparse LDL^T factor's own
    nonzero pattern, so memory is ``O(nnz(L))`` -- for a survey-scale ``n`` (10^5-10^6 cells) that is a
    small multiple of ``n``, where the dense alternative (``n^2`` float64 entries) is infeasible. Uses
    CHOLMOD (``scikit-sparse``) for the factorization when it is importable (it usually gives a
    better-conditioned, less-fill factor for a general sparsity pattern); otherwise factors via
    ``scipy.sparse.linalg.splu`` in natural, no-pivot mode, which is exact for a genuinely SPD matrix.
    """
    import scipy.sparse as sp

    A = sp.csc_matrix(precision_csc, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("precision matrix must be square.")
    n = A.shape[0]
    if n == 0:
        return np.zeros(0)
    if n == 1:
        return np.array([1.0 / A[0, 0]], dtype=float)

    try:
        from sksparse.cholmod import cholesky as _cholmod_cholesky

        factor = _cholmod_cholesky(A)
        L = factor.L()
        perm = np.asarray(factor.P())
        D = np.asarray(L.diagonal(), dtype=float) ** 2
        L_unit = (L @ sp.diags(1.0 / np.sqrt(D))).tocsc()
        rows, vals = _column_children(L_unit)
        diag_perm = _takahashi_recursion(n, rows, vals, D)
        diag = np.empty(n)
        diag[perm] = diag_perm
        return diag
    except ImportError:
        pass
    except Exception:
        # CHOLMOD is an optional accelerator; any surprise in its (version-sensitive) API falls back
        # to the always-correct pure-scipy path below rather than failing the whole call.
        pass

    try:
        L, D = _ldlt_natural_order(A)
    except ValueError:
        return _selected_inversion_by_columns(A)
    rows, vals = _column_children(L)
    return _takahashi_recursion(n, rows, vals, D)
