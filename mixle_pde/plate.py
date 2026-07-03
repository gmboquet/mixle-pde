"""Kirchhoff-Love thin-plate bending, static and dynamic, on a rectangular grid.

The transverse deflection ``w(x, y, t)`` of a thin elastic plate obeys the fourth-order equation

    rho*h * w_tt = -D * biharmonic(w) + f(x, y, t),    D = E*h^3 / (12*(1 - nu^2)),

with ``biharmonic(w) = laplacian(laplacian(w))`` the flexural (bending) operator and ``D`` the flexural
rigidity. This is the plate analogue of the beam/membrane equations already in the stack: static bending is
a fourth-order Poisson problem, dynamic bending is a second-order-in-time wave-like system.

We take simply-supported (Navier) edges, ``w = 0`` and ``laplacian(w) = 0`` on the boundary -- the case with
a closed-form double-sine-series solution to check against. The fourth-order operator factors into two
second-order Poisson problems: with the moment field ``M = -laplacian(w)`` (up to the sign, the bending
moment), the Navier conditions become ``w = 0`` and ``M = 0`` on the edges, so ``D * biharmonic(w) = f`` is
``L @ (L @ w) = f / D`` for ``L`` the Dirichlet negative Laplacian. The static solve assembles that biharmonic
as ``laplacian @ laplacian`` and inverts it with the adjoint sparse solve; the dynamic stepper applies the
biharmonic matrix-free as two passes of the 5-point Laplacian and leapfrogs ``(w, v = w_t)``.
"""

from __future__ import annotations

import numpy as np


class KirchhoffPlate:
    """A thin-plate (Kirchhoff-Love) bending solver on an ``n x n`` grid over an ``a x b`` rectangle.

    ``KirchhoffPlate(n, E=..., nu=..., h=..., rho=..., a=..., b=...)`` fixes the material (Young's modulus
    ``E``, Poisson ratio ``nu``, thickness ``h``, density ``rho``) and the plate extent. The flexural
    rigidity ``D = E*h^3 / (12*(1 - nu^2))`` follows. Edges are simply supported (``w = 0``,
    ``laplacian(w) = 0``).

    Use ``static(load, ops)`` for the equilibrium deflection under a load, or ``step((w, v), ops, load=...)``
    to advance the differentiable dynamic system one leapfrog step.
    """

    def __init__(
        self,
        n: int,
        *,
        E: float = 2.0e11,
        nu: float = 0.3,
        h: float = 0.01,
        rho: float = 7800.0,
        a: float = 1.0,
        b: float = 1.0,
    ):
        self.n = int(n)
        self.E = float(E)
        self.nu = float(nu)
        self.h = float(h)
        self.rho = float(rho)
        self.a = float(a)
        self.b = float(b)
        self.D = self.E * self.h**3 / (12.0 * (1.0 - self.nu**2))
        self.hx = self.a / (self.n - 1)  # grid spacing along x (first index)
        self.hy = self.b / (self.n - 1)  # grid spacing along y (second index)
        mask = np.ones((self.n, self.n))
        mask[0] = mask[-1] = mask[:, 0] = mask[:, -1] = 0.0
        self._mask = mask.ravel()  # 1 on interior nodes, 0 on the simply-supported edges

    # -- the negative Laplacian L = -lap (Dirichlet, identity boundary rows) ---------------------------
    def _neg_laplacian(self, ops):
        """Assemble ``L = -laplacian`` on the plate grid as ``(rows, cols, vals, n)``.

        For a square plate (``hx == hy``) this is exactly ``ops.laplacian``. For a general rectangle the
        two axes have different spacing, so the five-point stencil is built with per-axis weights; the form
        (identity rows on the boundary, ``2/hx^2 + 2/hy^2`` on the interior diagonal) matches ``ops.laplacian``.
        """
        n = self.n
        if abs(self.hx - self.hy) < 1e-12 * max(self.hx, self.hy):
            return ops.laplacian((n, n), spacing=self.hx)
        wx = 1.0 / self.hx**2
        wy = 1.0 / self.hy**2
        rows: list[int] = []
        cols: list[int] = []
        vals: list[float] = []
        for i in range(n):
            for j in range(n):
                k = i * n + j
                if i == 0 or i == n - 1 or j == 0 or j == n - 1:
                    rows.append(k)
                    cols.append(k)
                    vals.append(1.0)  # identity boundary row (the source sets the Dirichlet value)
                    continue
                rows += [k, k, k, k, k]
                cols += [k, (i - 1) * n + j, (i + 1) * n + j, i * n + (j - 1), i * n + (j + 1)]
                vals += [2.0 * wx + 2.0 * wy, -wx, -wx, -wy, -wy]
        r = ops.tensor(np.asarray(rows, dtype=np.int64)).long()
        c = ops.tensor(np.asarray(cols, dtype=np.int64)).long()
        v = ops.tensor(np.asarray(vals))
        return r, c, v, n * n

    def _biharmonic_operator(self, ops):
        """The biharmonic ``B = L @ L`` (``L = -lap``) with ``w = 0`` enforced on the boundary rows.

        Composing the two Dirichlet Laplacians realizes the Navier factorization ``L @ (L @ w) = f / D``:
        the inner ``L`` carries ``M = -lap(w)`` with ``M = 0`` implied at the edges, the outer ``L`` carries
        the moment balance, and the boundary rows are overwritten with the identity so the simply-supported
        ``w = 0`` condition is exact. Returns ``(rows, cols, vals, n)`` for the sparse solve.
        """
        import scipy.sparse as sp
        import torch

        rows, cols, vals, nn = self._neg_laplacian(ops)
        L = sp.csc_matrix((vals.detach().cpu().numpy(), (rows.cpu().numpy(), cols.cpu().numpy())), shape=(nn, nn))
        B = (L @ L).tolil()
        for bi in np.where(self._mask == 0.0)[0]:  # w = 0 on the simply-supported edges
            B.rows[bi] = [int(bi)]
            B.data[bi] = [1.0]
        B = B.tocoo()
        r = torch.as_tensor(B.row, dtype=torch.long)
        c = torch.as_tensor(B.col, dtype=torch.long)
        v = torch.as_tensor(B.data, dtype=torch.float64)
        return r, c, v, nn

    def static(self, load, ops):
        """Solve the static bending equilibrium ``D * biharmonic(w) = f`` for the deflection ``w``.

        ``load`` is the transverse load ``f`` (a scalar broadcast over the interior, or a per-node field of
        length ``n*n``). The biharmonic is assembled as ``laplacian @ laplacian`` with simply-supported
        boundary rows, and ``B w = f / D`` is solved with the adjoint sparse solve. Returns the flat ``w``.
        """
        rows, cols, vals, nn = self._biharmonic_operator(ops)
        f = load if hasattr(load, "shape") and getattr(load, "ndim", 0) > 0 else load * ops.zeros(nn).add(1.0)
        f = ops.tensor(f) if not hasattr(f, "reshape") else f
        rhs = (f / self.D) * ops.tensor(self._mask)  # zero on the boundary rows (w = 0)
        return ops.sparse_solve(rows, cols, vals, nn, rhs)

    # -- matrix-free biharmonic for the dynamic stepper ----------------------------------------------
    def _lap(self, u, ops):
        """The 5-point Laplacian ``lap(u)`` (interior; zero on the edges), with per-axis spacing."""
        n = self.n
        A = u.reshape(n, n)
        out = ops.zeros(n, n)
        out[1:-1, 1:-1] = (A[2:, 1:-1] + A[:-2, 1:-1] - 2 * A[1:-1, 1:-1]) / self.hx**2 + (
            A[1:-1, 2:] + A[1:-1, :-2] - 2 * A[1:-1, 1:-1]
        ) / self.hy**2
        return out.reshape(-1)

    def _biharmonic(self, w, ops):
        """``biharmonic(w) = lap(lap(w))`` matrix-free, as two 5-point Laplacian passes.

        The inner pass produces ``lap(w)``, which is zero on the edges (the Navier ``lap(w) = 0`` condition);
        the outer pass differentiates it again. Differentiable in ``w`` for the leapfrog stepper.
        """
        return self._lap(self._lap(w, ops), ops)

    def step(self, state, ops, load=0.0):
        """Advance ``(w, v = w_t)`` one leapfrog step of ``rho*h * w_tt = -D * biharmonic(w) + f``.

        ``state`` is the pair ``(w, v)`` of flat fields; ``load`` is the (scalar or per-node) transverse load
        for this step. Uses the symplectic update ``w <- w + dt v`` then ``v <- v + dt * accel(w_new)`` with
        ``accel = (-D * biharmonic(w) + f) / (rho*h)`` -- the plate analogue of the wave stepper. The time
        step ``dt`` is set on the instance (see :meth:`dynamic_dt`)."""
        w, v = state
        dt = self.dt
        w_next = w + dt * v
        accel = (-self.D * self._biharmonic(w_next, ops) + load) / (self.rho * self.h)
        v_next = v + dt * accel
        return w_next, v_next

    def dynamic_dt(self, safety: float = 0.5) -> float:
        """A stable explicit time step for the leapfrog stepper (fourth-order CFL, ``dt ~ h^2``).

        The biharmonic's largest eigenvalue scales like ``(4/hx^2 + 4/hy^2)^2``, so stability needs
        ``dt <= 2 sqrt(rho*h / D) / (4/hx^2 + 4/hy^2)``; ``safety`` (< 1) backs off from that limit. Sets and
        returns ``self.dt``."""
        c = np.sqrt(self.rho * self.h / self.D)
        lam = 4.0 / self.hx**2 + 4.0 / self.hy**2
        self.dt = float(safety * 2.0 * c / lam)
        return self.dt
