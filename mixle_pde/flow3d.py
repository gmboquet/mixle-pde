"""3D incompressible Navier-Stokes by Chorin projection (phase 5, the 3-D frontier of ``flow.py``).

The 2-D solver in :mod:`mixle_pde.flow` uses the streamfunction-vorticity formulation, which is clean in
the plane (one scalar streamfunction, automatic incompressibility) but does not extend to 3-D: vorticity is a
vector there and no single streamfunction exists. The standard replacement is the *velocity-pressure*
projection method (Chorin): advance the velocity ignoring the pressure, then project the result onto the
divergence-free subspace with a pressure Poisson solve. One step is

    u* = u + dt * ( -(u . grad) u + nu * laplacian(u) )        (advect + diffuse, per component)
    laplacian(p) = div(u*) / dt                                (pressure Poisson, the projection)
    u_{n+1} = u* - dt * grad(p)                                (correction -> divergence-free)

Every operator is the same differentiable machinery as the 2-D solver: advection, divergence, and the
pressure gradient are :func:`ops.grad` central differences; the componentwise diffusion Laplacian is the
3-D 7-point stencil; the pressure Poisson is a differentiable sparse solve through :func:`ops.sparse_solve`
(the Phase-1 adjoint solve). A forward that steps this solver is therefore differentiable end to end, and the
latent driver (an initial velocity, a viscosity, an inlet strength) flows through to whatever velocities a
forward records -- the same inverse-problem story as 2-D.

One subtlety makes the projection actually work. The Poisson operator must be *consistent* with the discrete
divergence and gradient it corrects: the correction is exact (``div(u_{n+1}) = div(u*) - dt * div(grad p)``)
only when the pressure Laplacian equals ``div(grad(.))`` formed from the very same central differences. The
plain 7-point ``ops.laplacian`` is a narrower stencil than the ``ops.grad`` composition, so pairing it with
``ops.grad`` leaves the divergence essentially uncorrected. So the pressure operator here is assembled once
as ``sum_axis grad_axis(grad_axis(.))`` -- the exact composition of :func:`ops.grad` with itself -- and that
single consistent solve drives the discrete divergence down by many orders of magnitude (to the solver
tolerance). It is a tiny Tikhonov term on the diagonal that pins the operator's null space (the wide central
stencil decouples the grid into odd/even sublattices) so the sparse factorization is well posed.

The walls are no-slip (velocity zero on the box boundary, enforced by the same interior mask 2-D uses). The
scheme is first-order in time (operator splitting) and explicit in diffusion, so ``dt`` obeys the usual
advection and diffusion CFL limits -- fine for the moderate-Reynolds, modest-grid regime where these inverse
problems are well posed; the sparse solve dominates the cost, so keep ``n`` moderate.
"""

from __future__ import annotations

import numpy as np


class NavierStokes3D:
    """A differentiable 3D incompressible Navier-Stokes stepper (velocity-pressure projection, explicit).

    ``NavierStokes3D(n, viscosity=..., dt=...)`` builds the forward on an ``n x n x n`` grid with no-slip
    walls (velocity zero on the box boundary). The state is the velocity triple ``(u, v, w)`` as three flat
    length-``n**3`` tensors. In a forward callback, advance with ``step(state, ops)``, read the components
    with ``velocity(state)`` (or ``u``/``v``/``w``), and check incompressibility with ``divergence(state, ops)``.
    """

    def __init__(
        self,
        n: int,
        *,
        viscosity: float,
        dt: float,
        spacing: float | None = None,
        pressure_reg: float = 1e-6,
    ):
        self.n = int(n)
        self.nu = float(viscosity)
        self.dt = float(dt)
        self.h = float(spacing) if spacing is not None else 1.0 / (n - 1)
        self.shape = (self.n, self.n, self.n)
        mask = np.ones(self.shape)
        mask[0] = mask[-1] = 0.0
        mask[:, 0] = mask[:, -1] = 0.0
        mask[:, :, 0] = mask[:, :, -1] = 0.0
        self._mask = mask.ravel()
        self._pressure_reg = float(pressure_reg)
        # Assemble the pressure Poisson operator ``div(grad(.))`` from the SAME central differences ops.grad
        # uses, so the projection is exact against the discrete divergence/gradient (see the module docstring).
        self._poisson = self._build_poisson()

    def _build_poisson(self):
        """Assemble ``L = sum_axis D_axis(D_axis(.))`` where ``D_axis`` is the exact ops.grad central stencil.

        Returned as ``(rows, cols, vals, n)`` for :func:`ops.sparse_solve`; a small Tikhonov term pins the
        wide-stencil null space so the sparse LU is well posed. Built once (fixed pattern; not the latent).

        The COO pattern for each ``D_axis`` comes from one vectorized broadcast over the grid -- a `meshgrid`
        of node indices, an interior mask for the two stencil offsets, and flat-index arithmetic -- with no
        Python loop over the ``n**3`` nodes, so this scales to survey-size 3-D grids (see ``c6_scale_test.py``).
        """
        import scipy.sparse as sp
        import torch

        n, h, N = self.n, self.h, self.n**3

        def diff_matrix(axis):
            # ops.grad along ``axis``: out[mid] = (a[+1] - a[-1]) / (2h); the two edge slabs are left at zero.
            ii, jj, kk = np.meshgrid(np.arange(n), np.arange(n), np.arange(n), indexing="ij")
            coord = (ii, jj, kk)[axis]
            interior = (coord != 0) & (coord != n - 1)

            def flat(a, b, c):
                return (a * n + b) * n + c

            r = flat(ii, jj, kk)[interior]
            hi = [ii, jj, kk]
            lo = [ii, jj, kk]
            hi = [hi[d] + 1 if d == axis else hi[d] for d in range(3)]
            lo = [lo[d] - 1 if d == axis else lo[d] for d in range(3)]
            c_hi = flat(*hi)[interior]
            c_lo = flat(*lo)[interior]

            rows = np.concatenate([r, r])
            cols = np.concatenate([c_hi, c_lo])
            vals = np.concatenate([np.full(r.size, 1.0 / (2.0 * h)), np.full(r.size, -1.0 / (2.0 * h))])
            return sp.csr_matrix((vals, (rows, cols)), shape=(N, N))

        L = sum(D @ D for D in (diff_matrix(a) for a in range(3)))
        L = (L + self._pressure_reg * sp.identity(N)).tocoo()
        rows = torch.as_tensor(L.row, dtype=torch.long)
        cols = torch.as_tensor(L.col, dtype=torch.long)
        vals = torch.as_tensor(L.data, dtype=torch.float64)
        return rows, cols, vals, N

    def _interior_mask(self, ops):
        return ops.tensor(self._mask)

    def _lap(self, a, ops):
        """Componentwise diffusion Laplacian via the 3-D 7-point stencil (interior; zero on the walls)."""
        n, h = self.n, self.h
        A = a.reshape(n, n, n)
        out = ops.zeros(n, n, n)
        out[1:-1, 1:-1, 1:-1] = (
            A[2:, 1:-1, 1:-1]
            + A[:-2, 1:-1, 1:-1]
            + A[1:-1, 2:, 1:-1]
            + A[1:-1, :-2, 1:-1]
            + A[1:-1, 1:-1, 2:]
            + A[1:-1, 1:-1, :-2]
            - 6.0 * A[1:-1, 1:-1, 1:-1]
        ) / h**2
        return out.reshape(-1)

    def velocity(self, state):
        """The velocity triple ``(u, v, w)`` (three flat length-``n**3`` tensors) held by ``state``."""
        return state[0], state[1], state[2]

    def u(self, state):
        return state[0]

    def v(self, state):
        return state[1]

    def w(self, state):
        return state[2]

    def divergence(self, state, ops):
        """The discrete divergence ``du/dx + dv/dy + dw/dz`` (flat) -- zero in the incompressible limit."""
        u, v, w = self.velocity(state)
        return (
            ops.grad(u, self.shape, 0, spacing=self.h)
            + ops.grad(v, self.shape, 1, spacing=self.h)
            + ops.grad(w, self.shape, 2, spacing=self.h)
        )

    def _advect(self, u, v, w, f, ops):
        """The advective term ``(u . grad) f = u f_x + v f_y + w f_z`` for one velocity component ``f``."""
        return (
            u * ops.grad(f, self.shape, 0, spacing=self.h)
            + v * ops.grad(f, self.shape, 1, spacing=self.h)
            + w * ops.grad(f, self.shape, 2, spacing=self.h)
        )

    def pressure(self, ustar, ops):
        """Solve the pressure Poisson ``laplacian(p) = div(u*) / dt`` with the consistent ops.grad operator.

        With ``L = div(grad(.))`` assembled from the same central differences, the projection
        ``u_{n+1} = u* - dt*grad(p)`` gives ``div(u_{n+1}) = div(u*) - dt*L p``, so setting ``L p = div(u*)/dt``
        makes it vanish (to the solver tolerance). Differentiable through :func:`ops.sparse_solve`."""
        rows, cols, vals, n = self._poisson
        div_ustar = self.divergence(ustar, ops)
        return ops.sparse_solve(rows, cols, vals, n, div_ustar / self.dt)

    def step(self, state, ops):
        """Advance the velocity one time step: advect + diffuse, then project onto divergence-free."""
        mask = self._interior_mask(ops)
        u, v, w = self.velocity(state)
        # u* = u + dt*(-(u.grad)u + nu*lap u), per component; no-slip walls held by the mask.
        us = u + self.dt * (-self._advect(u, v, w, u, ops) + self.nu * self._lap(u, ops))
        vs = v + self.dt * (-self._advect(u, v, w, v, ops) + self.nu * self._lap(v, ops))
        ws = w + self.dt * (-self._advect(u, v, w, w, ops) + self.nu * self._lap(w, ops))
        ustar = (us * mask, vs * mask, ws * mask)
        # pressure projection: laplacian(p) = div(u*)/dt, then u_{n+1} = u* - dt*grad(p).
        p = self.pressure(ustar, ops)
        un = ustar[0] - self.dt * ops.grad(p, self.shape, 0, spacing=self.h)
        vn = ustar[1] - self.dt * ops.grad(p, self.shape, 1, spacing=self.h)
        wn = ustar[2] - self.dt * ops.grad(p, self.shape, 2, spacing=self.h)
        return (un * mask, vn * mask, wn * mask)
