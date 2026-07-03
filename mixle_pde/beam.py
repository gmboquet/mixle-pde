"""Euler-Bernoulli beam: transverse vibration and static deflection of a slender beam (a 1D 4th-order PDE).

The transverse displacement ``w(x, t)`` of a slender beam obeys the fourth-order Euler-Bernoulli equation

    rho*A * w_tt = -EI * w_xxxx + f(x, t),

where ``EI`` is the flexural rigidity (Young's modulus times the second moment of area), ``rho*A`` the mass
per unit length, and ``f`` a distributed transverse load. Bending resists the fourth spatial derivative, so
the operator is the biharmonic ``w_xxxx`` rather than the wave/diffusion Laplacian -- the same 1D solver
pattern as the rest of this package, one derivative order up.

Two solves share the discretization. The dynamic ``step((w, v), ops, load=...)`` is an explicit leapfrog on
the first-order system ``(w, v=w_t)`` -- the beam analogue of :class:`~mixle_pde.wave.WaveEquation2D`, so the
gradient w.r.t. ``EI`` (or a distributed load) flows through the recorded response. The ``static(load, ops)``
solve assembles the biharmonic-1D operator with boundary conditions and solves ``EI*w_xxxx = f`` as one
sparse linear system, in the manner of the flow / Poisson static solves.

Simply-supported ends (``w = 0`` and ``w_xx = 0`` at both ends) are the case with a clean analytical check:
a uniform load ``q`` gives the textbook center deflection ``5*q*L^4 / (384*EI)``, and the fundamental mode is
``sin(pi x / L)`` with natural frequency ``omega_1 = (pi/L)^2 sqrt(EI / (rho*A))``.
"""

from __future__ import annotations

import numpy as np


class EulerBernoulliBeam:
    """A differentiable Euler-Bernoulli beam solver (static deflection + dynamic transverse vibration).

    ``EulerBernoulliBeam(n, length=..., EI=..., rho=..., A=...)`` builds the forward on a 1D grid of ``n``
    nodes over ``[0, L]``. ``static(load, ops)`` solves the steady ``EI*w_xxxx = f`` sparse linear system;
    ``step((w, v), ops, load=...)`` advances the vibration ``rho*A*w_tt = -EI*w_xxxx + f`` one leapfrog step
    on the packed displacement/velocity state. Only simply-supported boundary conditions are implemented
    (``w = 0`` and ``w_xx = 0`` at both ends), the case with a closed-form benchmark.

    Args:
        n: number of grid nodes.
        length: beam length ``L``.
        EI: flexural rigidity (Young's modulus times second moment of area).
        rho: material density.
        A: cross-sectional area (so ``rho*A`` is the mass per unit length).
        dt: time step for the dynamic ``step`` (unused by ``static``).
    """

    def __init__(
        self,
        n: int,
        *,
        length: float = 1.0,
        EI: float = 1.0,
        rho: float = 1.0,
        A: float = 1.0,
        dt: float = 1e-3,
    ):
        self.n = int(n)
        self.L = float(length)
        self.EI = float(EI)
        self.rho = float(rho)
        self.A = float(A)
        self.dt = float(dt)
        self.h = self.L / (self.n - 1)
        self._biharmonic = self._assemble_biharmonic()

    # ------------------------------------------------------------------ static solve
    def _assemble_biharmonic(self):
        """Assemble the biharmonic operator ``EI/h^4 * D4`` (COO) for simply-supported ends.

        The interior 5-point stencil is ``(w[i-2] - 4w[i-1] + 6w[i] - 4w[i+1] + w[i+2]) / h^4``. The two end
        nodes are pinned (``w = 0``, identity rows). At the near-boundary nodes the ghost value implied by
        ``w_xx = 0`` at a pinned end is ``w[-1] = -w[1]`` (reflection), which folds the ``6`` on the diagonal
        down to ``5`` there. Returns ``(rows, cols, vals, n)`` in the ``ops.sparse_solve`` layout.
        """
        n, h = self.n, self.h
        rows: list[int] = []
        cols: list[int] = []
        vals: list[float] = []

        def add(i, j, v):
            rows.append(i)
            cols.append(j)
            vals.append(v)

        scale = self.EI / h**4
        # pinned ends: identity rows w[0] = w[n-1] = 0
        add(0, 0, 1.0)
        add(n - 1, n - 1, 1.0)
        for i in range(1, n - 1):
            if i == 1:
                # ghost w[-1] = -w[1] from w_xx(0)=0, w[0]=0 -> diagonal 6 folds to 5
                add(i, 1, scale * 5.0)
                add(i, 2, scale * -4.0)
                add(i, 3, scale * 1.0)
            elif i == n - 2:
                add(i, n - 2, scale * 5.0)
                add(i, n - 3, scale * -4.0)
                add(i, n - 4, scale * 1.0)
            else:
                add(i, i - 2, scale * 1.0)
                add(i, i - 1, scale * -4.0)
                add(i, i, scale * 6.0)
                add(i, i + 1, scale * -4.0)
                add(i, i + 2, scale * 1.0)
        return (np.array(rows), np.array(cols), np.array(vals, dtype=float), n)

    def static(self, load, ops):
        """Solve the static beam ``EI*w_xxxx = f`` under a distributed ``load`` (scalar or per-node array).

        Returns the deflection ``w`` at the ``n`` nodes. The pinned end rows read ``w = 0`` from the RHS, so
        the load at the boundary nodes is ignored (as it should be for a simple support)."""
        rows, cols, vals, n = self._biharmonic
        f = np.full(n, float(load)) if np.isscalar(load) else np.asarray(load, dtype=float).ravel().copy()
        b = ops.tensor(f).clone()
        b[0] = 0.0
        b[n - 1] = 0.0  # pinned ends: identity rows carry w = 0
        rows_t = ops.tensor(rows).long()
        cols_t = ops.tensor(cols).long()
        vals_t = ops.tensor(vals)
        return ops.sparse_solve(rows_t, cols_t, vals_t, n, b)

    # ------------------------------------------------------------------ dynamic step
    def _w_xxxx(self, w, ops):
        """The fourth derivative ``w_xxxx`` via the 5-point stencil, with simply-supported ends.

        Interior nodes use ``(w[i-2] - 4w[i-1] + 6w[i] - 4w[i+1] + w[i+2]) / h^4``; the near-boundary nodes
        fold in the ``w[-1] = -w[1]`` reflection from ``w_xx = 0`` at a pinned end; the end nodes are pinned
        (``w = 0``) so their bending contribution is zero."""
        n, h = self.n, self.h
        out = ops.zeros(n)
        # interior nodes 2 .. n-3
        out[2 : n - 2] = w[0 : n - 4] - 4 * w[1 : n - 3] + 6 * w[2 : n - 2] - 4 * w[3 : n - 1] + w[4:n]
        # near-boundary nodes with the reflected ghost (diagonal 6 -> 5)
        out[1] = 5 * w[1] - 4 * w[2] + w[3]
        out[n - 2] = 5 * w[n - 2] - 4 * w[n - 3] + w[n - 4]
        return out / h**4

    def step(self, state, ops, load=0.0):
        """Advance ``(w, v=w_t)`` one leapfrog step under ``rho*A*w_tt = -EI*w_xxxx + load``.

        ``state`` is the packed ``(w, v)`` (see :meth:`pack`); ``load`` is a scalar or per-node forcing for
        this step. The end nodes are held at ``w = 0`` (simple support), so the state stays pinned there."""
        n = self.n
        w, v = state[:n], state[n:]
        w_next = w + self.dt * v
        # enforce the pinned ends on the displacement
        pin = ops.tensor(np.array([0.0 if (i == 0 or i == n - 1) else 1.0 for i in range(n)]))
        w_next = w_next * pin
        accel = (-self.EI * self._w_xxxx(w_next, ops) + load) / (self.rho * self.A)
        v_next = (v + self.dt * accel) * pin
        return ops.cat([w_next, v_next])

    def pack(self, w, v):
        """Pack displacement ``w`` and velocity ``v = w_t`` into the leapfrog state."""
        import torch

        return torch.cat([torch.as_tensor(w), torch.as_tensor(v)])

    def displacement(self, state):
        """The displacement field ``w`` from a packed state."""
        return state[: self.n]
