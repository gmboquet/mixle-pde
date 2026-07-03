"""3-D source-free Maxwell curl equations on a Yee (staggered) grid, explicit leapfrog (FDTD).

The classic finite-difference time-domain scheme (Yee, 1966) for electromagnetics: the two first-order
curl equations

    dH/dt = -(1/mu)  curl E     (Faraday)
    dE/dt =  (1/eps) curl H     (Ampere, source free)

integrated by an explicit leapfrog in which E and H live a half time-step apart. The six field components
are staggered on the Yee cell -- E on the cell edges, H on the cell faces -- so that each curl is a
centred difference of the dual field at the point where the update lives. That staggering is what makes the
scheme charge-conserving: ``div(mu H)`` and ``div(eps E)`` are preserved to machine precision for all time,
so no spurious magnetic monopoles appear. PEC (perfect-electric-conductor) walls -- tangential ``E = 0`` on
the boundary -- turn the box into a resonant cavity whose modes ring at the analytical frequencies.

The stepper matches the :class:`~mixle_pde.wave.WaveEquation2D` API: ``Maxwell3D(n, dt=..., spacing=...)``
builds the forward on an ``n x n x n`` grid; ``pack``/``unpack`` move between the flat integrator state and
the six field arrays; ``step(state, ops)`` advances one full (H then E) leapfrog step. It is differentiable
through ``ops`` (slice arithmetic on autograd tensors), so a cavity/scattering observation is a
:class:`~mixle_pde.inverse.Differential` forward like the acoustic wave.
"""

from __future__ import annotations

import numpy as np


class Maxwell3D:
    """A differentiable 3-D FDTD Maxwell stepper on a Yee grid with PEC cavity walls.

    ``Maxwell3D(n, dt=..., spacing=..., eps=1.0, mu=1.0)`` builds the forward on an ``n x n x n`` cell grid.
    The state is the six field components packed in the order ``(Ex, Ey, Ez, Hx, Hy, Hz)``
    (``pack``/``unpack``); ``step(state, ops)`` advances one leapfrog step (update H a half step, then E a
    half step). The speed of light is ``c = 1/sqrt(eps*mu)``; the Courant limit for a 3-D cube is
    ``dt <= spacing / (c*sqrt(3))``.

    Yee staggering (component -> half-integer offsets from the ``(i,j,k)`` node, in units of ``spacing``):
    ``Ex@(i+1/2,j,k)``, ``Ey@(i,j+1/2,k)``, ``Ez@(i,j,k+1/2)``; ``Hx@(i,j+1/2,k+1/2)``,
    ``Hy@(i+1/2,j,k+1/2)``, ``Hz@(i+1/2,j+1/2,k)``. Each curl term is then a one-sided difference of the dual
    field between the two staggered points straddling the update location, i.e. a centred difference there.
    """

    def __init__(
        self,
        n: int,
        *,
        dt: float,
        spacing: float | None = None,
        eps: float = 1.0,
        mu: float = 1.0,
    ):
        self.n = int(n)
        self.dt = float(dt)
        self.h = float(spacing) if spacing is not None else 1.0 / (n - 1)
        self.eps = float(eps)
        self.mu = float(mu)
        self.c = 1.0 / np.sqrt(self.eps * self.mu)
        self._nc = self.n**3  # per-component length

    # ---- state packing -------------------------------------------------------------------------------
    def pack(self, Ex, Ey, Ez, Hx, Hy, Hz):
        """Pack the six field arrays (each ``n x n x n`` or flat) into the integrator state."""
        import torch

        parts = [torch.as_tensor(f).reshape(-1) for f in (Ex, Ey, Ez, Hx, Hy, Hz)]
        return torch.cat(parts)

    def unpack(self, state):
        """Split a packed state into ``(Ex, Ey, Ez, Hx, Hy, Hz)`` as flat length-``n^3`` tensors."""
        nc = self._nc
        return tuple(state[i * nc : (i + 1) * nc] for i in range(6))

    def fields(self, state, ops):
        """Return the six components reshaped to ``(n, n, n)`` grids (a view for accessors / sampling)."""
        n = self.n
        return tuple(c.reshape(n, n, n) for c in self.unpack(state))

    def zeros(self, ops):
        """A zero field state (all six components) as a packed integrator state."""
        return ops.zeros(6 * self._nc)

    # ---- staggered-grid curl --------------------------------------------------------------------------
    # Forward difference d_a f at cell (i..): (f[i+1] - f[i]) / h  along axis a. Used for curl E (H update):
    # H lives a half cell forward of E along the two transverse axes, so the forward diff is centred at H.
    @staticmethod
    def _dp(f, axis, h):
        # forward difference; the top slice (no i+1 neighbour) is left as the shifted-in value = 0 at the
        # far face, which is consistent with the PEC-terminated cavity (tangential E vanishes there).
        lo = [slice(None)] * 3
        hi = [slice(None)] * 3
        lo[axis] = slice(0, -1)
        hi[axis] = slice(1, None)
        out = f * 0.0
        out[tuple(lo)] = (f[tuple(hi)] - f[tuple(lo)]) / h
        return out

    # Backward difference d_a f at cell (i..): (f[i] - f[i-1]) / h. Used for curl H (E update): E lives a
    # half cell backward of H along the transverse axes, so the backward diff is centred at E.
    @staticmethod
    def _dm(f, axis, h):
        lo = [slice(None)] * 3
        hi = [slice(None)] * 3
        lo[axis] = slice(1, None)
        hi[axis] = slice(0, -1)
        out = f * 0.0
        out[tuple(lo)] = (f[tuple(lo)] - f[tuple(hi)]) / h
        return out

    def _curl_E(self, Ex, Ey, Ez, h):
        """curl E evaluated at the H points (forward differences of E)."""
        dp = self._dp
        cx = dp(Ez, 1, h) - dp(Ey, 2, h)  # (curl E)_x = dEz/dy - dEy/dz
        cy = dp(Ex, 2, h) - dp(Ez, 0, h)  # (curl E)_y = dEx/dz - dEz/dx
        cz = dp(Ey, 0, h) - dp(Ex, 1, h)  # (curl E)_z = dEy/dx - dEx/dy
        return cx, cy, cz

    def _curl_H(self, Hx, Hy, Hz, h):
        """curl H evaluated at the E points (backward differences of H)."""
        dm = self._dm
        cx = dm(Hz, 1, h) - dm(Hy, 2, h)  # (curl H)_x = dHz/dy - dHy/dz
        cy = dm(Hx, 2, h) - dm(Hz, 0, h)  # (curl H)_y = dHx/dz - dHz/dx
        cz = dm(Hy, 0, h) - dm(Hx, 1, h)  # (curl H)_z = dHy/dx - dHx/dy
        return cx, cy, cz

    # ---- PEC boundary --------------------------------------------------------------------------------
    def _apply_pec(self, Ex, Ey, Ez):
        """Zero the tangential E on the box faces (perfect electric conductor cavity walls).

        On a face with outward normal along axis ``a`` the two transverse E components are tangential and
        must vanish. Ex is tangential on the y- and z-faces, Ey on the x- and z-faces, Ez on the x- and
        y-faces. Enforced in place on the ``(n,n,n)`` component grids.
        """
        # Ex tangential on y-faces (j=0,n-1) and z-faces (k=0,n-1)
        Ex[:, 0, :] = 0.0
        Ex[:, -1, :] = 0.0
        Ex[:, :, 0] = 0.0
        Ex[:, :, -1] = 0.0
        # Ey tangential on x-faces (i=0,n-1) and z-faces (k=0,n-1)
        Ey[0, :, :] = 0.0
        Ey[-1, :, :] = 0.0
        Ey[:, :, 0] = 0.0
        Ey[:, :, -1] = 0.0
        # Ez tangential on x-faces (i=0,n-1) and y-faces (j=0,n-1)
        Ez[0, :, :] = 0.0
        Ez[-1, :, :] = 0.0
        Ez[:, 0, :] = 0.0
        Ez[:, -1, :] = 0.0
        return Ex, Ey, Ez

    # ---- leapfrog step -------------------------------------------------------------------------------
    def step(self, state, ops, *, pec: bool = True):
        """Advance one full leapfrog step: update H by ``-dt/mu curl E``, then E by ``dt/eps curl H``.

        With ``pec=True`` the tangential E on all six box faces is held at zero (a resonant PEC cavity).
        The H update sees the pre-update E; the E update sees the just-updated H (the standard staggered
        leapfrog ordering).
        """
        n, h, dt = self.n, self.h, self.dt
        Ex, Ey, Ez, Hx, Hy, Hz = (c.reshape(n, n, n) for c in self.unpack(state))

        # Faraday: H^{q+1/2} = H^{q-1/2} - (dt/mu) curl E^q
        cEx, cEy, cEz = self._curl_E(Ex, Ey, Ez, h)
        Hx = Hx - (dt / self.mu) * cEx
        Hy = Hy - (dt / self.mu) * cEy
        Hz = Hz - (dt / self.mu) * cEz

        # Ampere: E^{q+1} = E^q + (dt/eps) curl H^{q+1/2}
        cHx, cHy, cHz = self._curl_H(Hx, Hy, Hz, h)
        Ex = Ex + (dt / self.eps) * cHx
        Ey = Ey + (dt / self.eps) * cHy
        Ez = Ez + (dt / self.eps) * cHz

        if pec:
            Ex, Ey, Ez = self._apply_pec(Ex, Ey, Ez)

        return ops.cat([Ex.reshape(-1), Ey.reshape(-1), Ez.reshape(-1), Hx.reshape(-1), Hy.reshape(-1), Hz.reshape(-1)])

    # ---- diagnostics ---------------------------------------------------------------------------------
    def div_H(self, state, ops):
        """Discrete ``div(mu H)`` on the Yee grid, returned as an ``(n,n,n)`` field.

        The divergence dual to the H field is the difference operator that annihilates the H update's
        ``curl E`` term: since that curl is built from forward differences of E, the exact discrete identity
        ``div(curl E) = 0`` holds only for the forward-difference divergence of each H component along its
        own axis. The Yee update then preserves this exactly -- if ``div(mu H)`` is zero initially it stays
        zero (to rounding) for all time, so no spurious magnetic monopoles appear.
        """
        n, h = self.n, self.h
        _, _, _, Hx, Hy, Hz = (c.reshape(n, n, n) for c in self.unpack(state))
        d = self._dp(Hx, 0, h) + self._dp(Hy, 1, h) + self._dp(Hz, 2, h)
        return self.mu * d
