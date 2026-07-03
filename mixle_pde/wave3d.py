"""3D acoustic wave equation for volumetric full-waveform-inversion-style inverse problems.

The 3D counterpart of :class:`~mixle_pde.wave.WaveEquation2D`: a turnkey forward for time-domain wave
propagation, ``u_tt = c(x)^2 laplacian(u) + source(t)``, on an ``n x n x n`` grid, solved by an explicit
symplectic (leapfrog) step on the first-order system ``(u, w=u_t)``. A 3D absorbing sponge layer near the
six faces damps outgoing waves so the finite domain does not reflect them back (the practical stand-in for a
perfectly-matched layer). Built on the checkpointed time integrator, so the gradient w.r.t. the velocity
field -- the full-waveform-inversion sensitivity -- is available at feasible memory.

The inverse problem -- recover the velocity field ``c(x)`` (or a localized perturbation) from waveforms
recorded at a few receivers -- is a :class:`~mixle.ppl.Differential` observation that integrates this stepper
and records the displacement at the receivers; fit it with ``how='gauss_newton'``.
"""

from __future__ import annotations

import numpy as np


class WaveEquation3D:
    """A differentiable 3D acoustic wave-equation stepper with an absorbing sponge boundary.

    ``WaveEquation3D(n, dt=..., spacing=..., absorb_width=..., absorb_strength=...)`` builds the forward on
    an ``n x n x n`` grid. The state is the packed ``(u, w=u_t)`` (``pack``/``displacement``); ``step(state,
    c2, ops, source=...)`` advances one leapfrog step given the squared-velocity field ``c2`` (a driver or
    a fixed array) and an optional per-node source term for that step.
    """

    def __init__(
        self,
        n: int,
        *,
        dt: float,
        spacing: float | None = None,
        absorb_width: int = 0,
        absorb_strength: float = 2.0,
    ):
        self.n = int(n)
        self.dt = float(dt)
        self.h = float(spacing) if spacing is not None else 1.0 / (n - 1)
        self._gamma = self._build_sponge(absorb_width, absorb_strength)

    def _build_sponge(self, width, strength):
        n = self.n
        gamma = np.zeros((n, n, n))
        if width > 0:
            idx = np.arange(n)
            d = np.minimum(idx, n - 1 - idx)  # distance (in nodes) to the nearest edge
            ramp = np.where(d < width, (1.0 - d / width) ** 2, 0.0)
            # union of the three-axis ramps: damping near any of the six faces
            gx = np.broadcast_to(ramp[:, None, None], (n, n, n))
            gy = np.broadcast_to(ramp[None, :, None], (n, n, n))
            gz = np.broadcast_to(ramp[None, None, :], (n, n, n))
            gamma = float(strength) * np.maximum(np.maximum(gx, gy), gz)
        return gamma.ravel()

    def pack(self, u, w):
        """Pack displacement ``u`` and velocity ``w = u_t`` into the integrator state."""
        import torch

        return torch.cat([torch.as_tensor(u), torch.as_tensor(w)])

    def displacement(self, state):
        """The displacement field ``u`` from a packed state."""
        return state[: self.n**3]

    def _lap(self, u, ops):
        n, h = self.n, self.h
        A = u.reshape(n, n, n)
        out = ops.zeros(n, n, n)
        out[1:-1, 1:-1, 1:-1] = (
            A[2:, 1:-1, 1:-1]
            + A[:-2, 1:-1, 1:-1]
            + A[1:-1, 2:, 1:-1]
            + A[1:-1, :-2, 1:-1]
            + A[1:-1, 1:-1, 2:]
            + A[1:-1, 1:-1, :-2]
            - 6 * A[1:-1, 1:-1, 1:-1]
        ) / h**2
        return out.reshape(-1)

    def step(self, state, c2, ops, source=0.0):
        """Advance ``(u, w)`` one leapfrog step under ``u_tt = c2 * lap(u) + source - gamma * u_t``."""
        nnn = self.n**3
        u, w = state[:nnn], state[nnn:]
        gamma = ops.tensor(self._gamma)
        u_next = u + self.dt * w
        accel = c2 * self._lap(u_next, ops) + source - gamma * w
        w_next = w + self.dt * accel
        return ops.cat([u_next, w_next])
