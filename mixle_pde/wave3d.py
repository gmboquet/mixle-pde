"""3D acoustic wave equation for volumetric full-waveform-inversion-style inverse problems.

The 3D counterpart of :class:`~mixle_pde.wave.WaveEquation2D`: a turnkey forward for time-domain wave
propagation, ``u_tt = c(x)^2 laplacian(u) + source(t)``, on an ``n x n x n`` grid, solved by an explicit
symplectic (leapfrog) step on the first-order system ``(u, w=u_t)``. Two absorbing-boundary treatments are
available near the six faces:

* ``absorb_width`` / ``absorb_strength`` -- the original exponential-ramp sponge: an artificial ``-gamma * u_t``
  drag added near the boundary. Cheap, but a sponge only damps -- it never perfectly absorbs, so it always
  leaks some energy back into the interior (more so for short, broadband pulses).
* ``pml_width`` / ``pml_profile`` -- a true perfectly-matched layer: a split-field CPML built from complex
  coordinate stretching. ``u`` is additively split into three per-axis components ``u = ux + uy + uz``, each
  obeying its own damped second-order ODE ``d2(u_a)/dt2 + 2*sigma_a*d(u_a)/dt + sigma_a^2*u_a = c^2 d2(u)/da^2``
  with ``sigma_a`` a graded profile that is zero in the interior and rises to a peak over the outer
  ``pml_width`` nodes. Each split component carries its own auxiliary "memory" variable (its own velocity
  ``d(u_a)/dt``), which is what makes this convolutional rather than a bare damping term: the outgoing wave is
  absorbed as it enters the layer rather than merely slowed down, so the reflected energy is far smaller than
  the sponge's for the same layer width (see ``c6_scale_test.py``). Interior physics is unchanged either way
  (``sigma_a = 0`` there recovers exactly ``u_tt = c^2 laplacian(u)``). Built on the checkpointed time
  integrator, so the gradient w.r.t. the velocity field -- the full-waveform-inversion sensitivity -- is
  available at feasible memory.

The inverse problem -- recover the velocity field ``c(x)`` (or a localized perturbation) from waveforms
recorded at a few receivers -- is a :class:`~mixle.ppl.Differential` observation that integrates this stepper
and records the displacement at the receivers; fit it with ``how='gauss_newton'``.
"""

from __future__ import annotations

import numpy as np


class WaveEquation3D:
    """A differentiable 3D acoustic wave-equation stepper with a sponge or a true split-field CPML boundary.

    ``WaveEquation3D(n, dt=..., spacing=..., absorb_width=..., absorb_strength=..., pml_width=...,
    pml_profile=...)`` builds the forward on an ``n x n x n`` grid. With ``pml_width == 0`` (the default) the
    state is the packed ``(u, w=u_t)`` and the boundary is the original exponential sponge (or none, if
    ``absorb_width`` is also 0). With ``pml_width > 0`` the boundary is a split-field CPML instead: the state
    packs three per-axis ``(u_a, w_a = d(u_a)/dt)`` components, and ``displacement`` recombines them into the
    physical field ``u = ux + uy + uz``. Either way, ``step(state, c2, ops, source=...)`` advances one
    leapfrog step given the squared-velocity field ``c2`` (a driver or a fixed array) and an optional per-node
    source term for that step; callers never need to know which boundary treatment is active.
    """

    def __init__(
        self,
        n: int,
        *,
        dt: float,
        spacing: float | None = None,
        absorb_width: int = 0,
        absorb_strength: float = 2.0,
        pml_width: int = 0,
        pml_profile: str = "polynomial",
    ):
        self.n = int(n)
        self.dt = float(dt)
        self.h = float(spacing) if spacing is not None else 1.0 / (n - 1)
        self._gamma = self._build_sponge(absorb_width, absorb_strength)
        self._pml_width = int(pml_width)
        self._pml_profile = pml_profile
        self._pml = self._build_cpml(self._pml_width, pml_profile, absorb_strength) if self._pml_width > 0 else None

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

    def _build_cpml(self, width, profile, strength):
        """Per-axis damping profile ``sigma_a`` (flat, length ``n**3``) for the split-field CPML.

        Zero in the interior, rising over the outer ``width`` nodes to a peak set by ``strength`` (reusing the
        sponge's strength knob -- there is no separate PML-only parameter in the frozen constructor). The
        ``strength * 3 / (2 * width * h)`` peak follows the same scaling as the 2-D reference PML in
        :mod:`mixle_pde.wave_pml` (with a nominal unit wave speed folded into ``strength``).
        """
        n, h = self.n, self.h
        width = int(width)
        idx = np.arange(n)
        d = np.minimum(idx, n - 1 - idx)  # distance (in nodes) to the nearest edge
        if profile == "polynomial":
            ramp = np.where(d < width, (1.0 - d / width) ** 2, 0.0)
        elif profile == "linear":
            ramp = np.where(d < width, (1.0 - d / width), 0.0)
        else:
            raise ValueError(f"unknown pml_profile {profile!r}; use 'polynomial' or 'linear'.")
        sigma_max = float(strength) * 3.0 / (2.0 * width * h)
        sigma_1d = sigma_max * ramp
        sx = np.broadcast_to(sigma_1d[:, None, None], (n, n, n)).ravel().copy()
        sy = np.broadcast_to(sigma_1d[None, :, None], (n, n, n)).ravel().copy()
        sz = np.broadcast_to(sigma_1d[None, None, :], (n, n, n)).ravel().copy()
        return sx, sy, sz

    def pack(self, u, w):
        """Pack displacement ``u`` and velocity ``w = u_t`` into the integrator state.

        With the CPML active, ``u``/``w`` are split evenly across the three axis components (their sum
        reproduces ``u``/``w`` exactly since the interior damping is zero, so the split point is arbitrary)."""
        import torch

        u = torch.as_tensor(u)
        w = torch.as_tensor(w)
        if self._pml is None:
            return torch.cat([u, w])
        u3, w3 = u / 3.0, w / 3.0
        return torch.cat([u3, w3, u3, w3, u3, w3])

    def displacement(self, state):
        """The displacement field ``u`` from a packed state (``ux + uy + uz`` when the CPML is active)."""
        nnn = self.n**3
        if self._pml is None:
            return state[:nnn]
        return state[0:nnn] + state[2 * nnn : 3 * nnn] + state[4 * nnn : 5 * nnn]

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

    def _lap_axis(self, u, axis, ops):
        """Second difference of ``u`` along one axis only (interior along that axis; zero at its two faces)."""
        n, h = self.n, self.h
        A = u.reshape(n, n, n)
        out = ops.zeros(n, n, n)
        lo, hi, mid = [slice(None)] * 3, [slice(None)] * 3, [slice(None)] * 3
        lo[axis] = slice(0, n - 2)
        hi[axis] = slice(2, n)
        mid[axis] = slice(1, n - 1)
        out[tuple(mid)] = (A[tuple(hi)] + A[tuple(lo)] - 2.0 * A[tuple(mid)]) / h**2
        return out.reshape(-1)

    def step(self, state, c2, ops, source=0.0):
        """Advance the state one leapfrog step.

        Without a CPML: ``u_tt = c2 * lap(u) + source - gamma * u_t`` (the exponential sponge, or none).
        With the CPML (``pml_width > 0``): the split-field update, one damped oscillator per axis component
        driven by the directional second derivative of the *recombined* field, ``source`` split evenly across
        the three axes so the total field still receives the full source.
        """
        nnn = self.n**3
        if self._pml is None:
            u, w = state[:nnn], state[nnn:]
            gamma = ops.tensor(self._gamma)
            u_next = u + self.dt * w
            accel = c2 * self._lap(u_next, ops) + source - gamma * w
            w_next = w + self.dt * accel
            return ops.cat([u_next, w_next])

        sx, sy, sz = (ops.tensor(s) for s in self._pml)
        ux, wx, uy, wy, uz, wz = (state[i * nnn : (i + 1) * nnn] for i in range(6))
        ux_next = ux + self.dt * wx
        uy_next = uy + self.dt * wy
        uz_next = uz + self.dt * wz
        u_next = ux_next + uy_next + uz_next
        src3 = source / 3.0
        accel_x = c2 * self._lap_axis(u_next, 0, ops) - 2.0 * sx * wx - sx * sx * ux + src3
        accel_y = c2 * self._lap_axis(u_next, 1, ops) - 2.0 * sy * wy - sy * sy * uy + src3
        accel_z = c2 * self._lap_axis(u_next, 2, ops) - 2.0 * sz * wz - sz * sz * uz + src3
        wx_next = wx + self.dt * accel_x
        wy_next = wy + self.dt * accel_y
        wz_next = wz + self.dt * accel_z
        return ops.cat([ux_next, wx_next, uy_next, wy_next, uz_next, wz_next])
