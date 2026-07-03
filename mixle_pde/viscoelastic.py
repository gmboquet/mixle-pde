"""1D time-domain constant-Q viscoacoustic wave solver (GSLS memory-variable formulation).

Seismic Q-FWI and ultrasonic NDE need a wave that loses energy the way real rock and tissue do: with a
quality factor Q that is nearly *frequency independent* over the recording band. A single relaxation
mechanism (one standard linear solid) gives a Q(omega) sharply peaked at one frequency, which is wrong. The
generalized standard linear solid (GSLS, a.k.a. generalized Maxwell / generalized Zener body) sums L
relaxation mechanisms whose relaxation frequencies are spread across the band, and fits their weights so the
combined 1/Q(omega) is flat -- the constant-Q model of Emmerich & Korn (1987) and Blanch, Robertsson &
Symes (1995).

The medium is integrated as the first-order velocity-stress system with L memory variables r_l:

    dv/dt      = (1/rho) d(sigma)/dx
    dr_l/dt    = -omega_l (r_l + M_u a_l e),     e = dv/dx  (the strain rate)
    dsigma/dt  = M_u e + sum_l r_l

with M_u the unrelaxed modulus. Each memory variable relaxes at its own frequency omega_l; the stress is the
instantaneous elastic response M_u e minus the accumulated relaxation -sum r_l, so a step in strain rate
decays over the L relaxation times rather than instantly, which is exactly viscoelastic loss. The weights
a_l come from the tau-method least-squares fit :func:`tau_fit` of a target Q0 over the band, and the exact
GSLS quality factor is :func:`q_of_omega`.

The stepper is differentiable through the ``ops`` namespace in the medium parameters (velocity ``c``,
density ``rho``, and the relaxation weights ``a_l``), so a Q-FWI observation is a
:class:`~mixle_pde.inverse.Differential` forward like the acoustic wave: integrate this stepper, record the
stress (pressure) at receivers, and the gradient w.r.t. the velocity and Q model is the attenuation-aware
sensitivity.
"""

from __future__ import annotations

import numpy as np

__all__ = ["tau_fit", "q_of_omega", "ViscoacousticWave1D"]


def tau_fit(q0, fmin, fmax, n_mech, *, n_target=40):
    """Tau-method least-squares fit of a constant target Q0 over ``[fmin, fmax]`` (Emmerich-Korn / Blanch).

    Places ``n_mech`` relaxation frequencies ``omega_l`` log-spaced across the band and solves the linear
    least-squares problem for the weights ``a_l`` so that the GSLS loss response
    ``sum_l a_l omega omega_l / (omega_l^2 + omega^2)`` matches ``1/Q0`` at ``n_target`` frequencies spanning
    the band. Returns ``(a, omega_l)`` as numpy arrays. With ``n_mech >= 3`` the resulting ``q_of_omega`` is
    flat to a few percent over the band; with ``n_mech = 1`` it is the peaked single-SLS response.
    """
    q0 = float(q0)
    omega_l = 2.0 * np.pi * np.logspace(np.log10(fmin), np.log10(fmax), int(n_mech))
    w = 2.0 * np.pi * np.logspace(np.log10(fmin), np.log10(fmax), int(n_target))
    kernel = (w[:, None] * omega_l[None, :]) / (omega_l[None, :] ** 2 + w[:, None] ** 2)
    a, *_ = np.linalg.lstsq(kernel, np.full(w.shape, 1.0 / q0), rcond=None)
    return a, omega_l


def q_of_omega(a, omega_l, omega):
    """Exact GSLS/generalized-Zener quality factor ``Q(omega)`` for weights ``a_l`` at frequencies ``omega_l``.

    ``Q = (1 + sum_l a_l omega^2 / (omega_l^2 + omega^2)) / (sum_l a_l omega omega_l / (omega_l^2 + omega^2))``,
    the ratio of the real (stored) to the imaginary (loss) part of the complex modulus. Accepts scalar or
    array ``omega``.
    """
    a = np.asarray(a, dtype=float)
    omega_l = np.asarray(omega_l, dtype=float)
    omega = np.asarray(omega, dtype=float)
    denom = omega_l[..., :] ** 2 + omega[..., None] ** 2
    num = np.sum(a * omega[..., None] * omega_l / denom, axis=-1)
    real = 1.0 + np.sum(a * omega[..., None] ** 2 / denom, axis=-1)
    return real / num


class ViscoacousticWave1D:
    """A differentiable 1D constant-Q viscoacoustic stepper (velocity-stress leapfrog + GSLS memory).

    ``ViscoacousticWave1D(nx, dt=..., spacing=..., c=..., rho=..., q0=..., band=(fmin, fmax), n_mech=3)``
    builds the forward on ``nx`` nodes. ``c`` is the (relaxed) phase velocity and ``rho`` the density; the
    unrelaxed modulus is ``M_u = rho c^2 (1 + sum a_l)`` with the relaxation weights from :func:`tau_fit`.
    The state packs the particle velocity ``v``, the stress ``sigma`` (pressure), and the ``n_mech`` memory
    variables (``pack``/``unpack``); ``step(state, ops, source=...)`` advances one leapfrog step with a
    semi-implicit memory-variable update (unconditionally stable in the relaxation term).

    The relaxation weights ``a`` are stored on the instance; pass them as an ``ops`` tensor via ``a_override``
    to ``step`` (or set :attr:`a`) to make Q differentiable. The pressure trace at receivers is
    :meth:`stress`. Courant limit ``dt <= spacing / c``.
    """

    def __init__(
        self,
        nx: int,
        *,
        dt: float,
        spacing: float,
        c: float = 2000.0,
        rho: float = 1000.0,
        q0: float = 30.0,
        band: tuple[float, float] = (10.0, 60.0),
        n_mech: int = 3,
        absorb_width: int = 0,
        absorb_strength: float = 2.0,
    ):
        self.nx = int(nx)
        self.dt = float(dt)
        self.h = float(spacing)
        self.c = float(c)
        self.rho = float(rho)
        self.q0 = float(q0)
        self.band = (float(band[0]), float(band[1]))
        self.n_mech = int(n_mech)

        a, omega_l = tau_fit(q0, band[0], band[1], n_mech)
        self.a = a
        self.omega_l = omega_l
        # relaxed modulus rho c^2; the unrelaxed modulus multiplies the instantaneous strain rate
        self.m_relaxed = rho * c**2
        self.m_unrelaxed = self.m_relaxed * (1.0 + float(np.sum(a)))
        self._gamma = self._build_sponge(absorb_width, absorb_strength)

    def _build_sponge(self, width, strength):
        nx = self.nx
        gamma = np.zeros(nx)
        if width > 0:
            idx = np.arange(nx)
            d = np.minimum(idx, nx - 1 - idx)
            gamma = float(strength) * np.where(d < width, (1.0 - d / width) ** 2, 0.0)
        return gamma

    # ---- state packing -------------------------------------------------------------------------------
    def pack(self, v, sigma, r):
        """Pack velocity ``v`` (nx,), stress ``sigma`` (nx,) and memory ``r`` (n_mech, nx) into the state."""
        import torch

        parts = [torch.as_tensor(v).reshape(-1), torch.as_tensor(sigma).reshape(-1)]
        parts += [torch.as_tensor(r).reshape(self.n_mech, self.nx)[i] for i in range(self.n_mech)]
        return torch.cat(parts)

    def zeros(self, ops):
        """A zero state (velocity, stress, all memory variables)."""
        return ops.zeros((2 + self.n_mech) * self.nx)

    def unpack(self, state):
        """Split the state into ``(v, sigma, r)`` with ``r`` a list of ``n_mech`` flat tensors."""
        nx = self.nx
        v = state[:nx]
        sigma = state[nx : 2 * nx]
        r = [state[(2 + i) * nx : (3 + i) * nx] for i in range(self.n_mech)]
        return v, sigma, r

    def stress(self, state):
        """The stress (pressure) field ``sigma`` from a packed state."""
        return state[self.nx : 2 * self.nx]

    def velocity(self, state):
        """The particle-velocity field ``v`` from a packed state."""
        return state[: self.nx]

    # ---- one-sided differences (staggered velocity-stress) -------------------------------------------
    @staticmethod
    def _dp(f, h, ops):
        """Forward difference ``(f[i+1] - f[i]) / h`` (zero at the far edge)."""
        out = ops.zeros(f.shape[0])
        out[:-1] = (f[1:] - f[:-1]) / h
        return out

    @staticmethod
    def _dm(f, h, ops):
        """Backward difference ``(f[i] - f[i-1]) / h`` (zero at the near edge)."""
        out = ops.zeros(f.shape[0])
        out[1:] = (f[1:] - f[:-1]) / h
        return out

    # ---- leapfrog step -------------------------------------------------------------------------------
    def step(self, state, ops, *, source=0.0, a_override=None):
        """Advance one velocity-stress leapfrog step of the GSLS viscoacoustic system.

        Updates the velocity from the stress gradient, then the memory variables (semi-implicit) and the
        stress from the strain rate. ``source`` is an additive stress-rate injection (scalar 0, a per-node
        array, or an ``ops`` tensor) applied this step. ``a_override`` replaces the relaxation weights with
        an ``ops`` tensor so gradients flow to Q; the unrelaxed modulus tracks it.
        """
        h, dt, rho = self.h, self.dt, self.rho
        v, sigma, r = self.unpack(state)
        omega_l = self.omega_l
        if a_override is None:
            a = ops.tensor(self.a)
            m_u = ops.tensor(self.m_unrelaxed)
        else:
            a = a_override
            m_u = ops.tensor(self.m_relaxed) * (1.0 + ops.sum(a))

        # velocity update: dv/dt = (1/rho) d(sigma)/dx  (forward diff -> velocity a half node forward)
        v = v + (dt / rho) * self._dp(sigma, h, ops)

        # strain rate e = dv/dx (backward diff, centred at the stress node)
        e = self._dm(v, h, ops)

        # memory-variable update (semi-implicit in the stiff relaxation term):
        #   r_l^{n+1} = (r_l^n - dt omega_l M_u a_l e) / (1 + dt omega_l)
        r_next = []
        r_sum = ops.zeros(self.nx)
        for l in range(self.n_mech):
            wl = float(omega_l[l])
            r_new = (r[l] - dt * wl * m_u * a[l] * e) / (1.0 + dt * wl)
            r_next.append(r_new)
            r_sum = r_sum + r_new

        # stress update: dsigma/dt = M_u e + sum_l r_l  (+ source)
        sigma = sigma + dt * (m_u * e + r_sum)
        if isinstance(source, np.ndarray):
            sigma = sigma + dt * ops.tensor(source.reshape(-1))
        elif not (isinstance(source, float | int) and source == 0.0):
            sigma = sigma + dt * source  # already an ops tensor

        # absorbing sponge on the velocity near the edges
        if self._gamma.any():
            damp = 1.0 / (1.0 + dt * ops.tensor(self._gamma))
            v = v * damp

        return ops.cat([v, sigma, *r_next])

    def ricker_source(self, position, f0, amplitude=1.0, *, t0=None):
        """A Ricker-wavelet stress source at node ``position`` with dominant frequency ``f0`` (Hz).

        Returns ``source_at(t)`` giving the per-node stress-rate injection for :meth:`step` at time ``t``
        (a numpy array with the wavelet value at ``position``). ``t0`` defaults to ``1.5 / f0`` so the pulse
        turns on smoothly.
        """
        i = int(round(position))
        t0 = 1.5 / f0 if t0 is None else float(t0)

        def source_at(t):
            arg = (np.pi * f0 * (t - t0)) ** 2
            w = (1.0 - 2.0 * arg) * np.exp(-arg)
            f = np.zeros(self.nx)
            f[i] = float(amplitude) * w
            return f

        return source_at
