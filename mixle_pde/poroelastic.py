"""1D Biot poroelastic wave equation on a staggered grid, explicit velocity-stress leapfrog.

The physics of wave propagation through a fluid-saturated porous rock (seismics in a reservoir, ultrasonics
in a core plug). Biot (1956) couples the solid frame to the pore fluid, so a single P excitation splits into
*two* compressional waves: a fast P wave in which the solid and fluid move in phase, and a slow (Biot) P wave
in which they move out of phase. The slow wave is diffusive at seismic frequencies -- the viscous drag of the
fluid relative to the frame overwhelms its inertia -- and becomes a true propagating mode only above the Biot
transition frequency.

The 1D low-frequency Biot system in the velocity-stress form, with solid particle velocity ``v`` and relative
fluid velocity ``q = phi (v_fluid - v_solid)`` (the Darcy flux rate), total stress ``sigma`` and pore-fluid
pressure ``pf``:

    rho   dv/dt + rho_f dq/dt              = d sigma / dx
    rho_f dv/dt + m    dq/dt + (eta/k) q   = -d pf / dx
    d sigma/dt =  H dv/dx + C dq/dx
    d pf/dt    = -(C dv/dx + M dq/dx)

with the Biot-Gassmann poroelastic moduli: the drained frame gives ``K_dry, mu``; Gassmann fluid substitution
gives the undrained modulus ``K_sat = K_dry + alpha^2 M`` with Biot coefficient ``alpha = 1 - K_dry/K_s`` and
Biot modulus ``M = 1 / (phi/K_f + (alpha - phi)/K_s)``; the P-wave modulus is ``H = K_sat + 4/3 mu`` and the
solid-fluid coupling modulus is ``C = alpha M``. The bulk density is ``rho = (1-phi) rho_s + phi rho_f`` and
the effective fluid-inertia coefficient is ``m = tortuosity rho_f / phi``. The fast-P phase velocity in the
low-frequency undrained limit is the Biot-Gassmann value ``vp = sqrt(H / rho)``.

Staggering follows the velocity-stress scheme of :class:`~mixle_pde.elastic.ElasticWave3D`, reduced to one
dimension: the velocities ``v, q`` live at the integer nodes ``i`` and the stress/pressure ``sigma, pf`` at
the half nodes ``i + 1/2``, so each spatial derivative in the update is a centred one-sided difference at the
point where the update lives. The stress and pressure are advanced a half step (forward difference of the
velocities), then the two velocities are advanced a half step (backward difference of the stress/pressure).

The viscous drag ``(eta/k) q`` is stiff: at low permeability ``eta/k`` is enormous compared with the inertia,
so the drag is integrated *implicitly* while everything else stays explicit. Writing the coupled velocity
update as ``(T + dt D) V_next = T V_cur + dt F`` with the constant 2x2 inertia matrix
``T = [[rho, rho_f], [rho_f, m]]`` and drag ``D = diag(0, eta/k)``, the 2x2 solve is done once per step. This
gives the fast wave the standard explicit leapfrog and the slow wave its correct diffusive decay, without a
vanishing time step. The Courant limit is ``dt <= spacing / c_max`` where ``c_max`` is the larger root of the
undrained characteristic system ``det(H_matrix - c^2 T) = 0`` (slightly above ``vp``).

The whole stepper is differentiable through the ``ops`` namespace (slice arithmetic on autograd tensors), so a
poroelastic waveform is a :class:`~mixle_pde.inverse.Differential` forward like the acoustic and elastic waves
and the gradient of a recorded trace w.r.t. the rock properties is available for inversion.
"""

from __future__ import annotations

import numpy as np

# packed state: solid velocity, relative fluid velocity, total stress, pore pressure
_COMPONENTS = ("v", "q", "sigma", "pf")


def gassmann_moduli(
    *,
    k_solid: float,
    k_fluid: float,
    k_dry: float,
    mu: float,
    phi: float,
) -> dict[str, float]:
    """The Biot-Gassmann poroelastic moduli from the drained-frame and constituent properties.

    Returns a dict with the Biot coefficient ``alpha``, Biot modulus ``M``, undrained (saturated) bulk
    modulus ``k_sat``, undrained P-wave modulus ``H = k_sat + 4/3 mu`` and coupling modulus ``C = alpha M``.
    ``k_solid`` is the mineral grain modulus, ``k_fluid`` the pore-fluid modulus, ``k_dry`` and ``mu`` the
    drained frame moduli, ``phi`` the porosity.
    """
    alpha = 1.0 - k_dry / k_solid
    M = 1.0 / (phi / k_fluid + (alpha - phi) / k_solid)
    k_sat = k_dry + alpha**2 * M
    H = k_sat + 4.0 / 3.0 * mu
    C = alpha * M
    return {"alpha": alpha, "M": M, "k_sat": k_sat, "H": H, "C": C}


def biot_gassmann_velocity(
    *,
    k_solid: float,
    k_fluid: float,
    k_dry: float,
    mu: float,
    phi: float,
    rho_solid: float,
    rho_fluid: float,
) -> float:
    """The low-frequency (undrained) fast-P velocity ``sqrt(H / rho)`` of a fluid-saturated porous rock.

    This is the Biot-Gassmann reference velocity: Gassmann fluid substitution for the undrained modulus, the
    Voigt bulk density for ``rho``. The slow (Biot) wave is diffusive at low frequency and is not a wave speed
    here.
    """
    mod = gassmann_moduli(k_solid=k_solid, k_fluid=k_fluid, k_dry=k_dry, mu=mu, phi=phi)
    rho = (1.0 - phi) * rho_solid + phi * rho_fluid
    return float(np.sqrt(mod["H"] / rho))


class BiotPoroelastic1D:
    """A differentiable 1D Biot poroelastic-wave stepper (velocity-stress staggered leapfrog).

    ``BiotPoroelastic1D(n, dt=..., spacing=..., k_solid=..., k_fluid=..., k_dry=..., mu=..., phi=...,
    rho_solid=..., rho_fluid=..., eta=..., permeability=..., tortuosity=...)`` builds the forward on a
    length-``n`` grid. The rock is specified by the drained frame (``k_dry, mu``), the constituents
    (``k_solid, k_fluid, phi``), the densities (``rho_solid, rho_fluid``) and the fluid transport
    (``eta`` viscosity, ``permeability``, ``tortuosity``); the Biot-Gassmann moduli are derived internally.

    The state packs the four components in the order ``(v, q, sigma, pf)`` (``pack``/``unpack``): solid
    particle velocity ``v`` and relative fluid velocity ``q`` at the integer nodes, total stress ``sigma``
    and pore-fluid pressure ``pf`` at the half nodes. ``step(state, ops, source=...)`` advances one full
    leapfrog step (stress/pressure a half step, then the two velocities a half step with the drag integrated
    implicitly). ``vp`` is the Biot-Gassmann fast-P velocity; ``c_max`` sets the Courant limit
    ``dt <= spacing / c_max``.
    """

    def __init__(
        self,
        n: int,
        *,
        dt: float,
        spacing: float | None = None,
        k_solid: float = 36.0e9,
        k_fluid: float = 2.2e9,
        k_dry: float = 9.0e9,
        mu: float = 7.0e9,
        phi: float = 0.25,
        rho_solid: float = 2650.0,
        rho_fluid: float = 1000.0,
        eta: float = 1.0e-3,
        permeability: float = 1.0e-12,
        tortuosity: float = 2.0,
        absorb_width: int = 0,
        absorb_strength: float = 2.0,
    ):
        self.n = int(n)
        self.dt = float(dt)
        self.h = float(spacing) if spacing is not None else 1.0

        mod = gassmann_moduli(k_solid=k_solid, k_fluid=k_fluid, k_dry=k_dry, mu=mu, phi=phi)
        self.alpha = mod["alpha"]
        self.M = mod["M"]
        self.k_sat = mod["k_sat"]
        self.H = mod["H"]
        self.C = mod["C"]

        self.phi = float(phi)
        self.rho = (1.0 - phi) * rho_solid + phi * rho_fluid
        self.rho_f = float(rho_fluid)
        self.m = float(tortuosity) * rho_fluid / phi
        self.drag = float(eta) / float(permeability)  # eta / k

        # Biot-Gassmann fast-P velocity and the exact undrained characteristic speeds
        self.vp = float(np.sqrt(self.H / self.rho))
        R = np.array([[self.H, self.C], [self.C, self.M]])
        T = np.array([[self.rho, self.rho_f], [self.rho_f, self.m]])
        speeds = np.sqrt(np.linalg.eigvals(np.linalg.solve(T, R)).real)
        self.c_max = float(speeds.max())  # slightly above vp; sets the Courant limit

        # semi-implicit velocity update: (T + dt D) V_next = T V_cur + dt F, precompute (T + dt D)^{-1}
        self._T = T
        D = np.array([[0.0, 0.0], [0.0, self.drag]])
        Ainv = np.linalg.inv(T + self.dt * D)
        self._ainv = Ainv  # 2x2

        self._gamma = self._build_sponge(absorb_width, absorb_strength)

    def _build_sponge(self, width, strength):
        """A velocity-damping ramp near the two ends so the finite grid does not reflect outgoing waves."""
        n = self.n
        gamma = np.zeros(n)
        if width > 0:
            idx = np.arange(n)
            d = np.minimum(idx, n - 1 - idx)
            gamma = float(strength) * np.where(d < width, (1.0 - d / width) ** 2, 0.0)
        return gamma

    # ---- state packing -------------------------------------------------------------------------------
    def pack(self, v, q, sigma, pf):
        """Pack the four field arrays (each length ``n`` or flat) into the integrator state."""
        import torch

        parts = [torch.as_tensor(f, dtype=torch.float64).reshape(-1) for f in (v, q, sigma, pf)]
        return torch.cat(parts)

    def unpack(self, state):
        """Split a packed state into ``(v, q, sigma, pf)`` as flat length-``n`` tensors."""
        n = self.n
        return tuple(state[i * n : (i + 1) * n] for i in range(4))

    def fields(self, state):
        """Return the four components ``(v, q, sigma, pf)`` as length-``n`` tensors."""
        return self.unpack(state)

    def solid_velocity(self, state):
        """The solid particle velocity ``v`` (length ``n``)."""
        return self.unpack(state)[0]

    def fluid_velocity(self, state):
        """The relative fluid velocity ``q`` (length ``n``)."""
        return self.unpack(state)[1]

    def zeros(self, ops):
        """A zero field state (all four components) as a packed integrator state."""
        return ops.zeros(4 * self.n)

    # ---- staggered one-sided differences -------------------------------------------------------------
    @staticmethod
    def _dp(f, h):
        """Forward difference ``(f[i+1] - f[i]) / h``, centred a half cell forward (node -> half node)."""
        out = f * 0.0
        out[:-1] = (f[1:] - f[:-1]) / h
        return out

    @staticmethod
    def _dm(f, h):
        """Backward difference ``(f[i] - f[i-1]) / h``, centred a half cell backward (half node -> node)."""
        out = f * 0.0
        out[1:] = (f[1:] - f[:-1]) / h
        return out

    # ---- leapfrog step -------------------------------------------------------------------------------
    def step(self, state, ops, *, source=None):
        """Advance one full Biot leapfrog step (stress/pressure, then the two velocities).

        ``source`` is an optional dict of additive per-node rate injections for this step, keyed by component
        name (``"v"``, ``"q"``, ``"sigma"``, ``"pf"``); use :meth:`solid_source` or :meth:`pressure_source`
        to build one. The stress/pressure update sees the pre-update velocities; the velocity update sees the
        just-updated stress/pressure, with the viscous drag integrated implicitly (the 2x2 solve). The
        velocities are damped by the sponge near the ends.
        """
        h, dt = self.h, self.dt
        v, q, sigma, pf = self.unpack(state)
        src = source or {}

        def add(name, field):
            s = src.get(name)
            return field if s is None else field + dt * ops.tensor(np.asarray(s).reshape(-1))

        # --- stress / pressure update (forward difference of the velocities, at the half nodes) ---------
        dv = self._dp(v, h)
        dq = self._dp(q, h)
        sigma = add("sigma", sigma + dt * (self.H * dv + self.C * dq))
        pf = add("pf", pf + dt * (-(self.C * dv + self.M * dq)))

        # --- velocity update (backward difference of stress/pressure, at the nodes) ---------------------
        # explicit forces F = [d sigma/dx ; -d pf/dx]; implicit drag folded into (T + dt D)^{-1}
        fx = self._dm(sigma, h)
        gx = -self._dm(pf, h)
        rhs_v = self.rho * v + self.rho_f * q + dt * fx
        rhs_q = self.rho_f * v + self.m * q + dt * gx
        v_new = self._ainv[0, 0] * rhs_v + self._ainv[0, 1] * rhs_q
        q_new = self._ainv[1, 0] * rhs_v + self._ainv[1, 1] * rhs_q
        v = add("v", v_new)
        q = add("q", q_new)

        # absorbing sponge on the velocities near the ends
        gamma = ops.tensor(self._gamma)
        damp = 1.0 / (1.0 + dt * gamma)
        v = v * damp
        q = q * damp

        return ops.cat([v, q, sigma, pf])

    # ---- point sources -------------------------------------------------------------------------------
    def solid_source(self, position, amplitude=1.0):
        """A solid-velocity rate injection at ``position`` -- the standard way to launch a fast-P wave.

        Returns a ``source`` dict for :meth:`step` that adds ``amplitude`` to ``dv/dt`` at the node.
        """
        i = int(round(position))
        f = np.zeros(self.n)
        f[i] = float(amplitude)
        return {"v": f}

    def pressure_source(self, position, amplitude=1.0):
        """A pore-pressure rate injection at ``position`` -- an out-of-phase drive that excites the slow wave.

        Returns a ``source`` dict for :meth:`step` that adds ``amplitude`` to ``d pf/dt`` at the node.
        """
        i = int(round(position))
        f = np.zeros(self.n)
        f[i] = float(amplitude)
        return {"pf": f}
