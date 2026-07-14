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

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

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


# ======================================================================================================
# Quasi-static surface deformation (workstream G4): aquifer/reservoir volume change -> InSAR LOS
# displacement, and the inverse. `BiotPoroelastic1D` above is a *dynamic wave* stepper -- it has no
# notion of a static surface-deformation Green's function -- so this is a genuinely new forward, built
# on the same Biot-Gassmann moduli (`gassmann_moduli`) rather than on the wave stepper itself.
#
# Forward: a subsurface volume change dV (a dewatering aquifer cell contracting, a reservoir depleting)
# radiates a static elastic deformation field at the surface. Modelled as a superposition of Mogi (1958)
# point "nuclei of strain" -- the standard, cheap closed-form approximation for a source small compared
# to its depth (the same simplifying step `gravity_point_sensitivity`/`magnetic_dipole_sensitivity` make
# in mixle_pde.geophysics for a point-mass/point-dipole cell): each cell's dV contributes independently
# and linearly, so the whole operator is one (n_obs, n_cells) sensitivity matrix, exact and O(1) to
# evaluate (no adjoint or finite-difference Jacobian is needed -- unlike the iterative PDE forwards C1's
# adjoint targets, this forward is already a closed-form linear kernel, so its own derivative IS the
# kernel; see the note in `invert_deformation`).
#
# For a point source of strength dV at depth `d` below an unbounded elastic half-space of Poisson ratio
# `nu`, the Mogi surface displacement at horizontal offset `r` from the point directly above the source
# is: `u_z = (1-nu)/pi * dV * d / (d^2+r^2)^1.5` (vertical) and `u_r = (1-nu)/pi * dV * r / (d^2+r^2)^1.5`
# (radially outward from the epicentre, positive dV = inflation = uplift; negative dV = deflation /
# dewatering = subsidence). `nu` and the point-source strength are read off the Biot-Gassmann moduli
# (`gassmann_moduli`): `mu = 3/4 (H - k_sat)` recovers the shear modulus from the returned `H`/`k_sat`
# (since `H = k_sat + 4/3 mu`), `nu = (3 k_sat - 2 mu) / (2 (3 k_sat + mu))` is the standard elastic
# relation, and the point-source strength is `alpha * dV` -- the Biot coefficient scales the *mechanical*
# (skeleton) volume change a pore-pressure/fluid volume change `dV` actually produces (Geertsma 1973;
# Segall 1985/1992's nucleus-of-strain reservoir-compaction model).
def _biot_nu_and_alpha(moduli: dict) -> tuple[float, float]:
    """Elastic Poisson ratio and the Biot coefficient, both recovered from a `gassmann_moduli()` dict."""
    alpha = float(moduli["alpha"])
    k_sat = float(moduli["k_sat"])
    mu = 0.75 * (float(moduli["H"]) - k_sat)
    nu = (3.0 * k_sat - 2.0 * mu) / (2.0 * (3.0 * k_sat + mu))
    return nu, alpha


def _mogi_los_sensitivity(
    cells: np.ndarray,
    obs_xy: np.ndarray,
    *,
    moduli: dict,
    los_vector: np.ndarray,
) -> np.ndarray:
    """The ``(n_obs, n_cells)`` linear sensitivity of LOS surface displacement to per-cell volume change.

    ``cells``/``obs_xy`` are ``(*, 3)`` ``(east, north, up)`` coordinates (metres), matching the
    convention `mixle_pde.geophysics` already uses -- so every observation must sit strictly above every
    cell (a positive Mogi source depth); this raises rather than silently producing an unphysical result
    otherwise. Superposes the per-cell Mogi point-source vertical/radial displacement, resolves the
    radial component into (east, north) via the cell-to-observation bearing, and projects the resulting
    3-vector onto the (already-normalized) ``los_vector``.
    """
    cells = np.atleast_2d(np.asarray(cells, dtype=float))
    obs = np.atleast_2d(np.asarray(obs_xy, dtype=float))
    if cells.ndim != 2 or cells.shape[1] != 3:
        raise ValueError(f"cells must be an (n_cells, 3) array, got shape {cells.shape}.")
    if obs.ndim != 2 or obs.shape[1] != 3:
        raise ValueError(f"obs_xy must be an (n_obs, 3) array, got shape {obs.shape}.")

    nu, alpha = _biot_nu_and_alpha(moduli)

    diff = obs[:, None, :] - cells[None, :, :]  # (n_obs, n_cells, 3)
    depth = diff[..., 2]
    if np.any(depth <= 0.0):
        raise ValueError(
            "every cell must sit strictly below every observation in the (east, north, up) convention "
            "(depth = obs_z - cell_z must be positive); got a non-positive depth for at least one pair."
        )
    horiz = diff[..., :2]
    r = np.maximum(np.linalg.norm(horiz, axis=-1), 1.0e-9)
    R3 = (depth**2 + r**2) ** 1.5

    coeff = (1.0 - nu) / np.pi * alpha / R3  # (n_obs, n_cells)
    uz = coeff * depth
    ur = coeff * r
    ux = ur * horiz[..., 0] / r
    uy = ur * horiz[..., 1] / r

    los = np.asarray(los_vector, dtype=float)
    los = los / np.linalg.norm(los)
    return ux * los[0] + uy * los[1] + uz * los[2]


def poroelastic_subsidence(
    volume_change: np.ndarray,
    cells: np.ndarray,
    obs_xy: np.ndarray,
    *,
    moduli: dict,
    los_vector: np.ndarray | tuple[float, float, float] = (0.0, 0.0, 1.0),
) -> np.ndarray:
    """Predicted surface LOS displacement from a per-cell subsurface volume change (the G4 forward).

    Args:
        volume_change: ``(n_cells,)`` volume change per cell, m^3 (negative = contraction/dewatering,
            positive = inflation).
        cells: ``(n_cells, 3)`` cell-centroid coordinates, ``(east, north, up)`` metres.
        obs_xy: ``(n_obs, 3)`` surface observation coordinates, ``(east, north, up)`` metres (``up`` is
            typically ``0`` at the ground surface).
        moduli: the dict returned by :func:`gassmann_moduli` (needs ``alpha``, ``k_sat``, ``H``).
        los_vector: unit line-of-sight vector, ``(east, north, up)`` convention (see
            :func:`mixle_pde.io.insar.load_insar`); defaults to straight up.

    Returns:
        ``(n_obs,)`` predicted LOS displacement (metres), linear in ``volume_change``.
    """
    G = _mogi_los_sensitivity(cells, obs_xy, moduli=moduli, los_vector=los_vector)
    dv = np.atleast_1d(np.asarray(volume_change, dtype=float))
    if dv.shape != (G.shape[1],):
        raise ValueError(f"volume_change must have shape ({G.shape[1]},), got {dv.shape}.")
    return G @ dv


@dataclass
class _PushforwardQuantity:
    """IC-1 `DerivedQuantity`: a pushforward's draws plus the honesty flag inherited from its posterior."""

    samples: np.ndarray
    prior_dominated: bool = False

    def credible_interval(self, level: float) -> tuple[np.ndarray, np.ndarray]:
        if not 0.0 < level < 1.0:
            raise ValueError("level must be in (0, 1).")
        a = (1.0 - level) / 2.0
        return np.quantile(self.samples, a, axis=0), np.quantile(self.samples, 1.0 - a, axis=0)


@dataclass
class DeformationPosterior:
    """IC-1 `Posterior` over per-cell volume/pressure change, recovered by :func:`invert_deformation`.

    Wraps the exact closed-form :class:`~mixle_pde.latent.PosteriorField3D` the linear-Gaussian
    `insar_los` inversion returns, bridging its field-specific singular `.sample`/`alpha`-parameterised
    `.credible_interval` surface onto the frozen IC-1 plural `samples`/`level`-parameterised surface
    (`mixle.reason.posterior_protocol.Posterior`) every downstream consumer (E3-E5, E10, H4, J2) types
    against, without waiting on `PosteriorField3D` itself growing that surface generically (a separate
    card, E1, does that).
    """

    field_posterior: Any  # mixle_pde.latent.PosteriorField3D
    prior_dominated: bool = False

    @property
    def mean(self) -> np.ndarray:
        return self.field_posterior.mean

    @property
    def cov(self) -> np.ndarray:
        return self.field_posterior.cov

    def samples(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Draw ``n`` per-cell volume/pressure-change samples; shape ``(n, n_cells)``."""
        return self.field_posterior.sample(n, rng)

    def credible_interval(self, level: float) -> tuple[np.ndarray, np.ndarray]:
        """Per-cell central credible interval covering ``level`` mass; ``(lo, hi)`` each ``(n_cells,)``."""
        if not 0.0 < level < 1.0:
            raise ValueError("level must be in (0, 1).")
        return self.field_posterior.credible_interval(alpha=1.0 - level)

    def derived_quantity(
        self, fn: Callable[[np.ndarray], np.ndarray], n: int, rng: np.random.Generator
    ) -> _PushforwardQuantity:
        """Pushforward ``fn`` over ``n`` posterior draws (e.g. total recovered dV, or its centroid)."""
        draws = self.samples(n, rng)
        pushed = np.asarray([fn(draw) for draw in draws])
        return _PushforwardQuantity(samples=pushed, prior_dominated=self.prior_dominated)


def invert_deformation(
    insar_obs: list,
    cells: np.ndarray,
    *,
    moduli: dict,
    how: str = "laplace",
    prior_volume_scale: float = 1.0e4,
    smoothness_precision: float = 2.0e-4,
    length_scale: float | None = None,
) -> DeformationPosterior:
    """Posterior over per-cell subsurface volume/pressure change from a batch of `insar_los` observations.

    The forward (:func:`poroelastic_subsidence`) is an exact closed-form linear kernel in the per-cell
    volume change (a Mogi-source superposition, not an iterative PDE solve), so its Jacobian is the
    kernel itself -- cheap and exact, with none of the adjoint/finite-difference machinery C1 supplies
    for the PDE-based DC/EM operators needed here. This is registered as a
    :class:`~mixle_pde.observations.ForwardOperator` (the same pattern
    :func:`mixle_pde.observations.gravity_forward_operator` uses for the point-mass gravity kernel) and
    inverted exactly by :func:`mixle_pde.field_inversion.linear_gaussian_invert` against a Gaussian
    spatial-smoothness prior over the cells -- the linear-Gaussian path the algorithm names as the
    alternative to `Differential`/`joint(...).fit("laplace")`, and the natural one here since the
    forward is already linear rather than an ODE/PDE trajectory. Because the model is exactly
    linear-Gaussian, ``how="laplace"`` and ``how="map"`` coincide (the Laplace approximation at the MAP
    IS the exact posterior); both are accepted for API symmetry with the rest of the package.

    Args:
        insar_obs: a list of ``kind="insar_los"`` Observations (as returned by
            :func:`mixle_pde.io.insar.load_insar`); every observation must carry the SAME
            ``provenance["los_vector"]`` (one InSAR track's look direction is treated as constant over
            the AOI). Missing ``los_vector`` defaults to straight up.
        cells: ``(n_cells, 3)`` cell-centroid coordinates, ``(east, north, up)`` metres.
        moduli: the dict returned by :func:`gassmann_moduli`.
        how: ``"laplace"`` or ``"map"`` (see above; both take the same closed-form path).
        prior_volume_scale: prior standard deviation on a cell's volume change, m^3 -- a broad,
            weakly-informative default so a well-resolved anomaly is set by the data, not the prior.
        smoothness_precision: the spatial-smoothness prior's edge weight (see
            :class:`~mixle_pde.field_inversion.FieldGaussianPrior`); a small default so neighbouring
            cells are only weakly coupled and a resolvable anomaly's shape and total credible width are
            set by the data, not by over-smoothing.
        length_scale: the smoothness prior's correlation length, metres; defaults to the median nearest-
            neighbour cell spacing.

    Returns:
        A :class:`DeformationPosterior` (IC-1 `Posterior`) over the per-cell volume/pressure change.
    """
    if how not in ("laplace", "map"):
        raise ValueError(f"how must be 'laplace' or 'map', got {how!r}.")
    if not insar_obs:
        raise ValueError("invert_deformation needs at least one insar_los observation.")
    if prior_volume_scale <= 0.0:
        raise ValueError("prior_volume_scale must be positive.")

    from mixle_pde.field_inversion import FieldGaussianPrior, linear_gaussian_invert
    from mixle_pde.latent import Field3D
    from mixle_pde.observations import ForwardOperator, ForwardOperatorRegistry

    los_vectors = [tuple(np.round(obs.provenance.get("los_vector", (0.0, 0.0, 1.0)), 9)) for obs in insar_obs]
    if len(set(los_vectors)) > 1:
        raise ValueError("invert_deformation needs one shared los_vector across insar_obs; got several.")
    los_vector = np.asarray(los_vectors[0], dtype=float)

    cells = np.atleast_2d(np.asarray(cells, dtype=float))
    n_cells = cells.shape[0]
    grid = Field3D(coordinates=cells, spacing=1.0, units="m^3", property_name="volume_change", bounds=None)

    if length_scale is None:
        from scipy.spatial import cKDTree

        tree = cKDTree(cells)
        dist, _ = tree.query(cells, k=min(2, n_cells))
        length_scale = float(np.median(dist[:, -1])) if n_cells > 1 else 1.0
        length_scale = max(length_scale, 1.0e-6)

    def _jacobian(_grid: Field3D, obs_locations: np.ndarray) -> np.ndarray:
        return _mogi_los_sensitivity(cells, obs_locations, moduli=moduli, los_vector=los_vector)

    def _predict(_grid: Field3D, field_values: np.ndarray, obs_locations: np.ndarray) -> np.ndarray:
        return _jacobian(_grid, obs_locations) @ np.asarray(field_values, dtype=float)

    registry = ForwardOperatorRegistry()
    registry.register(ForwardOperator("insar_los", _predict, jacobian=_jacobian, differentiable=False))

    prior = FieldGaussianPrior(
        mean=0.0,
        smoothness_precision=smoothness_precision,
        marginal_precision=1.0 / prior_volume_scale**2,
        length_scale=length_scale,
    )
    field_posterior = linear_gaussian_invert(grid, insar_obs, registry, prior)

    prior_trace = float(np.trace(prior.precision(grid)))
    posterior_trace = float(np.trace(np.linalg.inv(field_posterior.cov)))
    prior_dominated = prior_trace > 0.5 * posterior_trace

    return DeformationPosterior(field_posterior=field_posterior, prior_dominated=prior_dominated)
