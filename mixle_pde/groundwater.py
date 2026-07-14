"""Groundwater flow, reactive solute transport, and contaminant source inversion (workstream G).

Two cards share this module:

* **G1** ("Groundwater flow + reactive solute-transport operator") owns :func:`darcy_velocity` (steady
  Darcy flow, via :func:`mixle_pde.multiphysics.solve_poisson`), :class:`GroundwaterTransportOperator`
  (a :class:`~mixle_pde.dynamics.DynamicsOperator` for reactive solute transport in a -- possibly
  spatially varying -- velocity field: per-axis upwind advection, velocity-dependent dispersion,
  first-order decay, and linear retardation), and :func:`ogata_banks_plume` (the classic 1-D
  continuous-injection analytic solution used to validate the operator against a textbook result).
* **G2** ("Contaminant source inversion") recovers a posterior over an unknown contaminant source's
  ``(location, rate, onset time)`` from noisy concentration time series at a handful of monitoring
  wells, built on top of :class:`GroundwaterTransportOperator` -- see :func:`invert_source`.

**Scope note.** G2 was implemented before G1 landed on ``release/0.8.0``; since G2's inversion only
ever needs a transport operator with a *given, fixed* velocity field (it never calls
:func:`darcy_velocity`), G2 originally shipped a narrower stand-in operator (uniform per-axis velocity
only, no Darcy coupling) under the same ``mixle_pde.groundwater`` module and ``"groundwater"``
dynamics-operator registry name, by design "additive on top of [G1] once it lands, not a rewrite of
it" (G2's own words). Landing G1 folds that stand-in into the fuller implementation below:
:class:`GroundwaterTransportOperator` now accepts BOTH a full per-cell velocity field (G1's spatially
varying Darcy-flow case) and a uniform per-axis velocity vector (G2's shorthand, broadcast to a
spatially-constant field), and a per-axis dispersivity array (G2's anisotropic-dispersion shorthand,
generalizing G1's single scalar dispersivity), and exposes the grid bookkeeping (``coords``,
``domain_bounds``, ``nearest_index``) G2's source-inversion code needs, so neither card's own test
suite has to change.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.special import erfc

from mixle_pde.dynamics import DynamicsOperator, laplacian_matrix, register_dynamics_operator, upwind_gradient_matrix
from mixle_pde.multiphysics import solve_poisson
from mixle_pde.observations import Observation

__all__ = [
    "darcy_velocity",
    "GroundwaterTransportOperator",
    "ogata_banks_plume",
    "SourceRecalibration",
    "HeldOutPrediction",
    "SourcePosterior",
    "simulate_wells",
    "invert_source",
]


def _shape_tuple(shape: Any) -> tuple[int, ...]:
    return tuple(int(s) for s in np.atleast_1d(shape))


def _spacing_tuple(spacing: Any, ndim: int) -> tuple[float, ...]:
    if np.isscalar(spacing):
        return tuple(float(spacing) for _ in range(ndim))
    arr = np.atleast_1d(np.asarray(spacing, dtype=float))
    if arr.size != ndim:
        raise ValueError(f"spacing must be a scalar or length {ndim} (one per axis); got size {arr.size}.")
    return tuple(float(s) for s in arr)


def darcy_velocity(
    hydraulic_conductivity: Any,
    source: Any,
    shape: Any,
    *,
    spacing: Any = 1.0,
    bc: str = "neumann",
) -> np.ndarray:
    """Steady Darcy specific-discharge field ``q = -K grad h`` on a structured grid.

    ``h`` is the steady head solving ``-div(K grad h) = source`` via
    :func:`mixle_pde.multiphysics.solve_poisson` (``source`` plays the role of a pumping/
    injection/recharge term: positive where water is added, negative where it is withdrawn;
    the domain boundary is held at the solver's default fixed head of 0, i.e. a far-field
    aquifer boundary). ``q`` is then formed by a cell-centred (central-difference) gradient of
    ``h``, scaled by ``-K``, generalizing the scalar ``velocity`` of
    :class:`~mixle_pde.dynamics.AdvectionDiffusionOperator` to a spatially-varying field.

    Returns an ``(ndim, *shape)`` array: one flux component per grid axis.

    ``bc`` controls only the boundary treatment of that gradient (``solve_poisson`` itself has
    no Neumann option, just a fixed-head Dirichlet boundary): ``"neumann"`` (the default) zeroes
    the outward-normal flux component at the two faces of each axis (a closed, no-flow bounding
    box -- the common assumption when the domain edge is just a numerical truncation, not a real
    hydraulic boundary); anything else keeps the one-sided finite difference there instead (an
    open boundary, letting water cross the edge -- consistent with the fixed-head condition
    ``solve_poisson`` already applies at those same nodes).
    """
    shp = _shape_tuple(shape)
    ndim = len(shp)
    spc = _spacing_tuple(spacing, ndim)
    head = solve_poisson(shp, source, hydraulic_conductivity, spacing=spacing)
    k_field = (
        float(hydraulic_conductivity)
        if np.isscalar(hydraulic_conductivity)
        else np.asarray(hydraulic_conductivity, dtype=float).reshape(shp)
    )
    grads = np.gradient(head, *spc) if ndim > 1 else [np.gradient(head, spc[0])]
    q = np.empty((ndim, *shp), dtype=float)
    for axis in range(ndim):
        g = np.asarray(grads[axis], dtype=float)
        if bc == "neumann":
            g = g.copy()
            face = [slice(None)] * ndim
            face[axis] = 0
            g[tuple(face)] = 0.0
            face[axis] = -1
            g[tuple(face)] = 0.0
        q[axis] = -k_field * g
    return q


def _kron_axis(mat_1d: np.ndarray, axis: int, shape: tuple[int, ...]) -> np.ndarray:
    """Kronecker-expand a 1-D ``(n_axis, n_axis)`` operator to act along ``axis`` of an n-D grid.

    The grid is flattened in NumPy's default (row-major / C) order, for which the operator
    acting only along ``axis`` is the Kronecker product of an identity on every other axis with
    ``mat_1d`` in position ``axis`` -- the standard construction for a multi-dimensional finite-
    difference operator built from 1-D stencils.
    """
    out = None
    for ax, n_ax in enumerate(shape):
        block = mat_1d if ax == axis else np.eye(n_ax)
        out = block if out is None else np.kron(out, block)
    return out


class GroundwaterTransportOperator(DynamicsOperator):
    """Reactive solute transport advected by a (possibly spatially varying) Darcy velocity.

    Generalizes :class:`~mixle_pde.dynamics.AdvectionDiffusionOperator`'s single scalar
    ``velocity`` to a per-cell velocity field (e.g. from :func:`darcy_velocity`), and adds
    velocity-dependent dispersion, first-order decay, and linear retardation on top:

        R du/dt = -v . grad(u) + div(D grad u) - lambda u,   D_axis = alpha_axis |v| + D_mol

    ``velocity_field`` accepts three shapes: the full ``(ndim, *shape)`` per-cell field
    :func:`darcy_velocity` returns; for a 1-D ``shape`` a plain ``(n,)`` array as shorthand for
    ``(1, n)``; or a length-``ndim`` vector, a spatially UNIFORM per-axis velocity broadcast to
    the full field (the shorthand a fixed, given velocity -- e.g. G2's contaminant-source
    inversion, which never solves for the Darcy field itself -- uses). ``dispersivity`` (the
    longitudinal dispersivity ``alpha_L``) accepts either one scalar shared by every axis or a
    length-``ndim`` per-axis array (anisotropic dispersion); either way it is scaled by the local
    flow *speed* ``|v|`` (a per-cell field when ``velocity_field`` is spatially varying) plus a
    constant ``molecular_diffusion`` floor ``D_mol``. ``decay`` (``lambda``) and ``retardation``
    (``R``) may be scalars or per-cell arrays (shape ``shape``).
    """

    def __init__(
        self,
        velocity_field: Any,
        dispersivity: Any,
        shape: Any,
        *,
        retardation: Any = 1.0,
        decay: Any = 0.0,
        spacing: Any = 1.0,
        bc: str = "neumann",
        scheme: str = "implicit",
        molecular_diffusion: float = 0.0,
    ) -> None:
        self.shape = _shape_tuple(shape)
        if any(s < 3 for s in self.shape):
            raise ValueError("each grid axis needs at least 3 points.")
        self.ndim = len(self.shape)
        n = int(np.prod(self.shape))
        super().__init__(n=n, length=1.0, bc=bc, scheme=scheme)
        self.spacing = _spacing_tuple(spacing, self.ndim)
        self.dispersivity = np.broadcast_to(np.asarray(dispersivity, dtype=float), (self.ndim,)).copy()
        self.molecular_diffusion = float(molecular_diffusion)
        if np.isscalar(retardation) and float(retardation) <= 0.0:
            raise ValueError("retardation must be strictly positive.")
        self.decay = self._per_cell(decay)
        self.retardation = self._per_cell(retardation)
        self.velocity_field = self._normalize_velocity(velocity_field)

        # Cell-centre coordinates (C-order, matching the Kronecker construction below) and the
        # physical (lo, hi) domain box -- the grid bookkeeping G2's source inversion snaps
        # monitoring-well / candidate-source locations onto.
        axes = [np.arange(s) * self.spacing[a] for a, s in enumerate(self.shape)]
        mesh = np.meshgrid(*axes, indexing="ij")
        self.coords = np.stack([m.reshape(-1) for m in mesh], axis=1)  # (n, ndim)

    def _per_cell(self, value: Any) -> float | np.ndarray:
        if np.isscalar(value):
            return float(value)
        return np.asarray(value, dtype=float).reshape(self.shape)

    def _normalize_velocity(self, velocity_field: Any) -> np.ndarray:
        vf = np.asarray(velocity_field, dtype=float)
        full_shape = (self.ndim, *self.shape)
        if vf.shape == full_shape:
            return vf
        if self.ndim == 1 and vf.shape == self.shape:
            return vf[np.newaxis, ...]
        if vf.shape == (self.ndim,):
            # A spatially uniform per-axis velocity (G2's shorthand): broadcast to the full field.
            broadcast_shape = (self.ndim,) + (1,) * self.ndim
            return np.broadcast_to(vf.reshape(broadcast_shape), full_shape).copy()
        raise ValueError(
            f"velocity_field must have shape {full_shape}, {self.shape} (ndim == 1 shorthand), "
            f"or ({self.ndim},) (uniform per-axis); got {vf.shape}."
        )

    @property
    def domain_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """The physical ``(lo, hi)`` box the grid covers, one entry per axis."""
        hi = np.array([(s - 1) * self.spacing[a] for a, s in enumerate(self.shape)])
        return np.zeros_like(hi), hi

    def nearest_index(self, point: Any) -> int:
        """The flat cell index whose centre is closest to ``point`` (an ``(ndim,)``-ish coordinate)."""
        point = np.asarray(point, dtype=float).reshape(-1)[: self.coords.shape[1]]
        d2 = np.sum((self.coords - point) ** 2, axis=1)
        return int(np.argmin(d2))

    def operator_matrix(self) -> np.ndarray:
        """``-(1/R)[v . grad] + div(D grad) - lambda`` -- ``R`` divides the whole transient term."""
        n = self.n
        speed = np.sqrt(sum(self.velocity_field[axis] ** 2 for axis in range(self.ndim)))

        advection = np.zeros((n, n))
        dispersion = np.zeros((n, n))
        for axis in range(self.ndim):
            n_axis = self.shape[axis]
            h_axis = self.spacing[axis]
            lap_full = _kron_axis(laplacian_matrix(n_axis, h_axis, self.bc), axis, self.shape)
            g_pos_full = _kron_axis(upwind_gradient_matrix(n_axis, h_axis, 1.0, self.bc), axis, self.shape)
            g_neg_full = _kron_axis(upwind_gradient_matrix(n_axis, h_axis, -1.0, self.bc), axis, self.shape)

            q_flat = self.velocity_field[axis].reshape(-1)
            upwind_selected = np.where((q_flat >= 0.0)[:, None], g_pos_full, g_neg_full)
            advection += -(q_flat[:, None] * upwind_selected)

            dispersion_field_axis = self.dispersivity[axis] * speed + self.molecular_diffusion
            dispersion += dispersion_field_axis.reshape(-1)[:, None] * lap_full

        decay_diag = np.full(n, self.decay) if np.isscalar(self.decay) else self.decay.reshape(-1)
        raw = advection + dispersion - np.diag(decay_diag)
        if np.isscalar(self.retardation):
            return raw / self.retardation
        return raw / self.retardation.reshape(-1)[:, None]


register_dynamics_operator("groundwater", GroundwaterTransportOperator)


def ogata_banks_plume(x: Any, t: Any, velocity: float, dispersion: float, *, c0: float = 1.0) -> np.ndarray:
    """Ogata & Banks (1961) analytic 1-D continuous-injection solute plume.

    The exact solution of ``dC/dt = -v dC/dx + D d^2C/dx^2`` on a semi-infinite column
    ``x >= 0`` with concentration held at ``c0`` at ``x=0`` for all ``t>0`` and ``C(x,0)=0``:

        C(x,t)/c0 = 1/2 [erfc((x - v t)/(2 sqrt(D t))) + exp(v x / D) erfc((x + v t)/(2 sqrt(D t)))]

    ``x`` and ``t`` broadcast against each other (an array of positions at one time, an array of
    times at one position, or a matching pair of arrays). The reflected second term underflows
    to (correctly) zero for large ``v x / D`` before ``exp`` can overflow.
    """
    x = np.asarray(x, dtype=float)
    t = np.asarray(t, dtype=float)
    v = float(velocity)
    d = float(dispersion)
    t_safe = np.where(t > 0.0, t, np.nan)
    root = 2.0 * np.sqrt(d * t_safe)
    term1 = erfc((x - v * t_safe) / root)
    exponent = np.clip(v * x / d, None, 700.0)
    term2 = np.where(v * x / d > 700.0, 0.0, np.exp(exponent) * erfc((x + v * t_safe) / root))
    c = 0.5 * c0 * (term1 + term2)
    return np.where(t > 0.0, np.nan_to_num(c, nan=0.0), 0.0)


# --------------------------------------------------------------------------- shared forward physics
def _source_weights(coords: np.ndarray, location: np.ndarray, sigma: float) -> np.ndarray:
    diff = coords - location
    dist2 = np.sum(diff * diff, axis=1)
    w = np.exp(-dist2 / (2.0 * sigma * sigma))
    return w / w.sum()


def _onset_mask(t: float, onset: float, eps: float) -> float:
    return 0.5 * (1.0 + np.tanh((t - onset) / eps))


def _simulate_numpy(
    location: np.ndarray,
    rate: float,
    onset: float,
    operator: GroundwaterTransportOperator,
    transition: np.ndarray,
    times: np.ndarray,
    sigma: float,
) -> np.ndarray:
    """Plain-NumPy replica of the differentiable forward model -- same physics, no autograd -- used for
    synthetic-data generation and posterior-predictive Monte Carlo (neither needs a gradient)."""
    weight = _source_weights(operator.coords, np.asarray(location, dtype=float), sigma)
    dt = float(times[1] - times[0]) if len(times) > 1 else 1.0
    eps = max(dt * 0.5, 1.0e-6)
    state = np.zeros(operator.n)
    history = [state]
    for t_next in times[1:]:
        active = _onset_mask(float(t_next), float(onset), eps)
        source = float(rate) * active * weight
        state = transition @ (state + dt * source)
        history.append(state)
    return np.stack(history, axis=0)  # (len(times), n)


def simulate_wells(
    location,
    rate: float,
    onset: float,
    operator: GroundwaterTransportOperator,
    well_xy,
    times,
    *,
    sigma: float | None = None,
) -> np.ndarray:
    """Forward-simulate a known point source to a set of monitoring wells (synthetic-data convenience).

    ``well_xy`` is a sequence of physical coordinates (snapped to the nearest grid cell); ``times`` is
    the (uniform, increasing) sample-time grid. Returns ``(len(times), len(well_xy))`` concentrations --
    plain NumPy, not the differentiable path :func:`invert_source` fits against.
    """
    times = np.asarray(times, dtype=float)
    if times[0] != 0.0:
        times = np.concatenate([[0.0], times])
        prepended = True
    else:
        prepended = False
    sigma = float(np.mean(operator.spacing)) if sigma is None else float(sigma)
    dt = float(times[1] - times[0])
    transition = operator.transition_matrix(dt)
    sol = _simulate_numpy(location, rate, onset, operator, transition, times, sigma)
    idx = [operator.nearest_index(xy) for xy in well_xy]
    out = sol[:, idx]
    return out[1:] if prepended else out


# --------------------------------------------------------------------------- calibration bookkeeping
@dataclass(frozen=True)
class SourceRecalibration:
    """Chi-square variance inflation + split-conformal quantile from held-out wells (A3's recipe).

    ``inflation`` (``> 1`` means the raw Laplace posterior was overconfident) rescales the (location,
    rate) covariance; ``conformal_quantile`` is the distribution-free multiplier such that
    ``predicted +/- conformal_quantile * predictive_std`` should cover a fresh held-out reading at rate
    ``1 - alpha``, without relying on the Laplace fit's Gaussian assumption.
    """

    inflation: float
    conformal_quantile: float
    alpha: float


@dataclass(frozen=True)
class HeldOutPrediction:
    """Posterior-predictive concentration at the held-out wells used to compute :class:`SourceRecalibration`.

    Kept on the returned posterior so a caller can re-check split-conformal coverage against *fresh*
    noise draws around the same held-out points without re-fitting.
    """

    observed: np.ndarray
    predictive_mean: np.ndarray
    predictive_std: np.ndarray


@dataclass(frozen=True)
class _DerivedQuantity:
    """IC-1 ``DerivedQuantity``: a pushforward's draws + the prior-dominated honesty flag."""

    samples: np.ndarray
    prior_dominated: bool

    def credible_interval(self, level: float) -> tuple[np.ndarray, np.ndarray]:
        a = (1.0 - level) / 2.0
        return np.quantile(self.samples, a, axis=0), np.quantile(self.samples, 1.0 - a, axis=0)


@dataclass
class SourcePosterior:
    """IC-1 ``Posterior`` over the source ``(location..., rate)`` vector, recalibrated per A3's recipe.

    ``mean``/``cov`` are already the *recalibrated* (inflation-scaled) Gaussian approximation to the
    Laplace posterior; ``onset`` is reported separately (mean, sd) as a convenience -- it is not part of
    the vector IC-1's ``Posterior`` is over (G2's Public API is explicit: "a posterior ... over the
    (loc, rate) vector"). ``prior_dominated`` is always ``False`` here: the (location, rate, onset)
    drivers carry no informative prior beyond their support transform, so the fit's width is always
    data-determined, never regularizer-dominated -- the flag is still carried for IC-1 conformance.
    """

    mean: np.ndarray
    cov: np.ndarray
    recalibration: SourceRecalibration
    onset: tuple[float, float]
    held_out_prediction: HeldOutPrediction
    prior_dominated: bool = False

    def samples(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Draw ``n`` joint (location..., rate) samples from the recalibrated Gaussian, shape ``(n, d)``."""
        return rng.multivariate_normal(self.mean, self.cov, size=int(n))

    def credible_interval(self, level: float) -> tuple[np.ndarray, np.ndarray]:
        draws = self.samples(20_000, np.random.default_rng(0))
        a = (1.0 - level) / 2.0
        return np.quantile(draws, a, axis=0), np.quantile(draws, 1.0 - a, axis=0)

    def derived_quantity(
        self, fn: Callable[[np.ndarray], np.ndarray], n: int, rng: np.random.Generator
    ) -> _DerivedQuantity:
        draws = self.samples(n, rng)
        return _DerivedQuantity(samples=np.asarray(fn(draws)), prior_dominated=self.prior_dominated)


# --------------------------------------------------------------------------- the inversion
def _group_wells(observations: list[Observation]) -> list[list[Observation]]:
    """Group a flat observation list into per-well lists, in first-seen well order."""
    groups: dict[tuple[float, ...], list[Observation]] = {}
    order: list[tuple[float, ...]] = []
    for obs in observations:
        key = tuple(np.round(np.asarray(obs.location[0], dtype=float), 6).tolist())
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(obs)
    return [groups[k] for k in order]


def invert_source(
    concentration_obs: list[Observation],
    operator: GroundwaterTransportOperator,
    *,
    loc_support: str = "box",
    rate_support: str = "positive",
    onset_support: str = "real",
    how: str = "laplace",
    alpha: float = 0.1,
    held_out_fraction: float = 1.0 / 3.0,
    n_location_draws: int = 20_000,
    n_predictive_draws: int = 400,
    seed: int = 0,
) -> SourcePosterior:
    """Posterior over an unknown source's ``(location, rate, onset)`` from monitoring-well concentrations.

    ``concentration_obs`` is one :class:`~mixle_pde.observations.Observation` per (well, sample-time)
    reading (``location`` the well's xyz, ``value``/``time`` the single concentration/time it was taken
    at, ``noise_cov`` its variance) -- the natural granularity :class:`Observation` already supports.
    ``operator`` is a fitted/known :class:`GroundwaterTransportOperator` (the transport physics; only its
    *source term* is unknown). The last ``held_out_fraction`` of the distinct wells (by first-seen
    order) are withheld from the fit and used only to recalibrate the returned posterior -- a proper
    train/calibration split, not a diagnostic re-check of the same data the fit used.

    Returns a :class:`SourcePosterior` over the concatenated ``(location..., rate)`` vector, satisfying
    the IC-1 ``Posterior`` protocol (see the module docstring for why this duck-types rather than
    imports it).
    """
    if loc_support != "box":
        raise ValueError("invert_source only implements loc_support='box' (a domain-box reparameterization).")
    if rate_support != "positive":
        raise ValueError("invert_source only implements rate_support='positive'.")
    if onset_support != "real":
        raise ValueError("invert_source only implements onset_support='real'.")
    if how != "laplace":
        raise ValueError("invert_source only implements how='laplace'.")

    from mixle.ppl import free, joint

    from mixle_pde.inverse import Differential

    wells = _group_wells(list(concentration_obs))
    if len(wells) < 3:
        raise ValueError("invert_source needs at least 3 monitoring wells (some are held out for calibration).")
    n_held = max(1, round(len(wells) * held_out_fraction))
    if n_held >= len(wells):
        raise ValueError("held_out_fraction leaves no wells to fit on.")
    fit_wells, held_wells = wells[:-n_held], wells[-n_held:]
    fit_obs = [o for well in fit_wells for o in well]
    held_obs = [o for well in held_wells for o in well]

    ndim = operator.coords.shape[1]
    box_lo, box_hi = operator.domain_bounds
    box_lo = np.asarray(box_lo, dtype=float)
    box_hi = np.asarray(box_hi, dtype=float)
    sigma = float(np.mean(operator.spacing))

    all_obs = fit_obs + held_obs
    times = np.union1d(np.array([0.0]), np.array(sorted({float(o.time) for o in all_obs}), dtype=float))
    gaps = np.diff(times)
    if gaps.size and not np.allclose(gaps, gaps[0], atol=1e-6):
        raise ValueError("invert_source needs observation times on a common uniform grid (shared across wells).")
    dt = float(gaps[0]) if gaps.size else 1.0
    time_index = {round(float(t), 9): i for i, t in enumerate(times)}
    transition = operator.transition_matrix(dt)

    def _index_obs(obs_list: list[Observation]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        cell = np.array([operator.nearest_index(o.location[0]) for o in obs_list], dtype=np.int64)
        tidx = np.array([time_index[round(float(o.time), 9)] for o in obs_list], dtype=np.int64)
        val = np.array([float(o.value[0]) for o in obs_list], dtype=float)
        var = np.array([float(o.noise_cov[0] if o.is_diagonal else o.noise_cov[0, 0]) for o in obs_list], dtype=float)
        return cell, tidx, val, var

    fit_cell, fit_tidx, y, fit_var = _index_obs(fit_obs)
    held_cell, held_tidx, held_y, held_var = _index_obs(held_obs)
    scale = float(np.sqrt(np.mean(fit_var)))  # Differential's Gaussian family takes one global noise scale
    coords = operator.coords

    def _forward(p, ops):
        transition_t = ops.tensor(transition)
        coords_t = ops.tensor(coords)
        lo_t = ops.tensor(box_lo)
        span_t = ops.tensor(box_hi - box_lo)
        frac = 1.0 / (1.0 + ops.exp(-p.location))
        loc_phys = lo_t + span_t * frac
        diff = coords_t - loc_phys
        dist2 = ops.sum(diff * diff, axis=1)
        weight = ops.exp(-dist2 / (2.0 * sigma * sigma))
        weight = weight / ops.sum(weight)
        eps = max(dt * 0.5, 1.0e-6)
        state = ops.zeros(operator.n)
        history = [state]
        for t_next in times[1:]:
            active = ops.heaviside(float(t_next) - p.onset, eps=eps)
            source = p.rate * active * weight
            state = ops.matmul(transition_t, state + dt * source)
            history.append(state)
        return ops.stack(history, axis=0)

    def _observe(solution, p, ops):
        return solution[fit_tidx, fit_cell]

    location = free(ndim, name="location", support="real")
    rate = free(1, name="rate", support="positive")
    onset = free(1, name="onset", support="real")

    diff = Differential(
        y,
        forward=_forward,
        observe=_observe,
        drivers=[location, rate, onset],
        scale=scale,
    )
    fitted = joint([diff]).fit(how=how, max_iter=800)

    _, onset_sd = fitted.posterior("onset")
    onset_mean = float(fitted.mean("onset"))

    rng = np.random.default_rng(seed)

    def _location_to_box(loc_unconstrained: np.ndarray) -> np.ndarray:
        frac = 1.0 / (1.0 + np.exp(-loc_unconstrained))
        return box_lo + (box_hi - box_lo) * frac

    marginal_draws = fitted.sample(n_location_draws, rng, nodes=["location", "rate"])
    loc_phys_draws = _location_to_box(marginal_draws["location"])
    rate_draws = marginal_draws["rate"]
    combined = np.concatenate([loc_phys_draws, rate_draws[:, None]], axis=1)
    raw_mean = combined.mean(axis=0)
    raw_cov = np.atleast_2d(np.cov(combined, rowvar=False))

    pred_draws = fitted.sample(n_predictive_draws, rng, nodes=["location", "rate", "onset"])
    pred_loc = _location_to_box(pred_draws["location"])
    pred_rate = pred_draws["rate"]
    pred_onset = pred_draws["onset"]
    pred_matrix = np.empty((n_predictive_draws, held_cell.size))
    for i in range(n_predictive_draws):
        sol = _simulate_numpy(pred_loc[i], pred_rate[i], pred_onset[i], operator, transition, times, sigma)
        pred_matrix[i] = sol[held_tidx, held_cell]
    pred_mean = pred_matrix.mean(axis=0)
    pred_std = np.sqrt(pred_matrix.var(axis=0) + held_var)

    z = (held_y - pred_mean) / pred_std
    inflation = float(np.sqrt(np.mean(z**2)))
    n_h = z.size
    rank = min(n_h, int(np.ceil((n_h + 1) * (1.0 - alpha))))
    conformal_quantile = float(np.sort(np.abs(z))[rank - 1])
    recalibration = SourceRecalibration(inflation=inflation, conformal_quantile=conformal_quantile, alpha=alpha)

    return SourcePosterior(
        mean=raw_mean,
        cov=raw_cov * (inflation**2),
        recalibration=recalibration,
        onset=(onset_mean, float(onset_sd)),
        held_out_prediction=HeldOutPrediction(observed=held_y, predictive_mean=pred_mean, predictive_std=pred_std),
    )
