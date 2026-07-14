"""Groundwater contaminant transport and Bayesian source inversion (workstream G, card G2).

**Scope note.** The work plan's G1 card ("Groundwater flow + reactive solute-transport operator")
owns a full :class:`GroundwaterTransportOperator` built on a Darcy-flow velocity solve
(``darcy_velocity``, via ``multiphysics.solve_poisson``) plus an Ogata-Banks analytic-plume helper,
with its own dedicated test suite. As of this change G1 had not landed on ``release/0.8.0`` (there was
no ``mixle_pde/groundwater.py`` and no merged ``work/g1-*`` branch), yet this card (G2, "Contaminant
source inversion") is specified to depend on it. G2's own inversion only ever needs a transport
operator with a *given, fixed* velocity field -- it never calls ``darcy_velocity`` -- so rather than
block on G1 landing, this module ships the narrow slice of the operator G2 actually consumes: a
Kronecker-sum generalization of :func:`mixle_pde.dynamics.laplacian_matrix` /
:func:`mixle_pde.dynamics.upwind_gradient_matrix` to an n-axis structured grid, with retardation and
first-order decay. It matches G1's frozen constructor signature exactly and registers under the same
``"groundwater"`` dynamics-operator name, so when a fuller G1 implementation (the Darcy solve, the
Ogata-Banks helper) lands it is additive on top of this file, not a rewrite of it.

The inversion itself (:func:`invert_source`) recovers a posterior over an unknown contaminant
source's ``(location, rate, onset time)`` from noisy concentration time series at a handful of
monitoring wells -- exactly the "contaminant source through downstream concentrations" example in
:mod:`mixle_pde.inverse`'s own module docstring. It is built on:

* :class:`mixle_pde.inverse.Differential` -- wraps the transport PDE as an observation whose forward
  model is the time-stepped solve, fit via ``mixle.ppl.field.joint([...]).fit(how="laplace")``.
* A domain-box reparameterization for the source location: ``free(ndim, support="real")`` declares an
  *unconstrained* latent, which the forward model squashes through a sigmoid and affine-maps onto the
  operator's domain box. ``mixle.ppl.field`` recognizes only ``support in {"real", "positive"}`` on
  this Laplace/Differential fitting path -- there is no native "box" support -- so ``loc_support="box"``
  is realized as this reparameterization rather than inventing a new support kind on that frozen surface.
* A chi-square variance-inflation + split-conformal recalibration on a held-out subset of the wells, in
  the spirit of workstream A3's ``posterior_calibration.recalibrate`` recipe (chi-square inflation plus
  a split-conformal quantile from held-out residuals). A3's ``recalibrate`` has since landed on
  ``release/0.8.0``, but it is typed against :class:`mixle_pde.latent.PosteriorField3D` and
  :class:`mixle_pde.observations.ForwardOperatorRegistry` -- it recalibrates the dense 3-D spatial
  field-inversion posterior through each observation kind's *linear* ``local_jacobian``. The
  ``Differential``/``joint(...).fit("laplace")`` path used here instead produces a low-dimensional
  (location, rate, onset) driver posterior from a genuinely nonlinear forward model, so A3's function
  cannot be called directly against it. The same algorithm (chi-square inflation, split-conformal
  quantile from a held-out split) is therefore implemented directly against that driver posterior below.

The returned posterior structurally satisfies the frozen IC-1 ``Posterior`` protocol
(``mixle/mixle/reason/posterior_protocol.py``, work-plan contracts.md): ``.samples(n, rng)``, ``.mean``,
``.cov``, ``.credible_interval(level)``, ``.derived_quantity(fn, n, rng)``. That module lives in the
core ``mixle`` distribution and had not landed as of this change either; since editing core ``mixle``
is outside this (mixle-pde-only) task's repo scope, :class:`SourcePosterior` duck-types the protocol
exactly rather than importing and asserting ``isinstance`` against it.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from mixle_pde.dynamics import DynamicsOperator, laplacian_matrix, register_dynamics_operator, upwind_gradient_matrix
from mixle_pde.observations import Observation

__all__ = [
    "GroundwaterTransportOperator",
    "SourceRecalibration",
    "HeldOutPrediction",
    "SourcePosterior",
    "simulate_wells",
    "invert_source",
]


class GroundwaterTransportOperator(DynamicsOperator):
    """Advection-dispersion-decay-retardation solute transport on a structured n-axis grid.

    ``du/dt = -(1/R)[v . grad u] + div(D grad u) - lambda u``, discretized by embedding the 1-D
    :func:`~mixle_pde.dynamics.laplacian_matrix` / :func:`~mixle_pde.dynamics.upwind_gradient_matrix`
    stencils for each axis into the full grid via a Kronecker sum (the standard construction for a
    separable operator on a structured grid: axis ``i``'s 1-D operator embedded as
    ``I x ... x M_i x ... x I``, identity on every other axis). ``shape`` is the grid's per-axis point
    count (e.g. ``(nx, ny)`` for a 2-D aquifer slice); ``velocity_field`` is a per-axis (uniform-in-
    space) Darcy velocity; ``dispersivity`` is a per-axis dispersivity, scaled by the flow *speed*
    (``|velocity_field|``) into a per-axis dispersion coefficient, plus a small molecular-diffusion
    floor so a zero-velocity axis still disperses.

    A spatially-varying velocity field (the general Darcy-flow case) is G1's remit, not implemented
    here -- see the module scope note.
    """

    def __init__(
        self,
        velocity_field,
        dispersivity,
        shape,
        *,
        retardation: float = 1.0,
        decay: float = 0.0,
        spacing: float = 1.0,
        bc: str = "neumann",
        scheme: str = "implicit",
    ) -> None:
        if scheme not in ("implicit", "explicit", "exact"):
            raise ValueError("scheme must be 'implicit', 'explicit', or 'exact'.")
        self.shape = tuple(int(s) for s in shape)
        if len(self.shape) == 0:
            raise ValueError("shape must have at least one axis.")
        if any(s < 3 for s in self.shape):
            raise ValueError("each grid axis needs at least 3 points.")
        ndim = len(self.shape)

        self.bc = bc
        self.scheme = scheme
        self.retardation = float(retardation)
        if self.retardation <= 0.0:
            raise ValueError("retardation must be strictly positive.")
        self.decay = float(decay)
        self.n = int(np.prod(self.shape))

        self.spacing = np.broadcast_to(np.asarray(spacing, dtype=float), (ndim,)).copy()
        self.velocity = np.broadcast_to(np.asarray(velocity_field, dtype=float), (ndim,)).copy()
        alpha = np.broadcast_to(np.asarray(dispersivity, dtype=float), (ndim,)).copy()
        speed = float(np.linalg.norm(self.velocity))
        molecular_floor = 1.0e-6
        self.diffusion = alpha * speed + molecular_floor

        axes = [np.arange(s) * self.spacing[a] for a, s in enumerate(self.shape)]
        mesh = np.meshgrid(*axes, indexing="ij")
        self.coords = np.stack([m.reshape(-1) for m in mesh], axis=1)  # (n, ndim), C-order matches kron

        self._lap = [laplacian_matrix(s, self.spacing[a], bc) for a, s in enumerate(self.shape)]
        self._grad = [
            upwind_gradient_matrix(s, self.spacing[a], self.velocity[a], bc) for a, s in enumerate(self.shape)
        ]

    def _embed(self, mats: list[np.ndarray], axis: int) -> np.ndarray:
        """Kronecker-embed a per-axis 1-D matrix into the full grid operator (identity on other axes)."""
        out = mats[axis] if axis == 0 else np.eye(self.shape[0])
        for a in range(1, len(self.shape)):
            factor = mats[axis] if a == axis else np.eye(self.shape[a])
            out = np.kron(out, factor)
        return out

    @property
    def domain_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """The physical ``(lo, hi)`` box the grid covers, one entry per axis."""
        hi = np.array([(s - 1) * self.spacing[a] for a, s in enumerate(self.shape)])
        return np.zeros_like(hi), hi

    def nearest_index(self, point) -> int:
        """The flat cell index whose centre is closest to ``point`` (an ``(ndim,)``-ish coordinate)."""
        point = np.asarray(point, dtype=float).reshape(-1)[: self.coords.shape[1]]
        d2 = np.sum((self.coords - point) ** 2, axis=1)
        return int(np.argmin(d2))

    def operator_matrix(self) -> np.ndarray:
        g = np.zeros((self.n, self.n), dtype=float)
        for axis in range(len(self.shape)):
            g += self.diffusion[axis] * self._embed(self._lap, axis)
            g -= (self.velocity[axis] / self.retardation) * self._embed(self._grad, axis)
        g -= self.decay * np.eye(self.n)
        return g


register_dynamics_operator("groundwater", GroundwaterTransportOperator)


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
