"""G3 -- air-quality / dust dispersion + source apportionment (work-plan workstream G).

Two complementary forward models for how an emission spreads through the air, plus an inverse
problem that turns receptor (monitor) readings back into per-source emission rates:

* :func:`gaussian_plume` -- the classic steady-state analytic plume for flat, open terrain: a
  point source's concentration falls off as a Gaussian in the crosswind (``y``) and vertical
  (``z``) directions around the downwind (``x``) centerline, with a ground-reflection image
  source so the plume does not diffuse through the ground. The plume's spread rates
  ``sigma_y(x)``/``sigma_z(x)`` are the empirical Pasquill-Gifford stability-class curves, using
  the widely tabulated Briggs (1973) rural power-law fit to the original nomograph (Briggs, G.A.,
  "Diffusion Estimation for Small Emissions", ATDL-106, 1973; reproduced e.g. in Arya, *Air
  Pollution Meteorology and Dispersion* (1999), Table 6.1, and in most introductory air-quality
  texts, e.g. Wark, Warner & Davis, *Air Pollution: Its Origin and Control*). This is the
  right-hand-side "textbook" formula the module docstring and DoD test check against; it is
  cheap and exact when its assumptions (steady wind, uniform terrain, no chemistry) hold.
* :class:`DispersionOperator` -- a thin, domain-named subclass of
  :class:`~mixle_pde.dynamics.AdvectionDiffusionOperator` for the gridded/obstacle-affected case
  where the analytic plume's flat-terrain assumption breaks down: the wind vector becomes the
  advection velocity and a turbulent (eddy) diffusivity becomes the dispersion coefficient, so
  the same method-of-lines solver stack (:mod:`mixle_pde.dynamics`) advances concentration
  through a numerically resolved domain instead.
* :func:`apportion_sources` -- source apportionment: given concentration readings at a set of
  receptors, candidate source locations, and the prevailing wind, build the linear
  source -> receptor transfer matrix by superposing unit-rate plumes, then solve for the
  non-negative per-source emission rates by NNLS and wrap the fit as an IC-1 ``Posterior`` (a
  Gaussian centered at the NNLS solution, with covariance the inverse Fisher information of the
  linearized weighted least-squares problem there).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from scipy.optimize import nnls
from scipy.stats import norm

from mixle_pde.dynamics import AdvectionDiffusionOperator, register_dynamics_operator
from mixle_pde.observations import Observation

__all__ = ["gaussian_plume", "DispersionOperator", "apportion_sources"]

# ---------------------------------------------------------------------------
# Pasquill-Gifford stability-class dispersion coefficients (Briggs 1973, rural)
# ---------------------------------------------------------------------------
# Each entry is (a, b, p) for sigma(x) = a * x * (1 + b * x) ** p  (x in metres); b == 0.0 means
# the plain power law sigma_z = a * x (classes A/B have no correction term in the Briggs fit).
_SIGMA_Y_COEFFS: dict[str, tuple[float, float, float]] = {
    "A": (0.22, 0.0001, -0.5),
    "B": (0.16, 0.0001, -0.5),
    "C": (0.11, 0.0001, -0.5),
    "D": (0.08, 0.0001, -0.5),
    "E": (0.06, 0.0001, -0.5),
    "F": (0.04, 0.0001, -0.5),
}
_SIGMA_Z_COEFFS: dict[str, tuple[float, float, float]] = {
    "A": (0.20, 0.0, 1.0),
    "B": (0.12, 0.0, 1.0),
    "C": (0.08, 0.0002, -0.5),
    "D": (0.06, 0.0015, -0.5),
    "E": (0.03, 0.0003, -1.0),
    "F": (0.016, 0.0003, -1.0),
}


def _normalize_stability(stability: str) -> str:
    key = str(stability).strip().upper()
    if key not in _SIGMA_Y_COEFFS:
        raise ValueError(f"stability must be one of the Pasquill classes 'A'-'F', got {stability!r}.")
    return key


def _sigma_y(x: np.ndarray, stability: str) -> np.ndarray:
    """Pasquill-Gifford crosswind spread ``sigma_y(x)`` (Briggs 1973 rural fit), ``x`` in metres."""
    a, b, p = _SIGMA_Y_COEFFS[stability]
    return a * x * (1.0 + b * x) ** p


def _sigma_z(x: np.ndarray, stability: str) -> np.ndarray:
    """Pasquill-Gifford vertical spread ``sigma_z(x)`` (Briggs 1973 rural fit), ``x`` in metres."""
    a, b, p = _SIGMA_Z_COEFFS[stability]
    if b == 0.0:
        return a * x
    return a * x * (1.0 + b * x) ** p


def gaussian_plume(
    Q: float,
    u: float,
    stability: str,
    x: Any,
    y: Any,
    z: Any,
    *,
    H: float = 0.0,
) -> np.ndarray:
    """Steady-state Gaussian plume concentration downwind of a continuous point source.

    ``Q`` is the emission rate (mass/time), ``u`` the mean wind speed (length/time) along the
    downwind axis, ``stability`` a Pasquill stability class (``'A'`` most unstable/convective
    through ``'F'`` most stable). ``x``/``y``/``z`` are downwind, crosswind, and vertical
    coordinates from the source (broadcastable arrays or scalars, same length units as ``u``'s
    denominator); ``H`` is the source (stack) release height above ground.

    Returns the concentration ``C(x, y, z) = Q / (2 pi u sigma_y sigma_z) * exp(-y^2 / (2
    sigma_y^2)) * [exp(-(z-H)^2 / (2 sigma_z^2)) + exp(-(z+H)^2 / (2 sigma_z^2))]``, where
    ``sigma_y(x)``/``sigma_z(x)`` are the Pasquill-Gifford spread rates (:func:`_sigma_y`,
    :func:`_sigma_z`) and the second exponential in the bracket is the ground-reflection image
    source (an idealized impermeable, non-absorbing ground at ``z = 0``). Points at or upwind of
    the source (``x <= 0``) get zero concentration -- the steady-plume model has no upwind
    diffusion. For ``H = 0`` and a ground-level receptor (``y = z = 0``) this collapses to the
    textbook ground-level centerline formula ``C = Q / (pi u sigma_y sigma_z)`` (the two
    reflection terms coincide and double the coefficient), the standard reference check for this
    function (see the module's DoD test).
    """
    stability = _normalize_stability(stability)
    Q = float(Q)
    u = float(u)
    if u <= 0.0:
        raise ValueError("u (wind speed) must be positive.")
    H = float(H)

    x_arr, y_arr, z_arr = np.broadcast_arrays(
        np.asarray(x, dtype=float), np.asarray(y, dtype=float), np.asarray(z, dtype=float)
    )
    downwind = x_arr > 0.0
    x_safe = np.where(downwind, x_arr, 1.0)  # placeholder for x<=0 so sigma(x) never divides by zero; masked below

    sig_y = _sigma_y(x_safe, stability)
    sig_z = _sigma_z(x_safe, stability)

    coeff = Q / (2.0 * np.pi * u * sig_y * sig_z)
    lateral = np.exp(-0.5 * (y_arr / sig_y) ** 2)
    vertical = np.exp(-0.5 * ((z_arr - H) / sig_z) ** 2) + np.exp(-0.5 * ((z_arr + H) / sig_z) ** 2)
    concentration = coeff * lateral * vertical
    return np.where(downwind, concentration, 0.0)


def _wind_speed(wind: Any) -> float:
    arr = np.atleast_1d(np.asarray(wind, dtype=float))
    return float(np.linalg.norm(arr)) if arr.size > 1 else float(arr[0])


class DispersionOperator(AdvectionDiffusionOperator):
    """Wind-advected advection-diffusion operator for gridded/obstacle-affected transport.

    A thin, domain-named adapter of :class:`~mixle_pde.dynamics.AdvectionDiffusionOperator` for
    the case :func:`gaussian_plume` cannot handle: near-field obstacles, non-uniform terrain, or
    anywhere the flat-terrain, steady-straight-line-wind assumption behind the analytic plume
    breaks down. ``wind`` is the advective velocity along the transport grid (a scalar speed, or
    a vector whose magnitude becomes the 1-D transport speed along the wind-aligned grid axis);
    ``turbulent_diffusivity`` is the isotropic eddy (turbulent) diffusivity standing in for the
    plume's ``sigma_y``/``sigma_z`` growth in this numerically resolved setting. Everything else
    (the method-of-lines discretisation, the ``transition_matrix`` time-stepping) is inherited
    unchanged from :class:`~mixle_pde.dynamics.AdvectionDiffusionOperator`.
    """

    def __init__(
        self,
        wind: Any,
        turbulent_diffusivity: float,
        n: int,
        length: float = 1.0,
        bc: str = "periodic",
        scheme: str = "implicit",
    ) -> None:
        self.wind = np.atleast_1d(np.asarray(wind, dtype=float))
        self.turbulent_diffusivity = float(turbulent_diffusivity)
        super().__init__(
            diffusivity=self.turbulent_diffusivity,
            velocity=_wind_speed(wind),
            n=n,
            length=length,
            bc=bc,
            scheme=scheme,
        )


register_dynamics_operator("dispersion", DispersionOperator)


# ---------------------------------------------------------------------------
# Source apportionment (linear source -> receptor transfer, NNLS)
# ---------------------------------------------------------------------------
def _flatten_receptors(receptor_obs: Sequence[Observation]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Concatenate a list of :class:`Observation` into ``(locations, values, variances)``."""
    if not receptor_obs:
        raise ValueError("apportion_sources needs at least one receptor observation.")
    locations, values, variances = [], [], []
    for obs in receptor_obs:
        locations.append(obs.location)
        values.append(obs.value)
        noise_cov = np.asarray(obs.noise_cov, dtype=float)
        variances.append(np.diag(noise_cov) if noise_cov.ndim == 2 else noise_cov)
    return np.concatenate(locations, axis=0), np.concatenate(values), np.concatenate(variances)


def _source_xyh(source: Any) -> tuple[float, float, float]:
    """Coerce one ``candidate_sources`` entry to ``(x, y, height)``.

    Accepts a mapping with a ``"location"`` key (``(x, y)`` or ``(x, y, z)``) and an optional
    ``"height"`` override, or a bare ``(x, y)`` / ``(x, y, height)`` array-like.
    """
    if isinstance(source, Mapping):
        loc = np.atleast_1d(np.asarray(source["location"], dtype=float))
        default_height = float(loc[2]) if loc.size > 2 else 0.0
        height = float(source.get("height", default_height))
        return float(loc[0]), float(loc[1]), height
    arr = np.atleast_1d(np.asarray(source, dtype=float))
    height = float(arr[2]) if arr.size > 2 else 0.0
    return float(arr[0]), float(arr[1]), height


def _wind_params(wind: Any) -> tuple[float, str, float]:
    """Coerce ``wind`` to ``(speed, stability, direction_degrees)``.

    Accepts a mapping with ``"speed"``/``"stability"``/optional ``"direction"`` keys, or a bare
    ``(speed, stability)`` / ``(speed, stability, direction)`` sequence. ``direction`` is the
    azimuth (degrees, measured counter-clockwise from the +x axis) the wind blows *toward*;
    it defaults to ``0.0`` (wind blowing in the +x direction, i.e. the plume's own downwind axis
    already lines up with the world x-axis).
    """
    if isinstance(wind, Mapping):
        speed = float(wind["speed"])
        stability = _normalize_stability(wind["stability"])
        direction = float(wind.get("direction", 0.0))
    else:
        speed = float(wind[0])
        stability = _normalize_stability(wind[1])
        direction = float(wind[2]) if len(wind) > 2 else 0.0
    return speed, stability, direction


def _transfer_matrix(
    receptor_xyz: np.ndarray,
    sources: Sequence[tuple[float, float, float]],
    *,
    speed: float,
    stability: str,
    direction: float,
) -> np.ndarray:
    """Build the ``(n_receptors, n_sources)`` unit-rate source -> receptor transfer matrix.

    Column ``j`` is :func:`gaussian_plume` evaluated at every receptor in source ``j``'s own
    downwind/crosswind frame (rotated by the wind ``direction``), at unit emission rate ``Q=1``
    -- superposition then makes the observed concentration ``A @ r`` for a rate vector ``r``.
    """
    theta = np.radians(direction)
    downwind_hat = np.array([np.cos(theta), np.sin(theta)])
    crosswind_hat = np.array([-np.sin(theta), np.cos(theta)])
    columns = []
    for sx, sy, sh in sources:
        dx = receptor_xyz[:, 0] - sx
        dy = receptor_xyz[:, 1] - sy
        x_prime = dx * downwind_hat[0] + dy * downwind_hat[1]
        y_prime = dx * crosswind_hat[0] + dy * crosswind_hat[1]
        columns.append(gaussian_plume(1.0, speed, stability, x_prime, y_prime, receptor_xyz[:, 2], H=sh))
    return np.stack(columns, axis=1)


class _RatePosterior:
    """IC-1 ``Posterior`` over non-negative per-source emission rates.

    A Gaussian approximation centered at the NNLS solution, with covariance the inverse Fisher
    information of the linearized (noise-weighted) least-squares problem there -- i.e. the
    NNLS optimum is treated as a local Gauss-Newton MAP and its curvature read off the same way
    a Laplace approximation would, without paying for a full nonlinear fit. A small ridge is
    added to the Fisher information before inverting so a source with (near-)zero sensitivity at
    every receptor gets a large-but-finite variance instead of a numerically singular one.
    Samples/intervals are clipped to the physical ``r >= 0`` half-space.
    """

    def __init__(self, mean: np.ndarray, cov: np.ndarray) -> None:
        self._mean = np.asarray(mean, dtype=float)
        self._cov = np.asarray(cov, dtype=float)

    def samples(self, n: int, rng: np.random.Generator) -> np.ndarray:
        draws = rng.multivariate_normal(self._mean, self._cov, size=int(n))
        return np.clip(draws, 0.0, None)

    @property
    def mean(self) -> np.ndarray:
        return self._mean

    @property
    def cov(self) -> np.ndarray:
        return self._cov

    def credible_interval(self, level: float) -> tuple[np.ndarray, np.ndarray]:
        z = float(norm.ppf(0.5 + level / 2.0))
        sd = np.sqrt(np.clip(np.diag(self._cov), 0.0, None))
        lo = np.clip(self._mean - z * sd, 0.0, None)
        hi = self._mean + z * sd
        return lo, hi

    def derived_quantity(self, fn: Any, n: int, rng: np.random.Generator) -> _SimpleDerivedQuantity:
        return _SimpleDerivedQuantity(fn(self.samples(n, rng)))


class _SimpleDerivedQuantity:
    """Minimal IC-1 ``DerivedQuantity``: samples + credible interval + the honesty flag."""

    def __init__(self, samples: Any) -> None:
        self.samples = np.asarray(samples)
        self.prior_dominated = False

    def credible_interval(self, level: float) -> tuple[np.ndarray, np.ndarray]:
        a = (1.0 - level) / 2.0
        return np.quantile(self.samples, a, axis=0), np.quantile(self.samples, 1 - a, axis=0)


def apportion_sources(
    receptor_obs: Sequence[Observation],
    candidate_sources: Sequence[Any],
    wind: Any,
) -> _RatePosterior:
    """Apportion receptor concentrations among candidate sources; a posterior over emission rates.

    ``receptor_obs`` is a list of :class:`~mixle_pde.observations.Observation` (each with
    ``location``/``value``/``noise_cov``; concatenated if more than one is given -- e.g. one per
    monitoring campaign). ``candidate_sources`` is a sequence of source descriptors (see
    :func:`_source_xyh`: a ``{"location": (x, y[, z]), "height": h}`` mapping or a bare ``(x, y[,
    height])`` array-like). ``wind`` is the prevailing wind (see :func:`_wind_params`: a
    ``{"speed", "stability", "direction"}`` mapping or a ``(speed, stability[, direction])``
    sequence).

    Builds the linear source -> receptor transfer matrix ``A`` (one column per candidate source,
    :func:`_transfer_matrix`) by superposing unit-rate :func:`gaussian_plume` evaluations, solves
    the noise-weighted non-negative least squares problem ``min ||W(A r - c)||, r >= 0`` (``W`` the
    per-receptor inverse-noise-standard-deviation weights), and returns the fit wrapped as an IC-1
    ``Posterior`` over ``r`` (:class:`_RatePosterior`): a Gaussian centered at the NNLS solution
    with the linearized (weighted Fisher-information) covariance, so ``.mean`` are the point rate
    estimates and ``.credible_interval(level)`` gives calibrated-width per-source rate intervals.
    """
    locations, values, variances = _flatten_receptors(receptor_obs)
    speed, stability, direction = _wind_params(wind)
    sources = [_source_xyh(s) for s in candidate_sources]
    if not sources:
        raise ValueError("apportion_sources needs at least one candidate source.")

    transfer = _transfer_matrix(locations, sources, speed=speed, stability=stability, direction=direction)
    weights = 1.0 / np.sqrt(np.clip(variances, 1e-12, None))
    weighted_transfer = transfer * weights[:, None]
    weighted_values = values * weights

    rate_hat, _residual_norm = nnls(weighted_transfer, weighted_values)

    fisher = weighted_transfer.T @ weighted_transfer
    ridge = 1e-8 * max(float(np.trace(fisher)) / max(fisher.shape[0], 1), 1e-6)
    cov = np.linalg.inv(fisher + ridge * np.eye(fisher.shape[0]))

    return _RatePosterior(rate_hat, cov)
