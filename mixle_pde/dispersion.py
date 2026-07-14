"""Air-quality / dust dispersion, source apportionment, and occupational exposure (workstream G/K).

Two cards share this module:

* **G3** ("Air-quality / dust dispersion + source apportionment") owns two complementary forward
  models for how an emission spreads through the outdoor air, plus an inverse problem that turns
  receptor (monitor) readings back into per-source emission rates:

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

* **K1** ("Occupational exposure transport") reproduces the concentration a worker breathes at a
  face/working position once a workplace's mechanical ventilation is accounted for: an
  advection-diffusion-deposition transport of an airborne contaminant (dust, fume, aerosol) from an
  emission :class:`SourceTerm` at the working face, diluted by fresh-air ventilation
  (:class:`VentilationBC`) and lost to first-order deposition/settling, resolved to a steady
  :class:`ConcentrationField` over a 1-D workplace :class:`Mesh`.

  **Scope note.** K1 was implemented while G3 was still an open, unmerged PR against this same base
  branch, so ``mixle_pde/dispersion.py`` (the file the work order describes K1 as an *addition* to)
  did not exist yet. Rather than speculatively duplicate G3's not-yet-frozen code, K1 built directly
  on the already-merged :class:`mixle_pde.dynamics.AdvectionDiffusionOperator` building blocks
  (:func:`~mixle_pde.dynamics.laplacian_matrix`, :func:`~mixle_pde.dynamics.upwind_gradient_matrix`)
  -- exactly what G3's own :class:`DispersionOperator` is itself a thin, domain-named subclass of --
  adding only the five names in its own ``__all__`` slice (:class:`Mesh`, :class:`SourceTerm`,
  :class:`VentilationBC`, :class:`ConcentrationField`, :func:`occupational_exposure`), disjoint from
  G3's ``gaussian_plume``/``DispersionOperator``/``apportion_sources``. Landing G3 first (as here)
  makes folding K1 in a content-preserving merge, not a redesign, per K1's own design intent.

  The workplace domain is a single 1-D control-volume line along the dominant ventilation axis
  from the working face (``x = 0``) to the exhaust/return (``x = length``) -- the standard
  "box-model" reduction used throughout industrial-hygiene ventilation practice (e.g. the ACGIH
  *Industrial Ventilation* manual's single-zone dilution model) when full ventilation-network CFD
  is out of scope (K1's non-goals). The steady-state balance solved at every interior node is

      D * d^2C/dx^2 - u * dC/dx - k * C = 0

  (``u`` = ventilation face velocity, ``D`` = turbulent eddy diffusivity, ``k`` = the species'
  first-order deposition-rate constant), closed with Danckwerts boundary conditions: a flux inlet
  at each of ``ventilation.inflow_faces`` that balances the incoming fresh/recirculated air against
  the :class:`SourceTerm` emission, and a zero-gradient (pure convective) outlet at the domain's far
  node, which stands in for the exhaust/return grille.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.optimize import nnls
from scipy.stats import norm

from mixle_pde.dynamics import (
    AdvectionDiffusionOperator,
    laplacian_matrix,
    register_dynamics_operator,
    upwind_gradient_matrix,
)
from mixle_pde.observations import Observation

__all__ = [
    "gaussian_plume",
    "DispersionOperator",
    "apportion_sources",
    "Mesh",
    "SourceTerm",
    "VentilationBC",
    "ConcentrationField",
    "occupational_exposure",
]

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


# ---------------------------------------------------------------------------
# Occupational exposure transport (workstream K1)
# ---------------------------------------------------------------------------
# Species -> first-order deposition/decay rate constant (s^-1), order-of-magnitude indoor
# particle-deposition literature values (larger/denser particle classes settle faster).
# Unlisted species fall back to _DEFAULT_DEPOSITION_RATE_S.
DEPOSITION_RATE_S: dict[str, float] = {
    "silica_pm4": 5.0e-4,  # respirable crystalline silica, PM4 (thoracic-like) convention
    "coal_dust_pm10": 1.5e-3,
    "diesel_pm": 2.0e-4,
    "welding_fume": 8.0e-4,
}
_DEFAULT_DEPOSITION_RATE_S = 5.0e-4

# Prandtl mixing-length heuristic: turbulent (eddy) diffusivity ~ ventilation speed * a mixing
# length taken as this fraction of the domain's ventilation-axis length. The occupational_exposure
# public API (frozen by the work order) has no explicit diffusivity argument, so it is derived
# from the ventilation intensity + workplace scale rather than left unconstrained.
_MIXING_LENGTH_FRACTION = 0.1


@dataclass(frozen=True)
class Mesh:
    """A 1-D workplace control-volume grid: ``n_nodes`` points spanning ``[0, length]`` along the
    ventilation flow axis, working face at ``x = 0`` to exhaust/return at ``x = length``.
    """

    n_nodes: int
    length: float
    cross_section_area: float = 1.0

    def __post_init__(self) -> None:
        if self.n_nodes < 3:
            raise ValueError("Mesh needs at least 3 grid points.")
        if self.length <= 0.0:
            raise ValueError("Mesh length must be positive.")
        if self.cross_section_area <= 0.0:
            raise ValueError("Mesh cross_section_area must be positive.")

    @property
    def h(self) -> float:
        """Uniform node spacing."""
        return self.length / (self.n_nodes - 1)

    def grid(self) -> np.ndarray:
        """Node positions, shape ``(n_nodes,)``."""
        return np.linspace(0.0, self.length, self.n_nodes)


@dataclass
class SourceTerm:
    """A localized workplace emission source: ``rate`` (mass/time) of ``species`` injected at grid
    node ``location`` (default ``0`` -- the working face).

    ``rate`` is normally a plain float, but may instead be an IC-1 ``Posterior``-shaped object
    (anything exposing ``.mean`` and ``.cov``) over the emission rate; :func:`occupational_exposure`
    then propagates that source-rate uncertainty into a per-cell :class:`ConcentrationField`
    variance (the concentration solve is linear in the emission rate, so this propagation is exact,
    not sampled).
    """

    rate: Any
    species: str = "silica_pm4"
    location: int = 0


@dataclass(frozen=True)
class VentilationBC:
    """Fresh-air ventilation boundary condition.

    ``inflow_faces`` names the grid node(s) where ventilation air enters (fresh-air dilution at
    face velocity ``airflow_mps``); ``recirc`` is the fraction of the exhaust concentration mixed
    back into that inflow air (``0`` = once-through fresh air, ``1`` = fully recirculated).
    """

    inflow_faces: np.ndarray
    airflow_mps: float
    recirc: float = 0.0

    def __post_init__(self) -> None:
        faces = np.atleast_1d(np.asarray(self.inflow_faces, dtype=int))
        object.__setattr__(self, "inflow_faces", faces)
        if self.airflow_mps <= 0.0:
            raise ValueError("airflow_mps must be positive.")
        if not (0.0 <= self.recirc <= 1.0):
            raise ValueError("recirc must lie in [0, 1].")


@dataclass
class ConcentrationField:
    """The result of :func:`occupational_exposure`: an IC-1 ``Posterior``-shaped concentration
    field over ``mesh``'s grid nodes -- ``mean`` (and optional ``variance``) is the ``(n_cells,)``
    per-cell contaminant concentration.
    """

    mean: np.ndarray
    mesh: Mesh
    variance: np.ndarray | None = None
    species: str = "silica_pm4"

    @property
    def std(self) -> np.ndarray | None:
        return None if self.variance is None else np.sqrt(np.clip(self.variance, 0.0, None))

    @property
    def cov(self) -> np.ndarray:
        """Diagonal covariance (per-cell concentration variance is independent by construction)."""
        n = self.mean.shape[0]
        if self.variance is None:
            return np.zeros((n, n))
        return np.diag(self.variance)

    def samples(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """IC-1 ``Posterior.samples``: ``n`` draws, shape ``(n, n_cells)``."""
        if self.variance is None:
            return np.broadcast_to(self.mean, (int(n),) + self.mean.shape).copy()
        sd = np.sqrt(np.clip(self.variance, 0.0, None))
        return self.mean + sd * rng.standard_normal((int(n),) + self.mean.shape)

    def credible_interval(self, level: float) -> tuple[np.ndarray, np.ndarray]:
        """IC-1 ``Posterior.credible_interval``: per-cell central ``level`` interval."""
        if self.variance is None:
            return self.mean.copy(), self.mean.copy()
        z = float(norm.ppf(0.5 + level / 2.0))
        sd = np.sqrt(np.clip(self.variance, 0.0, None))
        return self.mean - z * sd, self.mean + z * sd

    def derived_quantity(self, fn: Any, n: int, rng: np.random.Generator) -> _FieldDerivedQuantity:
        """IC-1 ``Posterior.derived_quantity``: pushforward ``fn`` over ``n`` field draws."""
        return _FieldDerivedQuantity(fn(self.samples(n, rng)))


@dataclass
class _FieldDerivedQuantity:
    """Minimal IC-1 ``DerivedQuantity``: samples + credible interval + the honesty flag.

    ``prior_dominated`` is always ``False`` here: the propagated width (when present) comes from
    the source-rate uncertainty supplied by the caller, never from an implicit regularizer/prior.
    """

    samples: np.ndarray
    prior_dominated: bool = field(default=False)

    def __post_init__(self) -> None:
        self.samples = np.asarray(self.samples)

    def credible_interval(self, level: float) -> tuple[np.ndarray, np.ndarray]:
        a = (1.0 - level) / 2.0
        return np.quantile(self.samples, a, axis=0), np.quantile(self.samples, 1 - a, axis=0)


def _deposition_rate(species: str) -> float:
    return DEPOSITION_RATE_S.get(species, _DEFAULT_DEPOSITION_RATE_S)


def _is_posterior_like(x: Any) -> bool:
    return hasattr(x, "mean") and hasattr(x, "cov") and not isinstance(x, (int, float, np.floating, np.integer))


def _scalar_mean_var(rate: Any) -> tuple[float, float | None]:
    """Coerce ``SourceTerm.rate`` to ``(mean, variance)``; ``variance`` is ``None`` for a bare
    scalar rate (no uncertainty to propagate)."""
    if not _is_posterior_like(rate):
        return float(rate), None
    mean = np.asarray(rate.mean, dtype=float).reshape(-1)
    cov = rate.cov
    if hasattr(cov, "matvec"):  # a scipy LinearOperator (IC-1 survey-scale case)
        var = (
            float(cov.matvec(np.ones(1))[0])
            if mean.size == 1
            else float(np.asarray(cov.matvec(np.eye(mean.size))).diagonal()[0])
        )
    else:
        cov_arr = np.atleast_2d(np.asarray(cov, dtype=float))
        var = float(cov_arr[0, 0])
    return float(mean[0]), var


def _assemble_transport_system(
    mesh: Mesh,
    ventilation: VentilationBC,
    *,
    deposition_rate: float,
    diffusivity: float,
    source_location: int,
    unit_rate: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the ``(n, n)`` steady advection-diffusion-deposition system ``M @ C = b``.

    The interior stencil reuses :mod:`mixle_pde.dynamics`'s own building blocks -- the same
    ``laplacian_matrix``/``upwind_gradient_matrix`` :class:`~mixle_pde.dynamics.AdvectionDiffusionOperator`
    composes into ``D C'' - u C'`` -- with a ``-k`` deposition sink added to the diagonal. Neither
    of ``AdvectionDiffusionOperator``'s built-in boundary treatments (``'dirichlet'``, ``'neumann'``,
    ``'periodic'``) expresses a Robin/Danckwerts flux condition, so the rows at
    ``ventilation.inflow_faces`` and the domain's far node are overwritten in place: each inflow row
    becomes a Danckwerts flux-inlet balance (fresh/recirculated-air dilution, plus the source flux
    if the emission is co-located there), and the far node becomes a Danckwerts zero-gradient
    outlet (the exhaust/return). ``unit_rate`` lets callers solve once at ``rate=1`` and rescale
    linearly (the system's matrix does not depend on the emission rate, only ``b`` does), which is
    how source-rate uncertainty is propagated exactly in :func:`occupational_exposure`.
    """
    n = mesh.n_nodes
    h = mesh.h
    u = float(ventilation.airflow_mps)
    d = float(diffusivity)
    k = float(deposition_rate)
    area = mesh.cross_section_area
    exhaust_idx = n - 1

    inflow = set(int(i) for i in ventilation.inflow_faces)
    for i in inflow:
        if not 0 <= i < n:
            raise ValueError(f"inflow_faces index {i} out of range for a {n}-node Mesh.")
    if exhaust_idx in inflow:
        raise ValueError("inflow_faces must not include the domain's far (exhaust/return) node.")
    if not 0 <= source_location < n:
        raise ValueError(f"SourceTerm.location {source_location} out of range for a {n}-node Mesh.")

    # Interior stencil: reuse dynamics.py's discretization building blocks directly, then add the
    # first-order deposition sink. `bc` is irrelevant here -- the boundary rows it produces are
    # replaced below -- so 'neumann' (dynamics.py's own default) is passed through unchanged.
    m = d * laplacian_matrix(n, h, bc="neumann") - u * upwind_gradient_matrix(n, h, u, bc="neumann")
    m -= k * np.eye(n)
    b = np.zeros(n)
    if 0 <= source_location < n and source_location not in inflow and source_location != exhaust_idx:
        b[source_location] += unit_rate / (area * h)

    for i in inflow:
        # Danckwerts inlet: convective + dispersive flux in balances the incoming (possibly
        # recirculated) air, plus the source's flux if it is co-located with this face.
        m[i, :] = 0.0
        m[i, i] += u + d / h
        if i + 1 < n:
            m[i, i + 1] -= d / h
        if ventilation.recirc > 0.0:
            m[i, exhaust_idx] -= u * ventilation.recirc
        b[i] = unit_rate / area if i == source_location else 0.0

    if exhaust_idx not in inflow:
        # Danckwerts outlet: zero concentration gradient (pure convective outflow).
        m[exhaust_idx, :] = 0.0
        m[exhaust_idx, exhaust_idx] += 1.0 / h
        m[exhaust_idx, exhaust_idx - 1] -= 1.0 / h
        b[exhaust_idx] = unit_rate / (area * h) if exhaust_idx == source_location else 0.0

    return m, b


def occupational_exposure(
    source: SourceTerm,
    geometry: Mesh,
    *,
    ventilation: VentilationBC,
    species: str = "silica_pm4",
    steady: bool = True,
) -> ConcentrationField:
    """Solve the steady occupational concentration field for a face source under ventilation.

    1. Builds the advection-diffusion-deposition transport system on the workplace ``geometry``
       (turbulent diffusivity derived from ``ventilation.airflow_mps`` by a mixing-length
       heuristic; see :data:`_MIXING_LENGTH_FRACTION`).
    2. Imposes ``ventilation`` as a Danckwerts flux-inlet (Robin-type) condition at
       ``ventilation.inflow_faces`` -- fresh-air dilution at ``airflow_mps``, ``recirc`` fraction
       of the exhaust concentration mixed back in.
    3. Places ``source`` (emission rate + particle class) at its ``location`` node (the working
       face by default).
    4. Solves the steady advection-diffusion-deposition balance directly (species-specific
       first-order deposition/settling sink, :data:`DEPOSITION_RATE_S`).
    5. Returns a :class:`ConcentrationField`; when ``source.rate`` is an IC-1 ``Posterior``-shaped
       object, its uncertainty is propagated to a per-cell variance (the solve is linear in the
       emission rate, so this is exact, not a Monte Carlo approximation).
    """
    if not steady:
        raise NotImplementedError(
            "occupational_exposure only supports the steady-state solve (transient transport is a K1 non-goal)."
        )

    diffusivity = ventilation.airflow_mps * _MIXING_LENGTH_FRACTION * geometry.length
    k_dep = _deposition_rate(species if species else source.species)

    rate_mean, rate_var = _scalar_mean_var(source.rate)

    m, b_unit = _assemble_transport_system(
        geometry,
        ventilation,
        deposition_rate=k_dep,
        diffusivity=diffusivity,
        source_location=source.location,
        unit_rate=1.0,
    )
    c_unit = np.linalg.solve(m, b_unit)

    mean = c_unit * rate_mean
    variance = None if rate_var is None else (c_unit**2) * rate_var

    return ConcentrationField(mean=mean, mesh=geometry, variance=variance, species=species)
