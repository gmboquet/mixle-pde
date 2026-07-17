"""G6 -- monitoring-network design / sensor placement (work-plan workstream G).

Chooses *where* to put monitoring wells/sensors so the resulting network is maximally
informative about a contaminant source, then cross-checks the design against a detection
objective (how fast the network would catch a plume that exceeds a regulatory limit).

Per the work plan, G6 builds on G2's ``invert_source`` (a `Posterior` over source location,
rate, and onset time -- IC-1) and on G1's ``GroundwaterTransportOperator``. Neither module
exists yet on this branch (G1/G2 are separate, still-open work items), so this module carries
a small self-contained analytic transport proxy (:func:`_concentration`) that plays the same
role for design purposes: it maps source parameters + a candidate site + a time to an expected
concentration. It blends the classic Ogata-Banks longitudinal breakthrough curve with a
Gaussian transverse profile and 1/r geometric attenuation -- a standard first-cut analytic
stand-in for 2-D continuous-injection advection-dispersion transport.

Both public functions only touch their inputs through the frozen IC-1 `Posterior` duck type
(``.samples``, ``.mean``, ``.cov``) and plain arrays/dicts, so once G2's ``invert_source`` lands
its output can be passed in as ``plume_prior`` unchanged -- there is no interface to migrate.
:class:`GaussianSourcePosterior` is a light IC-1-conforming convenience for building a
``plume_prior`` today (a linear-Gaussian stand-in for G2's Laplace posterior over the same
``(location, rate, onset)`` parameter vector); it is not part of G2's contract, just a helper so
callers (and this module's own tests) have something to pass in before G2 exists.

Scoring uses core `mixle.doe.active`: :func:`expected_information_gain_linear` for the default
linear-Gaussian (Bayesian D-optimal) criterion, or :func:`expected_information_gain_nmc` for a
nonlinear nested-Monte-Carlo criterion, over a per-candidate sensitivity row built by finite-
differencing the analytic transport proxy at the prior mean. Sites are added greedily
(submodular greedy) by largest marginal gain until the site budget is exhausted.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from mixle.doe.active import expected_information_gain_linear, expected_information_gain_nmc
from scipy.special import erfc
from scipy.stats import norm

__all__ = [
    "design_monitoring_network",
    "expected_detection_time",
    "GaussianSourcePosterior",
]

# --- self-contained analytic transport proxy (stands in for G1/G2 until they land) ---

_DEFAULT_VELOCITY = 1.0  # uniform groundwater flow speed along +x
_DEFAULT_DISPERSION_L = 5.0  # longitudinal dispersion
_DEFAULT_DISPERSION_T = 1.0  # transverse dispersion
_DEFAULT_THRESHOLD = 0.01  # regulatory concentration limit used when a scenario omits one
_OBS_NOISE = 0.002  # assumed sensor observation noise (concentration units)
_EPS = 1e-9


def _concentration(
    site_xy: Sequence[float],
    theta: np.ndarray,
    t: Any,
    *,
    velocity: float = _DEFAULT_VELOCITY,
    dispersion_l: float = _DEFAULT_DISPERSION_L,
    dispersion_t: float = _DEFAULT_DISPERSION_T,
) -> np.ndarray:
    """Analytic proxy concentration at ``site_xy`` and time(s) ``t`` for source ``theta``.

    ``theta = (x0, y0, log_rate, onset)``, broadcastable over a leading batch dimension so this
    doubles as the per-candidate forward model (single ``theta``) and the NMC inner/outer batch
    forward model (``theta`` of shape ``(m, 4)``). ``t`` may be a scalar or an array of times.
    """
    theta = np.asarray(theta, dtype=np.float64)
    x0, y0, log_q = theta[..., 0], theta[..., 1], theta[..., 2]
    onset = theta[..., 3]
    q = np.exp(log_q)
    dx = float(site_xy[0]) - x0
    dy = float(site_xy[1]) - y0
    tau = np.asarray(t, dtype=np.float64) - onset
    tau_safe = np.clip(tau, _EPS, None)
    r = np.sqrt(dx * dx + dy * dy)
    breakthrough = 0.5 * erfc((dx - velocity * tau_safe) / (2.0 * np.sqrt(dispersion_l * tau_safe)))
    transverse = np.exp(-(dy * dy) / (4.0 * dispersion_t * tau_safe))
    scale = q / (4.0 * np.pi * np.sqrt(dispersion_l * dispersion_t) * np.clip(r, _EPS, None))
    c = scale * breakthrough * transverse
    return np.where(tau > 0.0, c, 0.0)


def _travel_time_span(sites: np.ndarray, theta: np.ndarray, *, velocity: float = _DEFAULT_VELOCITY) -> float:
    """A generous monitoring-horizon *duration* (from onset): several multiples of the slowest
    candidate site's advective travel time, so most sites have a chance to break through."""
    x0 = float(np.atleast_1d(theta)[0])
    max_dx = float(np.max(np.abs(np.atleast_2d(sites)[:, 0] - x0))) if np.size(sites) else 1.0
    travel = max_dx / velocity if velocity else max_dx
    return 3.0 * max(travel, 1.0)


def _time_to_exceed(
    site_xy: Sequence[float],
    theta: np.ndarray,
    threshold: float,
    *,
    horizon_span: float,
    n_grid: int = 400,
    **transport_kwargs: Any,
) -> float:
    """First time (absolute) that ``site_xy`` would see ``theta``'s plume exceed ``threshold``,
    searched over ``[onset, onset + horizon_span]``; returns the horizon end if never exceeded
    (a censored, conservative "not yet detected" time)."""
    onset = float(np.atleast_1d(theta)[3])
    t_grid = np.linspace(onset, onset + horizon_span, int(n_grid))
    c = _concentration(site_xy, theta, t_grid, **transport_kwargs)
    hit = np.nonzero(c >= threshold)[0]
    if hit.size == 0:
        return onset + horizon_span
    return float(t_grid[int(hit[0])])


def _jacobian_row(site_xy: Sequence[float], theta0: np.ndarray, t_eval: float, eps: float = 1e-4) -> np.ndarray:
    """Central-difference sensitivity row d(concentration)/d(theta) at ``theta0``, evaluated at a
    single canonical monitoring time ``t_eval`` -- the per-candidate row of the linearized
    source -> receptor observation operator ("G2's forward" per the work-plan algorithm)."""
    theta0 = np.asarray(theta0, dtype=np.float64)
    p = theta0.shape[0]
    row = np.empty(p, dtype=np.float64)
    for j in range(p):
        step = np.zeros(p)
        step[j] = eps * max(1.0, abs(float(theta0[j])))
        plus = _concentration(site_xy, theta0 + step, t_eval)
        minus = _concentration(site_xy, theta0 - step, t_eval)
        row[j] = float((plus - minus) / (2.0 * step[j]))
    return row


class GaussianSourcePosterior:
    """A linear-Gaussian `Posterior` (IC-1) over ``theta = (x0, y0, log_rate, onset)``.

    Stand-in for G2's ``invert_source`` output until G2 lands on this branch: same parameter
    vector (source location, rate, onset), same duck-typed IC-1 surface. Any object exposing
    ``samples(n, rng)``/``mean``/``cov``/``credible_interval(level)``/``derived_quantity(fn, n,
    rng)`` -- including G2's eventual Laplace posterior -- works as ``plume_prior`` below.
    """

    def __init__(self, mean: Any, cov: Any) -> None:
        self._mean = np.asarray(mean, dtype=np.float64)
        self._cov = np.asarray(cov, dtype=np.float64)

    def samples(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return rng.multivariate_normal(self._mean, self._cov, size=int(n))

    @property
    def mean(self) -> np.ndarray:
        return self._mean

    @property
    def cov(self) -> np.ndarray:
        return self._cov

    def credible_interval(self, level: float) -> tuple[np.ndarray, np.ndarray]:
        z = float(norm.ppf(0.5 + level / 2.0))
        sd = np.sqrt(np.clip(np.diag(self._cov), 0.0, None))
        return self._mean - z * sd, self._mean + z * sd

    def derived_quantity(self, fn: Any, n: int, rng: np.random.Generator) -> _SimpleDerivedQuantity:
        return _SimpleDerivedQuantity(fn(self.samples(n, rng)))


class _SimpleDerivedQuantity:
    """Minimal IC-1 `DerivedQuantity`: samples + credible interval + the honesty flag."""

    def __init__(self, samples: Any) -> None:
        self.samples = np.asarray(samples)
        self.prior_dominated = False

    def credible_interval(self, level: float) -> tuple[np.ndarray, np.ndarray]:
        a = (1.0 - level) / 2.0
        return np.quantile(self.samples, a, axis=0), np.quantile(self.samples, 1 - a, axis=0)


def _eig_linear_score(jac_rows: np.ndarray, indices: Sequence[int], prior_cov: np.ndarray) -> float:
    if not indices:
        return 0.0
    return expected_information_gain_linear(jac_rows[list(indices)], noise=_OBS_NOISE, prior_cov=prior_cov)


def _eig_nmc_score(
    sites: np.ndarray,
    indices: Sequence[int],
    plume_prior: Any,
    t_eval: float,
    rng: np.random.RandomState,
) -> float:
    if not indices:
        return 0.0
    chosen = sites[list(indices)]

    def prior_sampler(r: np.random.RandomState, n: int) -> np.ndarray:
        return np.asarray(plume_prior.samples(int(n), np.random.default_rng(r.randint(0, 2**31 - 1))))

    def simulate(theta: np.ndarray, r: np.random.RandomState) -> np.ndarray:
        base = np.array([_concentration(s, theta, t_eval) for s in chosen])
        return base + r.normal(scale=_OBS_NOISE, size=base.shape[0])

    def log_likelihood(thetas: np.ndarray, y: np.ndarray) -> np.ndarray:
        thetas = np.atleast_2d(thetas)
        preds = np.stack([_concentration(s, thetas, t_eval) for s in chosen], axis=-1)
        resid = np.asarray(y)[None, :] - preds
        return -0.5 * np.sum((resid / _OBS_NOISE) ** 2 + np.log(2 * np.pi * _OBS_NOISE**2), axis=-1)

    return expected_information_gain_nmc(prior_sampler, log_likelihood, simulate, n_outer=48, n_inner=48, seed=rng)


def design_monitoring_network(
    candidate_sites: Any,
    plume_prior: Any,
    *,
    budget: float,
    k: int,
    criterion: str = "eig",
    min_separation: float = 0.0,
    existing_sites: Any = None,
) -> list[int]:
    """Greedily choose monitoring sites maximizing expected information gain about the source.

    ``candidate_sites`` is an ``(n, 2)`` array of candidate well/sensor coordinates.
    ``plume_prior`` is an IC-1 `Posterior` over ``theta = (x0, y0, log_rate, onset)`` -- e.g. G2's
    ``invert_source`` output (via :meth:`~mixle_pde.groundwater.SourcePosterior.to_doe_prior`), or
    :class:`GaussianSourcePosterior` before G2 lands. ``criterion`` is ``"eig"`` (linear-Gaussian
    Bayesian D-optimality, the default -- exact and cheap) or ``"nmc"`` (nested-Monte-Carlo EIG for
    the nonlinear forward model -- more expensive).

    **A real limitation, not a hypothetical one**: this greedy search scores every candidate by how
    much it would tighten the posterior *around the current prior mean* -- it has no way to ask
    "what would tell me my current belief is wrong." Chained into a sequential design loop (repeatedly
    re-fitting a nonlinear posterior and calling this again with the updated prior), if an early fit
    lands in a wrong local optimum, this criterion can systematically pick sites that reinforce that
    wrong optimum rather than ones that would expose it -- observed in practice over several rounds
    before the underlying nonlinear solver broke out on its own (see
    ``experiments/adaptive-groundwater-monitoring`` in the top-level repo for a worked, real example:
    uncertainty *grew* for 3 rounds before the fit corrected itself). ``min_separation``/
    ``existing_sites`` below reduce the risk (candidates get some minimum spatial diversity from
    what's already been surveyed, so the network can't cluster entirely around one possibly-wrong
    location) -- they do not eliminate it. This criterion is exact for a *linear* Gaussian forward
    (see ``mixle_pde.geophysics.invert_potential_field`` for that regime, where the analogous design
    is estimate-independent and this failure mode cannot occur); it is only a locally-greedy
    approximation for the nonlinear ``"eig"``/``"nmc"`` case used here.

    ``budget``/``k`` both cap the site count: real per-site cost models are out of scope (see
    module docstring / work-plan Non-goals), so with no cost model to convert a monetary
    ``budget`` into a site count, a unit cost per site is assumed and the network is capped at
    ``min(k, budget, n_candidates)`` sites.

    ``min_separation`` (default ``0.0``, disabled): when positive, a candidate within
    ``min_separation`` of any already-chosen site (within this call) *or* of any site in
    ``existing_sites`` (an ``(m, 2)`` array of prior-round sites, e.g. wells already drilled) is
    excluded from consideration at that greedy step -- a simple, real spatial-diversity constraint
    that stops the design from clustering entirely around one neighborhood every round.

    Returns the chosen site indices into ``candidate_sites``, in the order added by the greedy
    submodular search (largest marginal information gain first).
    """
    if criterion not in ("eig", "nmc"):
        raise ValueError("criterion must be 'eig' or 'nmc'.")

    sites = np.atleast_2d(np.asarray(candidate_sites, dtype=np.float64))
    n_candidates = sites.shape[0]
    max_sites = max(0, min(int(k), int(budget), n_candidates))
    if max_sites == 0:
        return []

    prior_sites = (
        np.atleast_2d(np.asarray(existing_sites, dtype=np.float64))
        if existing_sites is not None and np.size(existing_sites)
        else np.zeros((0, sites.shape[1]))
    )

    def _too_close(cand_idx: int, chosen_idx: list[int]) -> bool:
        if min_separation <= 0.0:
            return False
        against = (
            np.concatenate([sites[chosen_idx], prior_sites], axis=0)
            if chosen_idx or prior_sites.shape[0]
            else np.zeros((0, sites.shape[1]))
        )
        if against.shape[0] == 0:
            return False
        d = np.linalg.norm(against - sites[cand_idx], axis=1)
        return bool(np.min(d) < min_separation)

    theta0 = np.asarray(plume_prior.mean, dtype=np.float64)
    prior_cov = np.asarray(plume_prior.cov, dtype=np.float64)
    t_eval = float(theta0[3]) + _travel_time_span(sites, theta0)

    if criterion == "nmc":
        rng = np.random.RandomState(0)

        def score(indices: Sequence[int]) -> float:
            return _eig_nmc_score(sites, indices, plume_prior, t_eval, rng)
    else:
        jac_rows = np.stack([_jacobian_row(sites[i], theta0, t_eval) for i in range(n_candidates)])

        def score(indices: Sequence[int]) -> float:
            return _eig_linear_score(jac_rows, indices, prior_cov)

    chosen: list[int] = []
    remaining = list(range(n_candidates))
    base_score = score(chosen)
    while len(chosen) < max_sites and remaining:
        eligible = [c for c in remaining if not _too_close(c, chosen)]
        pool = eligible or remaining  # if the separation constraint would empty the pool, relax it
        best_idx, best_gain = None, -np.inf
        for cand in pool:
            gain = score(chosen + [cand]) - base_score
            if gain > best_gain:
                best_gain, best_idx = gain, cand
        chosen.append(best_idx)
        remaining.remove(best_idx)
        base_score = score(chosen)
    return chosen


def _scenario_theta(scenario: Mapping[str, Any]) -> np.ndarray:
    if "theta" in scenario:
        return np.asarray(scenario["theta"], dtype=np.float64)
    x0, y0 = scenario["location"]
    rate = float(scenario.get("rate", 1.0))
    onset = float(scenario.get("onset", 0.0))
    return np.array([float(x0), float(y0), float(np.log(max(rate, _EPS))), onset], dtype=np.float64)


def _scenario_transport_kwargs(scenario: Mapping[str, Any]) -> dict[str, float]:
    kwargs: dict[str, float] = {}
    for key in ("velocity", "dispersion_l", "dispersion_t"):
        if key in scenario:
            kwargs[key] = float(scenario[key])
    return kwargs


def expected_detection_time(sites: Any, plume_scenarios: Sequence[Mapping[str, Any]]) -> float:
    """Mean, over ``plume_scenarios``, of the earliest time any of ``sites`` detects the plume.

    Per scenario, "detected" means some site's concentration first crosses that scenario's
    regulatory ``threshold`` (default :data:`_DEFAULT_THRESHOLD` when a scenario omits one); the
    network's detection time for that scenario is the minimum such crossing time over its sites,
    and this returns the mean across scenarios -- the cross-check for the work-plan's IC-8
    ``prob_exceed``-flavoured "expected time-to-exceed a regulatory threshold" objective (the
    frozen signature carries no threshold parameter, so it travels with each scenario instead).

    Each scenario is a mapping with either a ``"theta"`` key (``(x0, y0, log_rate, onset)``) or
    ``"location"``/``"rate"``/``"onset"`` keys, plus optional ``"threshold"``,
    ``"velocity"``, ``"dispersion_l"``, ``"dispersion_t"``, ``"horizon_span"``.
    """
    site_arr = np.atleast_2d(np.asarray(sites, dtype=np.float64))
    if site_arr.shape[0] == 0:
        raise ValueError("expected_detection_time requires at least one site.")

    times: list[float] = []
    for scenario in plume_scenarios:
        theta = _scenario_theta(scenario)
        threshold = float(scenario.get("threshold", _DEFAULT_THRESHOLD))
        transport_kwargs = _scenario_transport_kwargs(scenario)
        velocity = transport_kwargs.get("velocity", _DEFAULT_VELOCITY)
        horizon_span = float(scenario.get("horizon_span", _travel_time_span(site_arr, theta, velocity=velocity)))
        site_times = [
            _time_to_exceed(site_arr[i], theta, threshold, horizon_span=horizon_span, **transport_kwargs)
            for i in range(site_arr.shape[0])
        ]
        times.append(min(site_times))
    return float(np.mean(times))
