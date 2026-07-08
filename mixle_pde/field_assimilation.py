"""4D assimilation and smoothing of an evolving latent field (workstream G7).

Workstream G6 (:mod:`mixle_pde.field_inversion`) inverts a STATIC field. This module adds the time
axis: a latent field that EVOLVES, observed at a sequence of times. The exact
:func:`assimilate_4d` path assimilates forward and smooths backward for linear-Gaussian observations.
:func:`assimilate_4d_ensemble` adds a stochastic ensemble path for nonlinear observation operators over
the same :class:`~mixle_pde.latent.Field3D` grid and the same
:class:`~mixle_pde.observations.ForwardOperatorRegistry` the static inversion uses.

State model (random walk in time, the honest default when no mechanistic evolution is asserted):

    m_0        ~ N(mu0, P0)                       (P0^-1 = prior smoothness precision)
    m_{t}      = m_{t-1} + w_t,  w_t ~ N(0, W)     (W = process_var * I, the drift budget per step)
    d_{t,i}    = J_{t,i} m_t + e,  e ~ N(0, R)     (linear observations at time t)

The filter walks forward (predict: inflate covariance by ``W``; update: fuse that step's observations
in precision form -- the exact same fusion the static inversion does); the RTS smoother walks backward
so each time's posterior also uses LATER observations. Both process and observation uncertainty are
preserved: an un-observed time still gets a posterior (wider, from the prior + neighbours in time), and
:meth:`PosteriorField4D.at_time` exposes the posterior slice at any assimilated time as an ordinary
:class:`~mixle_pde.latent.PosteriorField3D` (so its samples / intervals / posterior-predictive all work).
The ensemble path returns filtered Gaussian summaries from ensemble mean/covariance; it is a reference
posterior approximation for nonlinear/time-evolving cases, not a production particle smoother.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mixle_pde.field_inversion import FieldGaussianPrior, _noise_precision
from mixle_pde.latent import Field3D, PosteriorField3D
from mixle_pde.observations import ForwardOperatorRegistry, Observation


@dataclass
class PosteriorField4D:
    """A Gaussian posterior over an evolving field at a sequence of times.

    ``means[t]`` / ``covs[t]`` are the smoothed posterior mean and dense covariance at ``times[t]``.
    """

    grid: Field3D
    times: np.ndarray
    means: list[np.ndarray]
    covs: list[np.ndarray]

    def __post_init__(self) -> None:
        times = np.asarray(self.times, dtype=float).reshape(-1)
        if times.size == 0:
            raise ValueError("times must be a non-empty 1-D array.")
        if np.any(np.diff(times) <= 0.0):
            raise ValueError("times must be strictly increasing.")
        if len(self.means) != times.size or len(self.covs) != times.size:
            raise ValueError("means and covs must have one entry per time.")

        means = [np.asarray(mean, dtype=float) for mean in self.means]
        covs = [np.asarray(cov, dtype=float) for cov in self.covs]
        for mean in means:
            if mean.shape != (self.grid.n,):
                raise ValueError(f"each mean must have shape ({self.grid.n},).")
        for cov in covs:
            if cov.shape != (self.grid.n, self.grid.n):
                raise ValueError(f"each covariance must have shape ({self.grid.n}, {self.grid.n}).")
        self.times = times
        self.means = means
        self.covs = covs

    def _index_of(self, time: float, tol: float = 1e-9) -> int:
        matches = np.flatnonzero(np.isclose(self.times, time, atol=tol))
        if matches.size == 0:
            raise KeyError(f"time {time!r} is not an assimilated time; have {self.times.tolist()}.")
        return int(matches[0])

    @property
    def mean_array(self) -> np.ndarray:
        """Posterior mean as a ``(n_times, n_grid)`` array."""
        return np.vstack(self.means)

    @property
    def marginal_variance(self) -> np.ndarray:
        """Per-time, per-node posterior variance as a ``(n_times, n_grid)`` array."""
        return np.vstack([np.clip(np.diag(cov), 0.0, None) for cov in self.covs])

    @property
    def marginal_std(self) -> np.ndarray:
        """Per-time, per-node posterior standard deviation."""
        return np.sqrt(self.marginal_variance)

    def at_time(self, time: float, *, interpolate: bool = False) -> PosteriorField3D:
        """The posterior field slice at ``time`` as a static :class:`PosteriorField3D`.

        By default ``time`` must be one of the assimilated times. With ``interpolate=True``, the method
        linearly interpolates the marginal Gaussian summaries between adjacent assimilated times. The
        interpolation is a query convenience, not a replacement for storing full cross-time covariance.
        """
        if not interpolate:
            i = self._index_of(time)
            return PosteriorField3D(grid=self.grid, mean=self.means[i], map=self.means[i].copy(), cov=self.covs[i])
        t = float(time)
        if t < self.times[0] or t > self.times[-1]:
            raise ValueError("time is outside the posterior time range.")
        right = int(np.searchsorted(self.times, t, side="left"))
        if right < len(self.times) and np.isclose(t, self.times[right]):
            return self.at_time(float(self.times[right]))
        if right == 0:
            return self.at_time(float(self.times[0]))
        if right >= len(self.times):
            return self.at_time(float(self.times[-1]))
        left = right - 1
        weight = (t - self.times[left]) / (self.times[right] - self.times[left])
        mean = (1.0 - weight) * self.means[left] + weight * self.means[right]
        cov = (1.0 - weight) * self.covs[left] + weight * self.covs[right]
        return PosteriorField3D(grid=self.grid, mean=mean, map=mean.copy(), cov=cov)

    def credible_interval(self, alpha: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
        """Per-time central credible intervals as ``(n_times, n_grid)`` arrays."""
        lows: list[np.ndarray] = []
        highs: list[np.ndarray] = []
        for time in self.times:
            lo, hi = self.at_time(float(time)).credible_interval(alpha)
            lows.append(lo)
            highs.append(hi)
        return np.vstack(lows), np.vstack(highs)

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Draw per-time marginal posterior samples with shape ``(n, n_times, n_grid)``.

        The stored posterior contains marginal covariances at each time, not the full cross-time
        covariance, so samples are independent across time slices.
        """
        samples = [self.at_time(float(time)).sample(n, rng) for time in self.times]
        return np.stack(samples, axis=1)

    def predict_observation(self, registry: ForwardOperatorRegistry, observation: Observation) -> np.ndarray:
        """Posterior-predictive mean of ``observation`` at its own time (posterior-predictive check hook)."""
        if observation.time is None:
            raise ValueError("observation.time must be set to predict from a 4D posterior.")
        i = self._index_of(observation.time)
        op = registry.get(observation.kind)
        return op.predict_observation(self.grid, self.means[i], observation)


def _update(mean_pred: np.ndarray, cov_pred: np.ndarray, obs_list, grid, registry):
    """Fuse a time-step's observations into a predicted Gaussian, in precision form (exact)."""
    n = mean_pred.shape[0]
    prec_pred = np.linalg.inv(cov_pred)
    lam = prec_pred.copy()
    rhs = prec_pred @ mean_pred
    for obs in obs_list:
        op = registry.get(obs.kind)
        if not op.is_linear:
            raise ValueError(f"observation kind {obs.kind!r} needs a Jacobian for linear-Gaussian assimilation.")
        jac = np.atleast_2d(np.asarray(op.jacobian(grid, obs.location), dtype=float))
        jt_rinv = jac.T @ _noise_precision(obs)
        lam = lam + jt_rinv @ jac
        rhs = rhs + jt_rinv @ obs.value
    cov_upd = np.linalg.inv(lam + 1e-12 * np.eye(n))
    return cov_upd @ rhs, cov_upd


def assimilate_4d(
    grid: Field3D,
    times,
    observations_by_time,
    registry: ForwardOperatorRegistry,
    prior: FieldGaussianPrior,
    *,
    process_var: float,
) -> PosteriorField4D:
    """Kalman-filter + RTS-smooth an evolving field observed over ``times``.

    ``observations_by_time`` is a list parallel to ``times``: entry ``t`` is the list of
    :class:`Observation` s recorded at ``times[t]`` (possibly empty -- an un-observed time still gets a
    posterior). ``process_var`` is the per-step random-walk drift variance ``W = process_var * I`` (the
    field's budget to change between steps; larger = trusts data over persistence). The identity
    transform (``bounds=None``) is required, as in the static inversion.
    """
    if grid.bounds is not None:
        raise ValueError("assimilate_4d requires an identity-transform field (bounds=None).")
    times = np.asarray(times, dtype=float)
    if times.ndim != 1 or times.size == 0:
        raise ValueError("times must be a non-empty 1-D array.")
    if len(observations_by_time) != times.size:
        raise ValueError("observations_by_time must have one entry (list) per time.")
    if process_var <= 0.0:
        raise ValueError("process_var must be positive.")

    n = grid.n
    W = process_var * np.eye(n)

    # forward filter
    filt_mean: list[np.ndarray] = []
    filt_cov: list[np.ndarray] = []
    pred_mean: list[np.ndarray] = []
    pred_cov: list[np.ndarray] = []
    m_prev = prior.mean_vector(grid)
    P_prev = np.linalg.inv(prior.precision(grid))
    for t in range(times.size):
        if t == 0:
            m_pred, P_pred = m_prev, P_prev  # prior is the t0 forecast
        else:
            m_pred, P_pred = m_prev, P_prev + W  # random-walk predict (A = I)
        pred_mean.append(m_pred)
        pred_cov.append(P_pred)
        m_upd, P_upd = _update(m_pred, P_pred, observations_by_time[t], grid, registry)
        filt_mean.append(m_upd)
        filt_cov.append(P_upd)
        m_prev, P_prev = m_upd, P_upd

    # RTS backward smoother (A = I)
    sm_mean = [None] * times.size
    sm_cov = [None] * times.size
    sm_mean[-1] = filt_mean[-1]
    sm_cov[-1] = filt_cov[-1]
    for t in range(times.size - 2, -1, -1):
        C = filt_cov[t] @ np.linalg.inv(pred_cov[t + 1])
        sm_mean[t] = filt_mean[t] + C @ (sm_mean[t + 1] - pred_mean[t + 1])
        sm_cov[t] = filt_cov[t] + C @ (sm_cov[t + 1] - pred_cov[t + 1]) @ C.T

    return PosteriorField4D(grid=grid, times=times, means=list(sm_mean), covs=list(sm_cov))


def _ensemble_covariance(ensemble: np.ndarray, *, jitter: float) -> np.ndarray:
    centered = ensemble - ensemble.mean(axis=0, keepdims=True)
    cov = centered.T @ centered / max(ensemble.shape[0] - 1, 1)
    return cov + jitter * np.eye(ensemble.shape[1])


def _observation_noise_sample(observation: Observation, rng: np.random.Generator, size: int) -> np.ndarray:
    if observation.is_diagonal:
        return rng.normal(0.0, np.sqrt(observation.noise_cov), size=(size, observation.n))
    return rng.multivariate_normal(np.zeros(observation.n), observation.noise_cov, size=size)


def assimilate_4d_ensemble(
    grid: Field3D,
    times,
    observations_by_time,
    registry: ForwardOperatorRegistry,
    prior: FieldGaussianPrior,
    *,
    process_var: float,
    ensemble_size: int = 128,
    rng: np.random.Generator | None = None,
    inflation: float = 1.0,
    jitter: float = 1.0e-9,
) -> PosteriorField4D:
    """Ensemble Kalman assimilation for nonlinear evolving-field observations.

    Unlike :func:`assimilate_4d`, this path does not require fixed-linear observation operators. Each
    ensemble member is pushed through ``ForwardOperator.predict_observation``; the empirical
    state/observation covariance supplies the Kalman gain. The returned :class:`PosteriorField4D`
    contains filtered Gaussian summaries at each time.
    """
    if grid.bounds is not None:
        raise ValueError("assimilate_4d_ensemble currently requires an identity-transform field (bounds=None).")
    times = np.asarray(times, dtype=float)
    if times.ndim != 1 or times.size == 0:
        raise ValueError("times must be a non-empty 1-D array.")
    if len(observations_by_time) != times.size:
        raise ValueError("observations_by_time must have one entry (list) per time.")
    if process_var <= 0.0:
        raise ValueError("process_var must be positive.")
    if ensemble_size < 2:
        raise ValueError("ensemble_size must be at least 2.")
    if inflation <= 0.0:
        raise ValueError("inflation must be positive.")
    if jitter < 0.0:
        raise ValueError("jitter must be non-negative.")

    rng = np.random.default_rng() if rng is None else rng
    n = grid.n
    prior_mean = prior.mean_vector(grid)
    prior_cov = np.linalg.inv(prior.precision(grid) + jitter * np.eye(n))
    ensemble = rng.multivariate_normal(prior_mean, prior_cov, size=ensemble_size)
    means: list[np.ndarray] = []
    covs: list[np.ndarray] = []

    for t in range(times.size):
        if t > 0:
            ensemble = ensemble + rng.normal(0.0, np.sqrt(process_var), size=ensemble.shape)
        for obs in observations_by_time[t]:
            op = registry.get(obs.kind)
            predicted = np.vstack([op.predict_observation(grid, member, obs) for member in ensemble])
            x_mean = ensemble.mean(axis=0)
            y_mean = predicted.mean(axis=0)
            x_anom = (ensemble - x_mean) * np.sqrt(inflation)
            y_anom = (predicted - y_mean) * np.sqrt(inflation)
            cov_xy = x_anom.T @ y_anom / (ensemble_size - 1)
            noise_cov = obs.noise_cov if not obs.is_diagonal else np.diag(obs.noise_cov)
            cov_yy = y_anom.T @ y_anom / (ensemble_size - 1) + noise_cov + jitter * np.eye(obs.n)
            gain = cov_xy @ np.linalg.inv(cov_yy)
            perturbed = obs.value[None, :] + _observation_noise_sample(obs, rng, ensemble_size)
            innovation = perturbed - predicted
            ensemble = ensemble + innovation @ gain.T
        means.append(ensemble.mean(axis=0).copy())
        covs.append(_ensemble_covariance(ensemble, jitter=jitter))

    return PosteriorField4D(grid=grid, times=times, means=means, covs=covs)
