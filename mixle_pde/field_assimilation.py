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
from mixle_pde.latent import Field3D, PosteriorField3D, PosteriorFieldSamples3D
from mixle_pde.observations import ForwardOperatorRegistry, Observation


@dataclass
class PosteriorField4D:
    """A Gaussian posterior over an evolving field at a sequence of times.

    ``means[t]`` / ``covs[t]`` are the smoothed posterior mean and dense covariance at ``times[t]``.
    ``joint_cov`` optionally stores the full ``(n_times * grid.n, n_times * grid.n)`` covariance.
    """

    grid: Field3D
    times: np.ndarray
    means: list[np.ndarray]
    covs: list[np.ndarray]
    joint_cov: np.ndarray | None = None

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
        if self.joint_cov is not None:
            joint = np.asarray(self.joint_cov, dtype=float)
            size = times.size * self.grid.n
            if joint.shape != (size, size):
                raise ValueError(f"joint_cov must have shape ({size}, {size}).")
            self.joint_cov = joint
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
    def mean_vector(self) -> np.ndarray:
        """Posterior mean flattened as one 4D object vector."""
        return self.mean_array.reshape(-1)

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

    def cross_covariance(self, time_a: float, time_b: float) -> np.ndarray:
        """Cross-time covariance block between two assimilated times."""
        if self.joint_cov is None:
            raise ValueError("joint_cov is not stored for this posterior.")
        ia = self._index_of(time_a)
        ib = self._index_of(time_b)
        n = self.grid.n
        return self.joint_cov[ia * n : (ia + 1) * n, ib * n : (ib + 1) * n].copy()

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
        if self.joint_cov is not None:
            size = self.times.size * self.grid.n
            chol = np.linalg.cholesky(self.joint_cov + 1.0e-12 * np.eye(size))
            unconstrained = self.mean_vector[None, :] + rng.standard_normal((n, size)) @ chol.T
            return self.grid.from_unconstrained(unconstrained.reshape(n, self.times.size, self.grid.n))
        samples = [self.at_time(float(time)).sample(n, rng) for time in self.times]
        return np.stack(samples, axis=1)

    def predict_observation(self, registry: ForwardOperatorRegistry, observation: Observation) -> np.ndarray:
        """Posterior-predictive mean of ``observation`` at its own time (posterior-predictive check hook)."""
        if observation.time is None:
            raise ValueError("observation.time must be set to predict from a 4D posterior.")
        i = self._index_of(observation.time)
        op = registry.get(observation.kind)
        return op.predict_observation(self.grid, self.means[i], observation)

    def posterior_predictive_draws(
        self,
        registry: ForwardOperatorRegistry,
        observation: Observation,
        *,
        n: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Posterior-predictive draws for ``observation`` at its own time."""
        if observation.time is None:
            raise ValueError("observation.time must be set to draw from a 4D posterior.")
        return self.at_time(observation.time).posterior_predictive_draws(registry, observation, n=n, rng=rng)


@dataclass
class PosteriorFieldSamples4D:
    """Empirical posterior trajectories over an evolving :class:`Field3D`.

    ``samples`` are stored in the field's unconstrained coordinates with shape
    ``(n_samples, n_times, grid.n)``. Resampling returns whole trajectories, preserving cross-time
    dependence rather than drawing each time slice independently.
    """

    grid: Field3D
    times: np.ndarray
    samples: np.ndarray
    provenance: dict | None = None

    def __post_init__(self) -> None:
        times = np.asarray(self.times, dtype=float).reshape(-1)
        if times.size == 0:
            raise ValueError("times must be a non-empty 1-D array.")
        if np.any(np.diff(times) <= 0.0):
            raise ValueError("times must be strictly increasing.")
        samples = np.asarray(self.samples, dtype=float)
        if samples.ndim != 3 or samples.shape[1:] != (times.size, self.grid.n):
            raise ValueError(f"samples must have shape (n_samples, {times.size}, {self.grid.n}).")
        if samples.shape[0] == 0:
            raise ValueError("samples must contain at least one posterior trajectory.")
        if not np.all(np.isfinite(samples)):
            raise ValueError("samples must be finite.")
        self.times = times
        self.samples = samples
        self.provenance = {} if self.provenance is None else dict(self.provenance)

    def _index_of(self, time: float, tol: float = 1e-9) -> int:
        matches = np.flatnonzero(np.isclose(self.times, time, atol=tol))
        if matches.size == 0:
            raise KeyError(f"time {time!r} is not an assimilated time; have {self.times.tolist()}.")
        return int(matches[0])

    @property
    def n_samples(self) -> int:
        return int(self.samples.shape[0])

    @property
    def mean_array(self) -> np.ndarray:
        """Empirical posterior mean in unconstrained coordinates, shape ``(n_times, grid.n)``."""
        return np.mean(self.samples, axis=0)

    @property
    def marginal_variance(self) -> np.ndarray:
        """Empirical per-time variance in unconstrained coordinates."""
        if self.n_samples == 1:
            return np.zeros((self.times.size, self.grid.n))
        return np.var(self.samples, axis=0, ddof=1)

    @property
    def marginal_std(self) -> np.ndarray:
        return np.sqrt(self.marginal_variance)

    @property
    def physical_samples(self) -> np.ndarray:
        """Stored posterior trajectories mapped to physical units."""
        return self.grid.from_unconstrained(self.samples)

    def at_time(self, time: float) -> PosteriorFieldSamples3D:
        """Empirical posterior slice at one assimilated time."""
        i = self._index_of(time)
        return PosteriorFieldSamples3D(
            grid=self.grid,
            samples=self.samples[:, i, :],
            provenance=self.provenance | {"time": float(self.times[i])},
        )

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Resample whole physical trajectories with shape ``(n, n_times, grid.n)``."""
        if int(n) <= 0:
            raise ValueError("n must be positive.")
        idx = rng.integers(0, self.n_samples, size=int(n))
        return self.grid.from_unconstrained(self.samples[idx])

    def credible_interval(self, alpha: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
        """Per-time empirical central credible intervals in physical units."""
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be in (0, 1).")
        physical = self.physical_samples
        lo, hi = np.quantile(physical, [alpha / 2.0, 1.0 - alpha / 2.0], axis=0)
        return lo, hi

    def predict_observation(self, registry: ForwardOperatorRegistry, observation: Observation) -> np.ndarray:
        """Posterior-predictive mean for ``observation`` by averaging model predictions over trajectories."""
        if observation.time is None:
            raise ValueError("observation.time must be set to predict from a 4D sampled posterior.")
        i = self._index_of(observation.time)
        op = registry.get(observation.kind)
        physical = self.grid.from_unconstrained(self.samples[:, i, :])
        return np.mean([op.predict_observation(self.grid, draw, observation) for draw in physical], axis=0)

    def posterior_predictive_draws(
        self,
        registry: ForwardOperatorRegistry,
        observation: Observation,
        *,
        n: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Posterior-predictive model draws for ``observation`` at its own time."""
        if observation.time is None:
            raise ValueError("observation.time must be set to draw from a 4D sampled posterior.")
        if int(n) <= 0:
            raise ValueError("n must be positive.")
        i = self._index_of(observation.time)
        idx = rng.integers(0, self.n_samples, size=int(n))
        op = registry.get(observation.kind)
        physical = self.grid.from_unconstrained(self.samples[idx, i, :])
        return np.vstack([op.predict_observation(self.grid, draw, observation) for draw in physical])


@dataclass(frozen=True)
class ParticleAssimilationReport:
    """Diagnostics for sequential particle assimilation."""

    n_particles: int
    n_times: int
    resample_count: int
    effective_sample_size: list[float]
    log_evidence_increment: list[float]


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


def _coerce_transition_matrices(transitions, n_steps: int, n_state: int) -> list[np.ndarray]:
    if callable(transitions):
        mats = [np.asarray(transitions(step), dtype=float) for step in range(n_steps)]
    else:
        arr = np.asarray(transitions, dtype=float)
        if arr.shape == (n_state, n_state):
            mats = [arr.copy() for _ in range(n_steps)]
        elif arr.shape == (n_steps, n_state, n_state):
            mats = [arr[step].copy() for step in range(n_steps)]
        else:
            mats = [np.asarray(mat, dtype=float) for mat in transitions]
    if len(mats) != n_steps:
        raise ValueError(f"transitions must contain {n_steps} matrices.")
    for mat in mats:
        if mat.shape != (n_state, n_state):
            raise ValueError(f"each transition matrix must have shape ({n_state}, {n_state}).")
    return mats


def _coerce_process_covariances(process_cov, n_steps: int, n_state: int) -> list[np.ndarray]:
    arr = np.asarray(process_cov, dtype=float)
    if arr.ndim == 0:
        if float(arr) <= 0.0:
            raise ValueError("process_cov scalar must be positive.")
        return [float(arr) * np.eye(n_state) for _ in range(n_steps)]
    if arr.shape == (n_state,):
        if np.any(arr <= 0.0):
            raise ValueError("process_cov diagonal entries must be positive.")
        return [np.diag(arr) for _ in range(n_steps)]
    if arr.shape == (n_state, n_state):
        _validate_covariance(arr, "process_cov")
        return [arr.copy() for _ in range(n_steps)]
    if arr.shape == (n_steps, n_state):
        if np.any(arr <= 0.0):
            raise ValueError("process_cov diagonal entries must be positive.")
        return [np.diag(arr[step]) for step in range(n_steps)]
    if arr.shape == (n_steps, n_state, n_state):
        covs = [arr[step].copy() for step in range(n_steps)]
        for cov in covs:
            _validate_covariance(cov, "process_cov")
        return covs
    raise ValueError("process_cov must be scalar, shape (n,), shape (n,n), shape (n_steps,n), or shape (n_steps,n,n).")


def _validate_covariance(matrix: np.ndarray, name: str) -> None:
    if not np.allclose(matrix, matrix.T):
        raise ValueError(f"{name} must be symmetric.")
    sign, _ = np.linalg.slogdet(matrix)
    if sign <= 0.0:
        raise ValueError(f"{name} must be positive definite.")


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


def assimilate_4d_linear_dynamics(
    grid: Field3D,
    times,
    observations_by_time,
    registry: ForwardOperatorRegistry,
    prior: FieldGaussianPrior,
    *,
    transitions,
    process_cov,
) -> PosteriorField4D:
    """Kalman-filter + RTS-smooth an evolving field with supplied linear dynamics matrices."""
    if grid.bounds is not None:
        raise ValueError("assimilate_4d_linear_dynamics requires an identity-transform field (bounds=None).")
    times = np.asarray(times, dtype=float)
    if times.ndim != 1 or times.size == 0:
        raise ValueError("times must be a non-empty 1-D array.")
    if np.any(np.diff(times) <= 0.0):
        raise ValueError("times must be strictly increasing.")
    if len(observations_by_time) != times.size:
        raise ValueError("observations_by_time must have one entry (list) per time.")

    n = grid.n
    n_transitions = int(times.size - 1)
    transition_mats = _coerce_transition_matrices(transitions, n_transitions, n)
    process_covs = _coerce_process_covariances(process_cov, n_transitions, n)

    filt_mean: list[np.ndarray] = []
    filt_cov: list[np.ndarray] = []
    pred_mean: list[np.ndarray] = []
    pred_cov: list[np.ndarray] = []
    m_prev = prior.mean_vector(grid)
    P_prev = np.linalg.inv(prior.precision(grid))
    for t in range(times.size):
        if t == 0:
            m_pred, P_pred = m_prev, P_prev
        else:
            A = transition_mats[t - 1]
            m_pred = A @ m_prev
            P_pred = A @ P_prev @ A.T + process_covs[t - 1]
        pred_mean.append(m_pred)
        pred_cov.append(P_pred)
        m_upd, P_upd = _update(m_pred, P_pred, observations_by_time[t], grid, registry)
        filt_mean.append(m_upd)
        filt_cov.append(P_upd)
        m_prev, P_prev = m_upd, P_upd

    sm_mean = [None] * times.size
    sm_cov = [None] * times.size
    sm_mean[-1] = filt_mean[-1]
    sm_cov[-1] = filt_cov[-1]
    for t in range(times.size - 2, -1, -1):
        A = transition_mats[t]
        C = filt_cov[t] @ A.T @ np.linalg.inv(pred_cov[t + 1])
        sm_mean[t] = filt_mean[t] + C @ (sm_mean[t + 1] - pred_mean[t + 1])
        sm_cov[t] = filt_cov[t] + C @ (sm_cov[t + 1] - pred_cov[t + 1]) @ C.T

    return PosteriorField4D(grid=grid, times=times, means=list(sm_mean), covs=list(sm_cov))


def assimilate_4d_joint_linear_dynamics(
    grid: Field3D,
    times,
    observations_by_time,
    registry: ForwardOperatorRegistry,
    prior: FieldGaussianPrior,
    *,
    transitions,
    process_cov,
    jitter: float = 1.0e-10,
) -> PosteriorField4D:
    """Exact joint Gaussian posterior over the full 4D field for linear dynamics and observations."""
    if grid.bounds is not None:
        raise ValueError("assimilate_4d_joint_linear_dynamics requires an identity-transform field (bounds=None).")
    times = np.asarray(times, dtype=float)
    if times.ndim != 1 or times.size == 0:
        raise ValueError("times must be a non-empty 1-D array.")
    if np.any(np.diff(times) <= 0.0):
        raise ValueError("times must be strictly increasing.")
    if len(observations_by_time) != times.size:
        raise ValueError("observations_by_time must have one entry (list) per time.")
    if jitter < 0.0:
        raise ValueError("jitter must be non-negative.")

    n = grid.n
    n_times = int(times.size)
    n_steps = n_times - 1
    transition_mats = _coerce_transition_matrices(transitions, n_steps, n)
    process_covs = _coerce_process_covariances(process_cov, n_steps, n)
    size = n_times * n
    precision = np.zeros((size, size), dtype=float)
    rhs = np.zeros(size, dtype=float)

    prior_precision = prior.precision(grid)
    prior_mean = prior.mean_vector(grid)
    precision[:n, :n] += prior_precision
    rhs[:n] += prior_precision @ prior_mean

    for step in range(n_steps):
        A = transition_mats[step]
        q_inv = np.linalg.inv(process_covs[step])
        lo = step * n
        hi = (step + 1) * n
        precision[lo : lo + n, lo : lo + n] += A.T @ q_inv @ A
        precision[lo : lo + n, hi : hi + n] += -A.T @ q_inv
        precision[hi : hi + n, lo : lo + n] += -q_inv @ A
        precision[hi : hi + n, hi : hi + n] += q_inv

    for ti, obs_list in enumerate(observations_by_time):
        offset = ti * n
        for obs in obs_list:
            if obs.time is not None and not np.isclose(obs.time, times[ti]):
                raise ValueError(f"observation time {obs.time!r} does not match assimilation time {times[ti]!r}.")
            op = registry.get(obs.kind)
            if not op.is_linear:
                raise ValueError(f"observation kind {obs.kind!r} needs a fixed Jacobian for joint linear assimilation.")
            jac = np.atleast_2d(np.asarray(op.jacobian(grid, obs.location), dtype=float))
            if jac.shape != (obs.n, n):
                raise ValueError(f"operator {obs.kind!r} Jacobian shape {jac.shape} != ({obs.n}, {n}).")
            jt_rinv = jac.T @ _noise_precision(obs)
            precision[offset : offset + n, offset : offset + n] += jt_rinv @ jac
            rhs[offset : offset + n] += jt_rinv @ obs.value

    cov = np.linalg.inv(precision + float(jitter) * np.eye(size))
    mean = cov @ rhs
    means = [mean[ti * n : (ti + 1) * n].copy() for ti in range(n_times)]
    covs = [cov[ti * n : (ti + 1) * n, ti * n : (ti + 1) * n].copy() for ti in range(n_times)]
    return PosteriorField4D(grid=grid, times=times, means=means, covs=covs, joint_cov=cov)


def _ensemble_covariance(ensemble: np.ndarray, *, jitter: float) -> np.ndarray:
    centered = ensemble - ensemble.mean(axis=0, keepdims=True)
    cov = centered.T @ centered / max(ensemble.shape[0] - 1, 1)
    return cov + jitter * np.eye(ensemble.shape[1])


def _observation_noise_sample(observation: Observation, rng: np.random.Generator, size: int) -> np.ndarray:
    if observation.is_diagonal:
        return rng.normal(0.0, np.sqrt(observation.noise_cov), size=(size, observation.n))
    return rng.multivariate_normal(np.zeros(observation.n), observation.noise_cov, size=size)


def _logsumexp(values: np.ndarray) -> float:
    vals = np.asarray(values, dtype=float)
    vmax = float(np.max(vals))
    if not np.isfinite(vmax):
        return vmax
    return float(vmax + np.log(np.sum(np.exp(vals - vmax))))


def _systematic_resample(weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n = weights.size
    positions = (rng.random() + np.arange(n)) / n
    cumulative = np.cumsum(weights)
    cumulative[-1] = 1.0
    return np.searchsorted(cumulative, positions, side="right")


def particle_assimilate_4d(
    grid: Field3D,
    times,
    observations_by_time,
    registry: ForwardOperatorRegistry,
    prior: FieldGaussianPrior,
    *,
    process_var: float,
    n_particles: int = 512,
    rng: np.random.Generator | None = None,
    resample_threshold: float = 0.5,
    jitter: float = 1.0e-9,
) -> tuple[PosteriorFieldSamples4D, ParticleAssimilationReport]:
    """Sequential Monte Carlo assimilation for nonlinear/non-Gaussian 4D field posteriors.

    The state evolves as a random walk in the field's unconstrained coordinates. Observation likelihoods
    are evaluated in physical coordinates through the registry, so bounded fields and nonlinear
    operators use the same observation contract as the Gaussian and ensemble paths.
    """
    times = np.asarray(times, dtype=float)
    if times.ndim != 1 or times.size == 0:
        raise ValueError("times must be a non-empty 1-D array.")
    if np.any(np.diff(times) <= 0.0):
        raise ValueError("times must be strictly increasing.")
    if len(observations_by_time) != times.size:
        raise ValueError("observations_by_time must have one entry (list) per time.")
    if process_var <= 0.0:
        raise ValueError("process_var must be positive.")
    if int(n_particles) < 2:
        raise ValueError("n_particles must be at least 2.")
    if not 0.0 < resample_threshold <= 1.0:
        raise ValueError("resample_threshold must be in (0, 1].")
    if jitter < 0.0:
        raise ValueError("jitter must be non-negative.")
    for time, obs_list in zip(times, observations_by_time, strict=True):
        for obs in obs_list:
            if obs.time is not None and not np.isclose(obs.time, time):
                raise ValueError(f"observation time {obs.time!r} does not match assimilation time {time!r}.")

    rng = np.random.default_rng() if rng is None else rng
    n_particles = int(n_particles)
    n_state = grid.n
    prior_mean = prior.mean_vector(grid)
    prior_cov = np.linalg.inv(prior.precision(grid) + jitter * np.eye(n_state))
    particles = rng.multivariate_normal(prior_mean, prior_cov, size=n_particles)
    trajectories = np.empty((n_particles, times.size, n_state), dtype=float)
    log_weights = np.full(n_particles, -np.log(n_particles), dtype=float)
    ess_history: list[float] = []
    log_evidence: list[float] = []
    resample_count = 0

    for ti, time in enumerate(times):
        if ti > 0:
            particles = particles + rng.normal(0.0, np.sqrt(process_var), size=particles.shape)
        trajectories[:, ti, :] = particles

        obs_list = observations_by_time[ti]
        if obs_list:
            log_likelihood = np.empty(n_particles, dtype=float)
            physical = grid.from_unconstrained(particles)
            for pi in range(n_particles):
                log_likelihood[pi] = registry.total_log_likelihood(grid, physical[pi], obs_list)
        else:
            log_likelihood = np.zeros(n_particles, dtype=float)

        unnormalized = log_weights + log_likelihood
        log_norm = _logsumexp(unnormalized)
        if not np.isfinite(log_norm):
            raise ValueError(f"all particle weights are zero at time {time!r}.")
        weights = np.exp(unnormalized - log_norm)
        ess = float(1.0 / np.sum(weights**2))
        ess_history.append(ess)
        log_evidence.append(float(log_norm))

        should_resample = ess <= resample_threshold * n_particles or ti == times.size - 1
        if should_resample:
            idx = _systematic_resample(weights, rng)
            particles = particles[idx]
            trajectories[:, : ti + 1, :] = trajectories[idx, : ti + 1, :]
            log_weights = np.full(n_particles, -np.log(n_particles), dtype=float)
            resample_count += 1
        else:
            log_weights = np.log(np.maximum(weights, np.finfo(float).tiny))

    posterior = PosteriorFieldSamples4D(
        grid=grid,
        times=times,
        samples=trajectories,
        provenance={
            "method": "particle_filter",
            "process_var": float(process_var),
            "resample_threshold": float(resample_threshold),
        },
    )
    report = ParticleAssimilationReport(
        n_particles=n_particles,
        n_times=int(times.size),
        resample_count=resample_count,
        effective_sample_size=ess_history,
        log_evidence_increment=log_evidence,
    )
    return posterior, report


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
