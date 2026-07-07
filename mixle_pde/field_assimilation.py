"""4D assimilation and smoothing of an evolving latent field (workstream G7).

Workstream G6 (:mod:`mixle_pde.field_inversion`) inverts a STATIC field. This module adds the time
axis: a latent field that EVOLVES, observed at a sequence of times, assimilated forward and smoothed
backward into a posterior at every time -- the linear-Gaussian state-space (Kalman filter + RTS
smoother) over the same :class:`~mixle_pde.latent.Field3D` grid and the same
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

    def _index_of(self, time: float, tol: float = 1e-9) -> int:
        matches = np.flatnonzero(np.isclose(self.times, time, atol=tol))
        if matches.size == 0:
            raise KeyError(f"time {time!r} is not an assimilated time; have {self.times.tolist()}.")
        return int(matches[0])

    def at_time(self, time: float) -> PosteriorField3D:
        """The posterior field slice at an assimilated ``time`` as a static :class:`PosteriorField3D`."""
        i = self._index_of(time)
        return PosteriorField3D(grid=self.grid, mean=self.means[i], map=self.means[i].copy(), cov=self.covs[i])

    def predict_observation(self, registry: ForwardOperatorRegistry, observation: Observation) -> np.ndarray:
        """Posterior-predictive mean of ``observation`` at its own time (posterior-predictive check hook)."""
        if observation.time is None:
            raise ValueError("observation.time must be set to predict from a 4D posterior.")
        i = self._index_of(observation.time)
        jac = np.atleast_2d(np.asarray(registry.get(observation.kind).jacobian(self.grid, observation.location)))
        return jac @ self.means[i]


def _update(mean_pred: np.ndarray, cov_pred: np.ndarray, obs_list, grid, registry):
    """Fuse a time-step's observations into a predicted Gaussian, in precision form (exact)."""
    n = mean_pred.shape[0]
    prec_pred = np.linalg.inv(cov_pred)
    lam = prec_pred.copy()
    rhs = prec_pred @ mean_pred
    for obs in obs_list:
        op = registry.get(obs.kind)
        if not op.has_adjoint():
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
