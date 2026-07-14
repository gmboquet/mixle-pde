"""Value-of-information: a linearized pre-posterior variance-reduction proxy (workstream C8).

Before drilling a borehole or flying a survey line, the question is not "what did we learn" (we have
no data yet) but "how much WOULD we learn". :func:`expected_variance_reduction` scores a candidate
observation geometry -- its location and declared noise model, resolved through a
:class:`~mixle_pde.observations.ForwardOperator` -- by how much it would shrink the posterior
uncertainty of a target region's mass/quantity, using only the current posterior and the candidate's
linearization. This is the same idea as :func:`mixle_pde.posterior_calibration.identifiability_diagnostic`
(observation sensitivity vs. posterior uncertainty flags what is weakly constrained) run forward in
time: instead of diagnosing an already-collected batch, it previews one that has not been collected.

The proxy is exact for a fixed-linear candidate operator under the current Gaussian posterior (a linear
Gauss-Markov expected-information-gain calculation), and a local linearization otherwise: folding a
candidate's Jacobian ``J`` and noise covariance ``R`` into the posterior via Sherman-Morrison-Woodbury
gives the would-be updated covariance without ever assembling a fresh ``(n, n)`` matrix --

    Sigma_1 = Sigma_0 - Sigma_0 J^T (R + J Sigma_0 J^T)^-1 J Sigma_0

-- so only covariance-vector products against the small candidate Jacobian are needed, and the routine
degrades to whichever covariance-storage mode (dense, sparse-precision, low-rank+diagonal) the posterior
already carries. :func:`next_best_observation` is the greedy (one-step-lookahead) argmax over a
candidate list -- not a combinatorial survey-design search (see the C8 non-goals).

The region-mass functional and the covariance actions below operate in the field's unconstrained space,
exact for the identity-transform (``bounds=None``) fields the linear-Gaussian inversion path produces --
the same scope :func:`~mixle_pde.posterior_calibration.heldout_observation_check` already has.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _region_weights(n: int, region: Any, cell_volumes: Any) -> np.ndarray:
    region = np.asarray(region, dtype=bool)
    if region.shape != (n,):
        raise ValueError(f"region must have shape ({n},).")
    volumes = np.broadcast_to(np.asarray(cell_volumes, dtype=float), (n,))
    return np.where(region, volumes, 0.0)


def _covariance_action(posterior: Any, v: np.ndarray) -> np.ndarray:
    """``Sigma_0 @ v`` (vector) or ``Sigma_0 @ V`` (matrix), for whichever covariance-storage mode
    ``posterior`` uses -- never materializes a dense ``(n, n)`` covariance beyond what is already
    stored."""
    v = np.asarray(v, dtype=float)
    single = v.ndim == 1
    mat = v[:, None] if single else v
    if posterior.dense_cov is not None:
        out = posterior.dense_cov @ mat
    elif posterior.precision_factor is not None:
        out = posterior.precision_factor.solve(mat)
    elif posterior.low_rank is not None:
        out = posterior.low_rank @ (posterior.low_rank.T @ mat) + posterior.diag_var[:, None] * mat
    else:
        out = posterior.diag_var[:, None] * mat
    return out[:, 0] if single else out


def expected_variance_reduction(
    posterior: Any,
    candidate_geometry: Any,
    forward_op: Any,
    *,
    region: Any,
    cell_volumes: Any,
) -> float:
    """Linearized fractional variance reduction ``1 - var_post / var_prior`` a candidate observation
    would buy for the region-mass functional over ``region``, WITHOUT collecting its data.

    ``candidate_geometry`` is an :class:`~mixle_pde.observations.Observation`-shaped object supplying the
    candidate's ``location`` and ``noise_cov`` (its ``value`` is never read -- no data exists yet);
    ``forward_op`` is the :class:`~mixle_pde.observations.ForwardOperator` that would produce it. The
    candidate's would-be Jacobian and noise model determine the Gaussian information it would add; the
    resulting variance update is folded in through Sherman-Morrison-Woodbury (module docstring), so the
    reduction is bounded to ``[0, 1]`` and needs only covariance-vector products against the candidate's
    (typically few-row) Jacobian.
    """
    grid = posterior.grid
    weights = _region_weights(grid.n, region, cell_volumes)
    sigma_w = _covariance_action(posterior, weights)
    var_prior = float(weights @ sigma_w)
    if var_prior <= 0.0:
        return 0.0

    jac = forward_op.local_jacobian(grid, posterior.mean, candidate_geometry)
    noise_cov = candidate_geometry.noise_cov
    r_mat = np.diag(noise_cov) if candidate_geometry.is_diagonal else np.asarray(noise_cov, dtype=float)

    sigma_jt = _covariance_action(posterior, jac.T)  # (n, n_obs)
    innovation_cov = r_mat + jac @ sigma_jt  # (n_obs, n_obs)
    cross = jac @ sigma_w  # (n_obs,): J Sigma_0 w
    reduction_term = float(cross @ np.linalg.solve(innovation_cov, cross))
    var_post = max(var_prior - reduction_term, 0.0)
    reduction = 1.0 - var_post / var_prior
    return float(np.clip(reduction, 0.0, 1.0))


def next_best_observation(
    posterior: Any,
    candidates: list,
    forward_op: Any,
    *,
    region: Any,
    cell_volumes: Any,
) -> tuple[int, float]:
    """Greedily pick the candidate with the largest :func:`expected_variance_reduction`.

    Returns ``(index, expected_reduction)`` for the winning candidate. One-step lookahead only -- ranks
    each candidate independently rather than searching combinations of candidates (see C8 non-goals).
    """
    if not candidates:
        raise ValueError("candidates must contain at least one candidate observation geometry.")
    scores = [
        expected_variance_reduction(posterior, candidate, forward_op, region=region, cell_volumes=cell_volumes)
        for candidate in candidates
    ]
    best = int(np.argmax(scores))
    return best, float(scores[best])
