"""The first posterior-inference engine over a latent field (workstream G, step 6, first rung).

G1 (:mod:`mixle_pde.latent`) defines the latent object and its posterior container; G2
(:mod:`mixle_pde.observations`) defines observations and their likelihood. This module closes the
loop for the LINEAR rung of the inference ladder the plan calls for ("MAP/Gauss-Newton for fast
estimates ... every engine must return the same posterior artifact interface"): every
:class:`~mixle_pde.observations.ForwardOperator` this repo currently registers (gravity, magnetics,
borehole) is exactly linear in the field values (a fixed sensitivity/selection matrix, independent of
the current field), so for a Gaussian prior the MAP estimate and its Laplace covariance are the
EXACT closed-form Bayesian linear-Gaussian update -- no Gauss-Newton iteration needed, and the
returned covariance is not an approximation but the true posterior for this linear-Gaussian model.

A genuinely nonlinear forward operator (no fixed Jacobian, i.e. ``has_adjoint() is False``) is
explicitly out of scope here -- :func:`fit_map_posterior` raises rather than silently linearizing
around a point, mirroring G1's own "first card defines the object, a later card does the nonlinear
case" posture.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from mixle_pde.latent import Field3D, PosteriorField3D
from mixle_pde.observations import ForwardOperatorRegistry, Observation


def fit_map_posterior(
    grid: Field3D,
    observations: Sequence[Observation],
    registry: ForwardOperatorRegistry,
    *,
    prior_mean: np.ndarray | None = None,
    prior_cov: np.ndarray | float | None = None,
) -> PosteriorField3D:
    """Fit a :class:`~mixle_pde.latent.PosteriorField3D` from a (possibly multi-kind) batch of
    observations, in the grid's UNCONSTRAINED space, via the exact closed-form Bayesian
    linear-Gaussian update.

    ``prior_mean`` defaults to zeros (unconstrained space); ``prior_cov`` defaults to the identity
    times ``100.0`` (a weak/uninformative prior) and may be a scalar (isotropic) or a full ``(n, n)``
    matrix. Every observation's :class:`~mixle_pde.observations.ForwardOperator` must expose a
    Jacobian (``has_adjoint()``) -- this engine handles the LINEAR rung of the inference ladder only;
    a nonlinear operator raises rather than being silently (and possibly badly) linearized.
    """
    if grid.bounds is not None:
        raise ValueError(
            "fit_map_posterior only handles an UNBOUNDED property: a bounded property's physical "
            "values differ from the unconstrained space the Gaussian prior/posterior lives in, so "
            "the forward operators (which predict in physical units) would need to be re-expressed "
            "in unconstrained coordinates for this closed-form update to hold exactly. That "
            "transform-aware fit is a later card, not silently approximated here."
        )
    n = grid.n
    mean0 = np.zeros(n) if prior_mean is None else np.asarray(prior_mean, dtype=float)
    if mean0.shape != (n,):
        raise ValueError(f"prior_mean must have shape ({n},), got {mean0.shape}.")

    if prior_cov is None:
        cov0 = 100.0 * np.eye(n)
    elif np.isscalar(prior_cov):
        cov0 = float(prior_cov) * np.eye(n)
    else:
        cov0 = np.asarray(prior_cov, dtype=float)
        if cov0.shape != (n, n):
            raise ValueError(f"prior_cov must be a scalar or have shape ({n}, {n}), got {cov0.shape}.")
    prec0 = np.linalg.inv(cov0)

    precision = prec0.copy()
    rhs = prec0 @ mean0
    for obs in observations:
        op = registry.get(obs.kind)
        if not op.has_adjoint():
            raise ValueError(
                f"observation kind {obs.kind!r} has no Jacobian (has_adjoint() is False); "
                "fit_map_posterior only handles the linear rung of the inference ladder."
            )
        J = op.jacobian(grid, obs.location)
        if J.shape != (obs.n, n):
            raise ValueError(f"jacobian for kind {obs.kind!r} must have shape ({obs.n}, {n}), got {J.shape}.")
        if obs.is_diagonal:
            r_inv = 1.0 / obs.noise_cov
            precision += J.T @ (r_inv[:, None] * J)
            rhs += J.T @ (r_inv * obs.value)
        else:
            r_inv = np.linalg.inv(obs.noise_cov)
            precision += J.T @ r_inv @ J
            rhs += J.T @ r_inv @ obs.value

    cov_post = np.linalg.inv(precision)
    mean_post = cov_post @ rhs
    return PosteriorField3D(grid=grid, mean=mean_post, map=mean_post.copy(), cov=cov_post)


def posterior_predictive_log_likelihood(
    grid: Field3D, posterior: PosteriorField3D, observations: Sequence[Observation], registry: ForwardOperatorRegistry
) -> float:
    """Held-out check: total log-likelihood of ``observations`` under the posterior MEAN field --
    the posterior-predictive fit metric the plan's acceptance criteria ask every inversion to report.
    """
    return registry.total_log_likelihood(grid, posterior.mean, list(observations))
