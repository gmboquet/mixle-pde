"""Linear-Gaussian inversion of a latent 3D field from multimodal observations (workstream G6, core).

Steps G1 (:mod:`mixle_pde.latent`) and G2 (:mod:`mixle_pde.observations`) define the latent object and
the typed-observation / forward-operator contracts but do "no inversion yet". This module is the
inversion: given a :class:`~mixle_pde.latent.Field3D`, a spatial-smoothness Gaussian prior, and a batch
of :class:`~mixle_pde.observations.Observation` s resolved through a
:class:`~mixle_pde.observations.ForwardOperatorRegistry`, it returns the exact Gaussian posterior as a
:class:`~mixle_pde.latent.PosteriorField3D` -- mean/MAP, full covariance (hence marginal variance,
credible intervals, and samples for free).

When every forward operator is LINEAR in the field (it declares a Jacobian -- gravity, magnetics, and
borehole point-sampling all do) and the field's transform is the identity (``bounds=None``, so the
unconstrained space the prior/posterior Gaussian lives in coincides with physical units), the posterior
is closed-form and EXACT -- no optimization, no linearization error:

    prior      m ~ N(m0, Q^-1)                       (Q = smoothness precision)
    data       d_i = J_i m + e_i,  e_i ~ N(0, R_i)
    posterior  m | d ~ N(mu, Lambda^-1),
               Lambda = Q + sum_i J_i^T R_i^-1 J_i,
               mu     = Lambda^-1 (Q m0 + sum_i J_i^T R_i^-1 d_i)

The prior precision is a graph-Matern (nearest-neighbour graph-Laplacian) smoothness operator built here
from the grid coordinates -- self-contained (numpy + scipy only), so a clean-checkout test can create a
field, attach observations, invert, and extract posterior mean/variance/intervals/samples with no other
module. A bounded (non-identity-transform) field makes the forward map nonlinear in the unconstrained
variable; that is the Gauss-Newton path of a later card, and is rejected here with a clear error rather
than silently linearized.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from mixle_pde.latent import Field3D, PosteriorField3D
from mixle_pde.observations import ForwardOperatorRegistry, Observation


@dataclass
class FieldGaussianPrior:
    """A Gaussian smoothness prior over a :class:`Field3D`, in the field's unconstrained space.

    The precision is a graph-Laplacian ("graph-Matern") operator over the grid's nearest-neighbour
    graph: ``smoothness_precision`` weights each edge (penalising differences between neighbouring
    cells -> spatial smoothness), and ``marginal_precision`` (> 0) anchors every cell to ``mean`` so the
    precision is full rank and the prior proper. ``mean`` is the unconstrained-space prior mean (a
    scalar broadcast over the grid, or a per-cell array).
    """

    mean: float | np.ndarray = 0.0
    smoothness_precision: float = 1.0
    marginal_precision: float = 1.0e-2
    length_scale: float = 1.0
    neighbors: int = 6

    def __post_init__(self) -> None:
        if self.smoothness_precision < 0.0 or self.marginal_precision <= 0.0:
            raise ValueError("smoothness_precision must be >= 0 and marginal_precision must be > 0 (proper prior).")
        if self.length_scale <= 0.0:
            raise ValueError("length_scale must be positive.")
        if int(self.neighbors) <= 0:
            raise ValueError("neighbors must be positive.")

    def mean_vector(self, grid: Field3D) -> np.ndarray:
        return np.broadcast_to(np.asarray(self.mean, dtype=float), (grid.n,)).astype(float).copy()

    def precision(self, grid: Field3D) -> np.ndarray:
        """Dense ``(n, n)`` graph-Laplacian smoothness precision + marginal anchoring."""
        from scipy.spatial import cKDTree

        coords = grid.coordinates
        n = coords.shape[0]
        prec = np.zeros((n, n), dtype=float)
        np.fill_diagonal(prec, self.marginal_precision)
        if n == 1 or self.smoothness_precision == 0.0:
            return prec

        k = min(int(self.neighbors) + 1, n)
        _, indices = cKDTree(coords).query(coords, k=k)
        if indices.ndim == 1:
            indices = indices[:, None]
        scaled = coords / self.length_scale
        seen: set[tuple[int, int]] = set()
        for i, row in enumerate(indices):
            for j in row:
                j = int(j)
                if i == j:
                    continue
                edge = (min(i, j), max(i, j))
                if edge in seen:
                    continue
                seen.add(edge)
                weight = self.smoothness_precision * float(np.exp(-np.linalg.norm(scaled[i] - scaled[j])))
                if weight == 0.0:
                    continue
                prec[i, i] += weight
                prec[j, j] += weight
                prec[i, j] -= weight
                prec[j, i] -= weight
        return prec


def _noise_precision(observation: Observation) -> np.ndarray:
    """``R^-1`` as a dense ``(n, n)`` matrix, from the observation's diagonal or full noise covariance."""
    if observation.is_diagonal:
        return np.diag(1.0 / observation.noise_cov)
    return np.linalg.inv(observation.noise_cov)


def linear_gaussian_invert(
    grid: Field3D,
    observations: list[Observation],
    registry: ForwardOperatorRegistry,
    prior: FieldGaussianPrior,
    *,
    jitter: float = 1.0e-10,
) -> PosteriorField3D:
    """Exact closed-form Gaussian posterior over ``grid`` from a batch of linear observations.

    Every observation's forward operator must declare a Jacobian (be linear in the field), and the field
    must have the identity transform (``bounds=None``) so the observation model is linear in the
    posterior's own (unconstrained) variable -- both are checked, not assumed. Returns a
    :class:`PosteriorField3D` with a dense covariance (mean == MAP for a linear-Gaussian model).
    """
    if grid.bounds is not None:
        raise ValueError(
            "linear_gaussian_invert requires an identity-transform field (bounds=None); a bounded field "
            "makes the forward map nonlinear in the posterior variable -- use the Gauss-Newton path."
        )
    if not observations:
        raise ValueError("need at least one observation to invert.")

    n = grid.n
    lam = prior.precision(grid)  # Q
    m0 = prior.mean_vector(grid)
    rhs = lam @ m0  # Q m0

    for obs in observations:
        op = registry.get(obs.kind)
        if not op.has_adjoint():
            raise ValueError(
                f"observation kind {obs.kind!r} has no Jacobian; linear_gaussian_invert needs a linear "
                "(adjoint-declaring) operator for every observation."
            )
        jac = np.atleast_2d(np.asarray(op.jacobian(grid, obs.location), dtype=float))  # (n_obs, n)
        if jac.shape != (obs.n, n):
            raise ValueError(f"operator {obs.kind!r} Jacobian shape {jac.shape} != ({obs.n}, {n}).")
        jt_rinv = jac.T @ _noise_precision(obs)  # (n, n_obs)
        lam = lam + jt_rinv @ jac
        rhs = rhs + jt_rinv @ obs.value

    cov = np.linalg.inv(lam + jitter * np.eye(n))
    mean = cov @ rhs
    return PosteriorField3D(grid=grid, mean=mean, map=mean.copy(), cov=cov)


@dataclass
class PosteriorPredictiveCheck:
    """Held-out fit of a fitted posterior: per-observation standardized residual and coverage."""

    residuals: np.ndarray  # (predicted - observed) at the posterior mean, stacked over held-out obs
    standardized: np.ndarray  # residual / sqrt(noise var + predictive var), stacked
    coverage: float  # fraction of held-out points inside their 1-alpha predictive interval
    alpha: float = field(default=0.1)


def posterior_predictive_check(
    posterior: PosteriorField3D,
    registry: ForwardOperatorRegistry,
    held_out: list[Observation],
    *,
    alpha: float = 0.1,
) -> PosteriorPredictiveCheck:
    """Predict ``held_out`` observations from a fitted posterior and measure standardized residuals +
    interval coverage -- the honest "does the posterior explain data it never saw?" check.

    The predictive variance of a linear observation ``J m`` under ``m ~ N(mu, cov)`` is ``J cov J^T``;
    added to the observation's own noise variance, it gives the predictive interval each held-out point
    is scored against.
    """
    from scipy.special import ndtri

    grid = posterior.grid
    cov = posterior.cov if posterior.cov is not None else np.diag(posterior.marginal_variance)
    z = ndtri(1.0 - alpha / 2.0)
    residuals: list[float] = []
    standardized: list[float] = []
    inside = 0
    total = 0
    for obs in held_out:
        op = registry.get(obs.kind)
        jac = np.atleast_2d(np.asarray(op.jacobian(grid, obs.location), dtype=float))
        predicted = jac @ posterior.mean
        pred_var = np.diag(jac @ cov @ jac.T)
        noise_var = obs.noise_cov if obs.is_diagonal else np.diag(obs.noise_cov)
        total_std = np.sqrt(pred_var + noise_var)
        resid = predicted - obs.value
        residuals.extend(resid.tolist())
        standardized.extend((resid / total_std).tolist())
        inside += int(np.sum(np.abs(resid) <= z * total_std))
        total += obs.n
    return PosteriorPredictiveCheck(
        residuals=np.asarray(residuals),
        standardized=np.asarray(standardized),
        coverage=(inside / total) if total else 0.0,
        alpha=alpha,
    )
