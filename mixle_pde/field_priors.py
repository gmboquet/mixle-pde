"""Depth weighting and cross-property coupling priors (workstream G5, remaining rungs).

Workstream G6 shipped the spatial smoothness prior (:class:`~mixle_pde.field_inversion.FieldGaussianPrior`,
a graph-Matern/roughness precision) and G7 the temporal process prior. This module adds the two prior
ingredients step 5 still names:

* **Depth weighting.** A potential-field kernel decays with depth, so an un-weighted inversion piles all
  the recovered anomaly into the shallowest cells (they explain the data most cheaply). :func:`depth_weights`
  is the standard Li & Oldenburg depth weighting ``w(z) = (|z| + z0)^(-beta/2)``; folded into the prior's
  marginal precision via :func:`depth_weighted_marginal_precision`, it removes that bias so a body is
  recovered at its true depth rather than smeared to the surface.

* **Cross-property coupling.** Two physical properties over the same grid (e.g. density contrast and
  magnetic susceptibility) are rarely independent -- a petrophysical relation ``b ~ slope * a`` ties them.
  :class:`CrossPropertyPrior` is a JOINT Gaussian prior over the stacked ``[a; b]`` field with each
  property's own spatial smoothness PLUS a coupling term ``c * sum_i (b_i - slope * a_i)^2``, so
  observations of ONE property inform the OTHER. :func:`joint_linear_gaussian_invert` does the exact
  closed-form joint inversion: gravity (on ``a``) and magnetics (on ``b``) fused into one posterior, and a
  property with NO direct data of its own recovered through the coupling from the property that does.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mixle_pde.field_inversion import FieldGaussianPrior, _noise_precision
from mixle_pde.latent import Field3D, PosteriorField3D
from mixle_pde.observations import ForwardOperatorRegistry, Observation


def depth_weights(grid: Field3D, *, beta: float = 3.0, z0: float = 1.0) -> np.ndarray:
    """Li & Oldenburg depth weights ``w(z) = (|z| + z0)^(-beta/2)`` per cell (``z`` the vertical axis).

    ``beta`` matches the kernel's depth decay (3 for gravity/point-mass, 3 for a dipole); ``z0`` avoids a
    singularity at the surface. Larger weight = MORE confidently near the prior (a shallow cell), so deep
    cells are freed to carry anomaly.
    """
    if beta < 0.0:
        raise ValueError("beta must be non-negative.")
    if z0 <= 0.0:
        raise ValueError("z0 must be positive.")
    depth = np.abs(grid.coordinates[:, 2])
    return (depth + z0) ** (-beta / 2.0)


def depth_weighted_marginal_precision(
    prior: FieldGaussianPrior, grid: Field3D, *, beta: float = 3.0, z0: float = 1.0
) -> np.ndarray:
    """The prior precision with its marginal (anchoring) term depth-weighted per cell.

    Returns a dense precision like :meth:`FieldGaussianPrior.precision`, but the marginal-precision
    diagonal is scaled by :func:`depth_weights` (squared, since precision is inverse variance) so shallow
    cells are anchored harder to the prior and deep cells are freer to carry the recovered anomaly.
    """
    base = prior.precision(grid)
    w = depth_weights(grid, beta=beta, z0=z0)
    # replace the flat marginal_precision on the diagonal with a depth-scaled one
    base = base + np.diag(prior.marginal_precision * (w**2 - 1.0))
    return base


@dataclass
class CrossPropertyPrior:
    """A joint Gaussian prior over two properties ``a`` and ``b`` on the same grid, petrophysically coupled.

    Each property has its own spatial-smoothness :class:`FieldGaussianPrior`; ``coupling`` (>= 0) is the
    strength of the petrophysical tie ``b_i ~ slope * a_i`` -- a penalty ``coupling * sum_i (b_i - slope
    a_i)^2`` added to the joint precision. ``coupling = 0`` recovers two independent inversions.
    """

    prior_a: FieldGaussianPrior
    prior_b: FieldGaussianPrior
    coupling: float = 1.0
    slope: float = 1.0

    def __post_init__(self) -> None:
        if self.coupling < 0.0:
            raise ValueError("coupling must be non-negative.")

    def precision(self, grid: Field3D) -> np.ndarray:
        """The ``(2n, 2n)`` joint precision over the stacked ``[a; b]`` field."""
        n = grid.n
        Qa = self.prior_a.precision(grid)
        Qb = self.prior_b.precision(grid)
        joint = np.zeros((2 * n, 2 * n))
        joint[:n, :n] = Qa
        joint[n:, n:] = Qb
        # coupling c * (b - slope a)^2 -> precision blocks c*[[slope^2, -slope],[-slope, 1]] (x) I
        c, s = self.coupling, self.slope
        eye = np.eye(n)
        joint[:n, :n] += c * s * s * eye
        joint[n:, n:] += c * eye
        joint[:n, n:] += -c * s * eye
        joint[n:, :n] += -c * s * eye
        return joint

    def mean_vector(self, grid: Field3D) -> np.ndarray:
        return np.concatenate([self.prior_a.mean_vector(grid), self.prior_b.mean_vector(grid)])


def _data_normal_equations(grid, observations, registry, n):
    """``(sum J^T R^-1 J, sum J^T R^-1 d)`` for one property's observation batch."""
    lam = np.zeros((n, n))
    rhs = np.zeros(n)
    for obs in observations:
        op = registry.get(obs.kind)
        if not op.has_adjoint():
            raise ValueError(f"observation kind {obs.kind!r} needs a Jacobian for the joint inversion.")
        J = np.atleast_2d(np.asarray(op.jacobian(grid, obs.location), dtype=float))
        jt_rinv = J.T @ _noise_precision(obs)
        lam += jt_rinv @ J
        rhs += jt_rinv @ obs.value
    return lam, rhs


def joint_linear_gaussian_invert(
    grid_a: Field3D,
    grid_b: Field3D,
    prior: CrossPropertyPrior,
    observations_a: list[Observation],
    registry_a: ForwardOperatorRegistry,
    observations_b: list[Observation],
    registry_b: ForwardOperatorRegistry,
    *,
    jitter: float = 1.0e-10,
) -> tuple[PosteriorField3D, PosteriorField3D]:
    """Exact closed-form JOINT posterior over two coupled properties from their respective observations.

    ``grid_a`` / ``grid_b`` share coordinates but name different properties (density vs susceptibility);
    both must be identity-transform. Gravity-type observations of ``a`` and magnetics-type of ``b`` are
    fused through :class:`CrossPropertyPrior`, so a property with sparse or NO direct data is still
    recovered through the petrophysical coupling. Returns ``(posterior_a, posterior_b)``.
    """
    if grid_a.bounds is not None or grid_b.bounds is not None:
        raise ValueError("joint_linear_gaussian_invert requires identity-transform fields (bounds=None).")
    if grid_a.n != grid_b.n:
        raise ValueError("grid_a and grid_b must have the same number of cells.")
    n = grid_a.n

    Q = prior.precision(grid_a)
    m0 = prior.mean_vector(grid_a)
    lam = Q.copy()
    rhs = Q @ m0

    la, ra = _data_normal_equations(grid_a, observations_a, registry_a, n)
    lb, rb = _data_normal_equations(grid_b, observations_b, registry_b, n)
    lam[:n, :n] += la
    lam[n:, n:] += lb
    rhs[:n] += ra
    rhs[n:] += rb

    cov = np.linalg.inv(lam + jitter * np.eye(2 * n))
    mean = cov @ rhs
    post_a = PosteriorField3D(grid=grid_a, mean=mean[:n], map=mean[:n].copy(), cov=cov[:n, :n])
    post_b = PosteriorField3D(grid=grid_b, mean=mean[n:], map=mean[n:].copy(), cov=cov[n:, n:])
    return post_a, post_b
