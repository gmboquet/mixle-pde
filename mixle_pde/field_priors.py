"""Depth weighting, spatiotemporal, and cross-property coupling priors (workstream G5, remaining rungs).

Workstream G6 shipped the spatial smoothness prior (:class:`~mixle_pde.field_inversion.FieldGaussianPrior`,
a graph-Matern/roughness precision) and G7 the temporal process prior. This module adds the two prior
ingredients step 5 still names:

* **Depth weighting.** A potential-field kernel decays with depth, so an un-weighted inversion piles all
  the recovered anomaly into the shallowest cells (they explain the data most cheaply). :func:`depth_weights`
  is the standard Li & Oldenburg depth weighting ``w(z) = (|z| + z0)^(-beta/2)``; folded into the prior's
  marginal precision via :func:`depth_weighted_marginal_precision` or its sparse CSR counterpart, it
  removes that bias so a body is recovered at its true depth rather than smeared to the surface.

* **Spatiotemporal coupling.** :class:`SpatioTemporalGaussianPrior` lifts a spatial
  :class:`FieldGaussianPrior` onto a :class:`~mixle_pde.latent.Field4D` and adds random-walk temporal
  precision blocks, producing one sparse precision over the whole ``(time, space)`` object.

* **Cross-property coupling.** Two physical properties over the same grid (e.g. density contrast and
  magnetic susceptibility) are rarely independent -- a petrophysical relation ``b ~ slope * a`` ties them.
  :class:`CrossPropertyPrior` is a JOINT Gaussian prior over the stacked ``[a; b]`` field with each
  property's own spatial smoothness PLUS a coupling term ``c * sum_i (b_i - slope * a_i)^2``, so
  observations of ONE property inform the OTHER. :func:`joint_linear_gaussian_invert` does the exact
  closed-form joint inversion: gravity (on ``a``) and magnetics (on ``b``) fused into one posterior, and a
  property with NO direct data of its own recovered through the coupling from the property that does.
  ``CrossPropertyPrior.precision_sparse`` exposes the same block precision as a CSR matrix for scalable
  assembly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from mixle_pde.field_inversion import FieldGaussianPrior, _noise_precision
from mixle_pde.latent import Field3D, Field4D, PosteriorField3D
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


def depth_weighted_marginal_precision_sparse(
    prior: FieldGaussianPrior, grid: Field3D, *, beta: float = 3.0, z0: float = 1.0
):
    """Sparse CSR version of :func:`depth_weighted_marginal_precision`."""
    import scipy.sparse as sp

    base = prior.precision_sparse(grid)
    w = depth_weights(grid, beta=beta, z0=z0)
    correction = sp.diags(prior.marginal_precision * (w**2 - 1.0), format="csr")
    return (base + correction).tocsr()


@dataclass
class SpatioTemporalGaussianPrior:
    """A sparse Gaussian prior over a :class:`Field4D`.

    The spatial prior is applied independently at each time. ``temporal_precision`` adds a random-walk
    penalty ``temporal_precision / dt * ||x[t+1] - x[t]||^2`` for each interval, so shorter intervals
    constrain changes more tightly than longer intervals.
    """

    spatial_prior: FieldGaussianPrior
    temporal_precision: float

    def __post_init__(self) -> None:
        if self.temporal_precision < 0.0:
            raise ValueError("temporal_precision must be non-negative.")

    def precision_sparse(self, field: Field4D):
        """Sparse CSR precision over the flattened ``(time, space)`` field."""
        import scipy.sparse as sp

        n_t = field.n_times
        n_x = field.n_per_time
        q_space = self.spatial_prior.precision_sparse(field.spatial)
        q = sp.kron(sp.eye(n_t, format="csr"), q_space, format="csr")
        if self.temporal_precision > 0.0 and n_t > 1:
            diag = np.zeros(n_t, dtype=float)
            off = np.zeros(n_t - 1, dtype=float)
            for t in range(n_t - 1):
                dt = float(field.times[t + 1] - field.times[t])
                weight = self.temporal_precision / dt
                diag[t] += weight
                diag[t + 1] += weight
                off[t] = -weight
            temporal = sp.diags([off, diag, off], offsets=[-1, 0, 1], format="csr")
            q = q + sp.kron(temporal, sp.eye(n_x, format="csr"), format="csr")
        return q.tocsr()

    def precision(self, field: Field4D) -> np.ndarray:
        """Dense precision for small reference problems."""
        return self.precision_sparse(field).toarray()

    def mean_vector(self, field: Field4D) -> np.ndarray:
        """Flattened prior mean over all times."""
        return np.tile(self.spatial_prior.mean_vector(field.spatial), field.n_times)


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
        return self.precision_sparse(grid).toarray()

    def precision_sparse(self, grid: Field3D):
        """Sparse CSR joint precision over the stacked ``[a; b]`` field."""
        import scipy.sparse as sp

        n = grid.n
        # coupling c * (b - slope a)^2 -> precision blocks c*[[slope^2, -slope],[-slope, 1]] (x) I
        c, s = self.coupling, self.slope
        eye = sp.eye(n, format="csr")
        return sp.bmat(
            [
                [self.prior_a.precision_sparse(grid) + c * s * s * eye, -c * s * eye],
                [-c * s * eye, self.prior_b.precision_sparse(grid) + c * eye],
            ],
            format="csr",
        )

    def mean_vector(self, grid: Field3D) -> np.ndarray:
        return np.concatenate([self.prior_a.mean_vector(grid), self.prior_b.mean_vector(grid)])


def _data_normal_equations(grid, observations, registry, n):
    """``(sum J^T R^-1 J, sum J^T R^-1 d)`` for one property's observation batch."""
    lam = np.zeros((n, n))
    rhs = np.zeros(n)
    for obs in observations:
        op = registry.get(obs.kind)
        if not op.is_linear:
            raise ValueError(f"observation kind {obs.kind!r} needs a fixed Jacobian for the joint inversion.")
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


@dataclass
class TVFieldPrior:
    """Blocky/edge-preserving total-variation prior (workstream A1): the IRLS surrogate of the L1
    penalty on a roughness/gradient operator, config-object form of
    :func:`mixle_pde.blocky_priors.total_variation_weights`.

    ``roughness`` is any ``(k, n)`` scipy-sparse roughness/gradient operator -- typically
    :func:`mixle_pde.geophysics.roughness_operator` or, for a structurally-oriented smoothness,
    :func:`mixle_pde.blocky_priors.dip_rotated_gradient_operator`. :meth:`reweight` is the callable to
    hand to :func:`mixle_pde.geophysics.regularized_gauss_newton`'s ``reweight`` kwarg (directly, or via
    :func:`mixle_pde.blocky_priors.blocky_invert`'s outer IRLS loop), so a sharp, blocky recovery is one
    named prior object away rather than a hand-assembled weight function.
    """

    roughness: Any
    eps: float = 1e-6

    def __post_init__(self) -> None:
        if self.eps <= 0.0:
            raise ValueError("eps must be positive.")

    def reweight(self, m: np.ndarray) -> np.ndarray:
        """IRLS row weights for :meth:`roughness`, one per its row (see
        :func:`mixle_pde.blocky_priors.total_variation_weights`)."""
        from mixle_pde.blocky_priors import total_variation_weights

        return total_variation_weights(m, self.roughness, eps=self.eps)


@dataclass
class MinimumSupportPrior:
    """Compact/minimum-support prior (workstream A1): the Last & Kubik (1983) IRLS surrogate
    ``1/(m^2+eps^2)`` that concentrates recovered anomaly into a compact body, config-object form of
    :func:`mixle_pde.blocky_priors.minimum_support_weights`.

    Per-cell (not per-roughness-row): pair with ``roughness=None`` (identity damping) in
    :func:`mixle_pde.geophysics.regularized_gauss_newton`, or use directly via
    :func:`mixle_pde.blocky_priors.blocky_invert`'s ``prior="compact"``.
    """

    eps: float = 1e-6

    def __post_init__(self) -> None:
        if self.eps <= 0.0:
            raise ValueError("eps must be positive.")

    def reweight(self, m: np.ndarray) -> np.ndarray:
        """IRLS per-cell weights (see :func:`mixle_pde.blocky_priors.minimum_support_weights`)."""
        from mixle_pde.blocky_priors import minimum_support_weights

        return minimum_support_weights(m, eps=self.eps)
