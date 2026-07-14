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

* **Facies-mixture coupling (C7).** :class:`FaciesMixturePrior` generalizes the single linear tie into a
  ``K``-component Gaussian mixture over ``(a_i, b_i)``, so a bimodal (or ``K``-modal) rock-physics cloud is
  representable: each facies is its own Gaussian in property space, and the per-cell coupling precision is
  the responsibility-weighted sum of the per-facies quadratics. :meth:`FaciesMixturePrior.em_update` refits
  the mixture from a current model estimate, alternating with the Gauss-Newton model update.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mixle_pde.field_inversion import FieldGaussianPrior, _noise_precision
from mixle_pde.latent import Field3D, Field4D, PosteriorField3D
from mixle_pde.linear_solve import dense_spd_solve
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


def _gaussian_log_pdf_2d(points: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Log density of a 2-D Gaussian ``N(mean, cov)`` at each row of ``points`` (n, 2)."""
    d = points - mean
    prec = np.linalg.inv(cov)
    _sign, logdet = np.linalg.slogdet(cov)
    quad = np.einsum("ni,ij,nj->n", d, prec, d)
    return -0.5 * (quad + logdet + 2.0 * np.log(2.0 * np.pi))


@dataclass
class FaciesMixturePrior:
    """A ``K``-facies Gaussian-mixture petrophysical prior over two properties ``a`` and ``b``.

    :class:`CrossPropertyPrior` ties ``a`` and ``b`` with a single linear relation ``b ~ slope * a`` -- fine
    for one rock unit, but a real petrophysical cloud is often multimodal (e.g. shale vs. reservoir sand each
    with their own density/susceptibility trend). ``FaciesMixturePrior`` replaces that single tie with a
    ``K``-component Gaussian mixture over the per-cell pair ``(a_i, b_i)``: ``means`` (``K, 2``) and ``covs``
    (``K, 2, 2``) are the per-facies mean and covariance in property space, ``weights`` (``K,``) the global
    mixing proportions, and ``priors`` the ``[prior_a, prior_b]`` spatial-smoothness priors (as in
    :class:`CrossPropertyPrior`) applied to each property independently.

    The coupling precision is never a single scalar: cell ``i``'s effective ``(a, b)`` precision is the
    responsibility-weighted sum of the per-facies precisions, ``sum_k responsibilities[i, k] *
    inv(covs[k])`` -- so a bimodal (or ``K``-modal) rock-physics cloud is representable, each facies its own
    local Gaussian, with a soft per-cell assignment rather than one global slope. :meth:`em_update` refits
    the mixture (and returns the responsibilities) from a current ``(a, b)`` point estimate; the intended use
    alternates a Gauss-Newton model update with `em_update` (DR-ALG C7, step 3).
    """

    priors: list
    means: np.ndarray
    covs: np.ndarray
    weights: np.ndarray

    def __post_init__(self) -> None:
        if len(self.priors) != 2:
            raise ValueError("priors must be [prior_a, prior_b] -- one spatial-smoothness prior per property.")
        means = np.atleast_2d(np.asarray(self.means, dtype=float))
        if means.ndim != 2 or means.shape[1] != 2:
            raise ValueError("means must have shape (K, 2).")
        k = means.shape[0]
        covs = np.asarray(self.covs, dtype=float)
        if covs.shape != (k, 2, 2):
            raise ValueError("covs must have shape (K, 2, 2) matching means.")
        weights = np.atleast_1d(np.asarray(self.weights, dtype=float))
        if weights.shape != (k,):
            raise ValueError("weights must have shape (K,) matching means.")
        if k < 1:
            raise ValueError("at least one facies component is required.")
        if np.any(weights < 0.0):
            raise ValueError("weights must be non-negative.")
        self.means = means
        self.covs = covs
        self.weights = weights / weights.sum()

    def precision_sparse(self, grid: Field3D, *, responsibilities: np.ndarray):
        """Sparse CSR ``(2n, 2n)`` joint precision over the stacked ``[a; b]`` field.

        ``responsibilities`` is the ``(n, K)`` per-cell soft facies assignment (rows sum to 1, e.g. from
        :meth:`em_update`); cell ``i``'s coupling precision is the responsibility-weighted sum of the
        per-facies ``(a, b)`` precisions, added on the diagonal (this coupling is purely local to a cell --
        the spatial smoothness comes from ``priors`` alone, as in :class:`CrossPropertyPrior`).
        """
        import scipy.sparse as sp

        n = grid.n
        k = self.means.shape[0]
        resp = np.asarray(responsibilities, dtype=float)
        if resp.shape != (n, k):
            raise ValueError(f"responsibilities must have shape ({n}, {k}).")
        precs = np.linalg.inv(self.covs)  # (K, 2, 2)
        cell_prec = np.einsum("ik,kpq->ipq", resp, precs)  # (n, 2, 2), the per-cell blended precision
        prior_a, prior_b = self.priors
        q_aa = prior_a.precision_sparse(grid) + sp.diags(cell_prec[:, 0, 0], format="csr")
        q_bb = prior_b.precision_sparse(grid) + sp.diags(cell_prec[:, 1, 1], format="csr")
        q_ab = sp.diags(cell_prec[:, 0, 1], format="csr")
        return sp.bmat([[q_aa, q_ab], [q_ab, q_bb]], format="csr")

    def precision(self, grid: Field3D, *, responsibilities: np.ndarray) -> np.ndarray:
        """Dense ``(2n, 2n)`` joint precision -- see :meth:`precision_sparse`."""
        return self.precision_sparse(grid, responsibilities=responsibilities).toarray()

    def em_update(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """One expectation-maximization iteration over the per-cell facies responsibilities.

        E-step: soft-assigns each cell's ``(a_i, b_i)`` to a facies under the current ``(weights, means,
        covs)``. M-step: refits ``(weights, means, covs)`` to those responsibilities, in place. Returns the
        ``(n, K)`` responsibilities used for the M-step (feed them straight into :meth:`precision_sparse`'s
        ``responsibilities=``) -- a bimodal ``(a, b)`` cloud pulls the two facies means toward its two modes
        over repeated calls, which the single-slope :class:`CrossPropertyPrior` cannot represent.
        """
        a = np.asarray(a, dtype=float).reshape(-1)
        b = np.asarray(b, dtype=float).reshape(-1)
        if a.shape != b.shape:
            raise ValueError("a and b must have the same shape.")
        ab = np.stack([a, b], axis=1)  # (n, 2)
        n = ab.shape[0]
        k = self.means.shape[0]

        log_r = np.empty((n, k))
        for j in range(k):
            log_r[:, j] = np.log(max(float(self.weights[j]), 1.0e-300)) + _gaussian_log_pdf_2d(
                ab, self.means[j], self.covs[j]
            )
        log_r -= log_r.max(axis=1, keepdims=True)
        r = np.exp(log_r)
        r /= r.sum(axis=1, keepdims=True)

        nk = r.sum(axis=0)
        means = np.empty_like(self.means)
        covs = np.empty_like(self.covs)
        for j in range(k):
            denom = max(float(nk[j]), 1.0e-12)
            mean_j = (r[:, j : j + 1] * ab).sum(axis=0) / denom
            d = ab - mean_j
            cov_j = (r[:, j, None, None] * (d[:, :, None] * d[:, None, :])).sum(axis=0) / denom
            covs[j] = cov_j + 1.0e-9 * np.eye(2)
            means[j] = mean_j
        self.weights = nk / n
        self.means = means
        self.covs = covs
        return r


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

    cov = dense_spd_solve(lam + jitter * np.eye(2 * n), np.eye(2 * n))
    mean = cov @ rhs
    post_a = PosteriorField3D(grid=grid_a, mean=mean[:n], map=mean[:n].copy(), cov=cov[:n, :n])
    post_b = PosteriorField3D(grid=grid_b, mean=mean[n:], map=mean[n:].copy(), cov=cov[n:, n:])
    return post_a, post_b
