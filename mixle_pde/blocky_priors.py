"""Blocky, compact, and structurally anisotropic priors for the exploration inversions (workstream A1).

:func:`~mixle_pde.geophysics.regularized_gauss_newton` fills the potential-field data null space with an
L2 smoothness penalty, which is exactly the wrong prior for a targeted exploration body: it smears a
compact ore lens or a thin dyke into a diffuse blob spread across many cells, understating the peak grade
and the true extent. This module adds the machinery to invert for **blocky** (edge-preserving) and
**compact** (minimum-support) bodies instead, and to let the smoothness itself follow a known structural
dip rather than being isotropic:

* :func:`total_variation_weights` -- the IRLS row-reweighting that turns the quadratic ``||R m||^2``
  smoothness penalty into a surrogate for the L1/total-variation penalty ``||R m||_1``, which penalizes
  the *number* of interfaces rather than their steepness, producing sharp, blocky boundaries.
* :func:`minimum_support_weights` -- the Last & Kubik (1983) minimum-support (compactness) reweighting
  ``1/(m^2+eps^2)``, applied per model cell, which concentrates the recovered anomaly into as few cells
  as possible rather than spreading it to minimize roughness.
* :func:`dip_rotated_gradient_operator` -- a first-difference gradient operator whose basis is rotated
  by a structural strike/dip so the smoothness/blockiness penalty is applied along and across the known
  structure rather than along the (arbitrary) grid axes.
* :func:`blocky_invert` -- the outer Iteratively Reweighted Least Squares (IRLS) loop around
  :func:`~mixle_pde.geophysics.regularized_gauss_newton` that ties the above together: fix the weights
  from the current iterate, take one (possibly reweighted-)Gauss-Newton step, recompute the weights, and
  repeat. Feeding its ``(x, std)`` return into a :class:`~mixle_pde.latent.PosteriorField3D` produces a
  posterior conforming the shared ``Posterior`` protocol (IC-1), same as the plain-smoothness inverter.

Every reweighting here is a *row* (or cell) weight fed to ``regularized_gauss_newton``'s ``reweight``
kwarg -- nothing here duplicates the Gauss-Newton machinery itself.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import scipy.sparse as sp

from mixle_pde.geophysics import regularized_gauss_newton, roughness_operator

__all__ = [
    "dip_rotated_gradient_operator",
    "minimum_support_weights",
    "total_variation_weights",
    "blocky_invert",
]


def minimum_support_weights(m: np.ndarray, *, eps: float = 1e-6) -> np.ndarray:
    """Last & Kubik (1983) minimum-support (compactness) IRLS weights, one per model cell.

    ``1/(m^2+eps^2)`` grows without bound as a cell's value shrinks to zero, so the reweighted quadratic
    penalty ``sum_i w_i m_i^2 = sum_i m_i^2/(m_i^2+eps^2)`` saturates near 1 for cells already carrying
    anomaly and stays near zero for cells near the reference -- the IRLS surrogate of the minimum-support
    (``L0``-like) functional that concentrates recovered mass into a compact body rather than smearing it.
    Meant to be used as damping (``roughness=None``, i.e. the identity operator) in
    :func:`~mixle_pde.geophysics.regularized_gauss_newton`, since the weights are per cell, not per
    roughness row.
    """
    if eps <= 0.0:
        raise ValueError("eps must be positive.")
    m = np.asarray(m, dtype=float)
    return 1.0 / (m**2 + eps**2)


def total_variation_weights(m: np.ndarray, R, *, eps: float = 1e-6) -> np.ndarray:
    """IRLS row weights turning ``||R m||^2`` into a surrogate for the total-variation penalty ``||R m||_1``.

    ``1/(|R m|+eps)``, one weight per row of ``R``. Reweighting the quadratic smoothness penalty by this
    factor makes the effective penalty ``sum_k (R m)_k^2 / (|(R m)_k|+eps) \\approx sum_k |(R m)_k|`` for
    ``(R m)_k`` well above ``eps`` -- an L1 (total-variation) penalty on the model gradient, which
    penalizes the number of interfaces rather than how steep each one is, so the recovered model has
    sharp, blocky edges instead of a smoothly graded one. ``R`` is any roughness/gradient operator, e.g.
    :func:`~mixle_pde.geophysics.roughness_operator` or :func:`dip_rotated_gradient_operator`.
    """
    if eps <= 0.0:
        raise ValueError("eps must be positive.")
    m = np.asarray(m, dtype=float)
    rm = R @ m
    return 1.0 / (np.abs(rm) + eps)


def _axis_gradient_operators(shape: tuple[int, ...], spacing) -> list[sp.csr_matrix]:
    """One ``(n, n)`` directional first-difference operator per grid axis: a central difference in the
    interior, and a half-weighted one-sided difference at each of the two boundary layers along that
    axis. Built from the same grid connectivity (:func:`mixle.ppl._grid._grid_faces`) that
    :func:`~mixle_pde.geophysics.roughness_operator` uses, just kept separate per axis (instead of
    stacked into one face list) so the per-cell directional derivatives can be linearly recombined into
    a rotated basis.
    """
    from mixle.ppl._grid import _grid_faces

    shape = tuple(int(s) for s in shape)
    ndim = len(shape)
    spacing_arr = np.broadcast_to(np.asarray(spacing, dtype=float), (ndim,))
    g = _grid_faces(shape, spacing_arr)
    face_a, face_b = g["face_a"], g["face_b"]
    n = g["n"]
    ops = []
    offset = 0
    for ax in range(ndim):
        n_faces_ax = n - n // shape[ax]
        a = face_a[offset : offset + n_faces_ax]
        b = face_b[offset : offset + n_faces_ax]
        offset += n_faces_ax
        h = float(spacing_arr[ax])
        half = 0.5 / h
        rows = np.concatenate([a, a, b, b])
        cols = np.concatenate([b, a, b, a])
        data = np.concatenate(
            [np.full(len(a), half), np.full(len(a), -half), np.full(len(a), half), np.full(len(a), -half)]
        )
        ops.append(sp.csr_matrix((data, (rows, cols)), shape=(n, n)))
    return ops


def _strike_dip_rotation(ndim: int, strike_deg: float, dip_deg: float) -> np.ndarray:
    """The ``(ndim, ndim)`` rotation matrix mapping grid axes onto a structure-aligned basis.

    3-D grids are assumed ``(east, north, up)``: row 0 is the along-strike horizontal unit vector, row 1
    the down-dip unit vector (tilted below horizontal by ``dip_deg``, in the vertical plane
    perpendicular to strike), and row 2 the structure normal (their cross product). 2-D grids are treated
    as a single ``(x, z)`` cross-section, where ``dip_deg`` alone rotates the two grid axes in-plane
    (``strike_deg`` has no meaning in a 2-D profile and is ignored).
    """
    if ndim == 2:
        theta = np.radians(dip_deg)
        return np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
    if ndim == 3:
        strike = np.radians(strike_deg)
        dip = np.radians(dip_deg)
        strike_hat = np.array([np.sin(strike), np.cos(strike), 0.0])
        dip_azimuth = strike + np.pi / 2.0
        dip_hat = np.array([np.sin(dip_azimuth) * np.cos(dip), np.cos(dip_azimuth) * np.cos(dip), -np.sin(dip)])
        normal_hat = np.cross(strike_hat, dip_hat)
        return np.stack([strike_hat, dip_hat, normal_hat])
    raise ValueError("dip_rotated_gradient_operator supports 2-D or 3-D grids only.")


def dip_rotated_gradient_operator(shape, *, strike_deg: float, dip_deg: float, spacing=1.0) -> sp.csr_matrix:
    """A first-difference gradient operator rotated onto a structural strike/dip basis.

    Builds the per-axis directional-derivative stencils (:func:`_axis_gradient_operators`, which reuses
    the grid connectivity behind :func:`~mixle_pde.geophysics.roughness_operator`) and rotates them by
    the strike/dip rotation matrix (:func:`_strike_dip_rotation`), so the returned operator's rows are
    directional derivatives along-strike, down-dip, and (in 3-D) normal to the structure, rather than
    along the raw grid axes. Feeding this in place of :func:`~mixle_pde.geophysics.roughness_operator` as
    ``roughness=`` biases the smoothness penalty to follow known structural dip instead of being
    isotropic in grid space (``strike_deg=dip_deg=0`` recovers axis-aligned smoothing).

    Returns:
        scipy.sparse.csr_matrix of shape ``(ndim * n, n)`` (``ndim`` rotated directional derivatives per
        cell, stacked).
    """
    shape = tuple(int(s) for s in shape)
    ndim = len(shape)
    axis_ops = _axis_gradient_operators(shape, spacing)
    rot = _strike_dip_rotation(ndim, strike_deg, dip_deg)
    blocks = []
    for k in range(ndim):
        acc = rot[k, 0] * axis_ops[0]
        for j in range(1, ndim):
            acc = acc + rot[k, j] * axis_ops[j]
        blocks.append(acc.tocsr())
    return sp.vstack(blocks).tocsr()


def blocky_invert(
    forward: Callable,
    data,
    x0,
    *,
    prior: str = "tv",
    roughness=None,
    beta: float = 1.0,
    noise=1.0,
    strike_deg: float = 0.0,
    dip_deg: float = 0.0,
    shape=None,
    n_irls: int = 6,
    eps: float = 1e-6,
    **gn_kwargs,
):
    """Recover a blocky (``prior="tv"``) or compact (``prior="compact"``) body by an outer IRLS loop
    around :func:`~mixle_pde.geophysics.regularized_gauss_newton`.

    Each outer iteration fixes the IRLS weights at the current model estimate, runs one
    ``regularized_gauss_newton`` solve with ``reweight`` set to those frozen weights (which itself
    refreshes the reweighting every inner Gauss-Newton step), and starts the next outer iteration from
    the result -- the standard alternating scheme for turning a non-quadratic (L1 / minimum-support)
    model penalty into a sequence of quadratic solves.

    Args:
        forward, data, x0: as :func:`~mixle_pde.geophysics.regularized_gauss_newton`.
        prior: ``"tv"`` for the blocky/edge-preserving total-variation surrogate
            (:func:`total_variation_weights`, reweighting the *roughness* operator), or ``"compact"``
            (aliases ``"ms"``/``"minimum_support"``) for the Last-Kubik minimum-support surrogate
            (:func:`minimum_support_weights`, reweighting per-cell damping).
        roughness: explicit ``(k, n)`` roughness/gradient operator for ``prior="tv"``; if ``None`` and
            ``shape`` is given, built from :func:`dip_rotated_gradient_operator` when ``strike_deg`` or
            ``dip_deg`` is non-zero, else :func:`~mixle_pde.geophysics.roughness_operator`. Ignored for
            ``prior="compact"``, which always damps to the identity (compactness is a per-cell magnitude
            prior, not a gradient one).
        beta, noise: as :func:`~mixle_pde.geophysics.regularized_gauss_newton`.
        strike_deg, dip_deg: structural orientation fed to :func:`dip_rotated_gradient_operator` when the
            roughness operator is built from ``shape`` (``prior="tv"`` only).
        shape: grid shape, needed to build the default roughness/gradient operator.
        n_irls: number of outer IRLS (reweight-and-resolve) iterations.
        eps: IRLS regularization floor passed to the weight function.
        **gn_kwargs: forwarded to every inner :func:`~mixle_pde.geophysics.regularized_gauss_newton` call
            (e.g. ``lower``, ``upper``, ``ref``, ``n_iter``, ``jac_every``).

    Returns:
        ``(x, std)``, exactly as :func:`~mixle_pde.geophysics.regularized_gauss_newton` -- feed into a
        :class:`~mixle_pde.latent.PosteriorField3D` for an IC-1-conforming posterior.
    """
    x0 = np.asarray(x0, dtype=float)
    n = x0.size

    if prior == "tv":
        if roughness is not None:
            R = roughness.tocsr()
        elif shape is not None:
            if strike_deg or dip_deg:
                R = dip_rotated_gradient_operator(shape, strike_deg=strike_deg, dip_deg=dip_deg)
            else:
                R = roughness_operator(shape)
        else:
            R = sp.eye(n, format="csr")

        def reweight(m):
            return total_variation_weights(m, R, eps=eps)
    elif prior in ("compact", "ms", "minimum_support"):
        R = sp.eye(n, format="csr")

        def reweight(m):
            return minimum_support_weights(m, eps=eps)
    else:
        raise ValueError(f"unknown prior {prior!r}; expected 'tv' or 'compact'.")

    x = x0
    std = np.full(n, np.nan)
    for _ in range(n_irls):
        x, std = regularized_gauss_newton(
            forward, data, x, noise=noise, beta=beta, roughness=R, reweight=reweight, **gn_kwargs
        )
    return x, std
