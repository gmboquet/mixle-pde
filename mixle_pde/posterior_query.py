"""Posterior extraction and compact storage for latent-field posteriors (workstream G8).

Workstream G6/G7 produce :class:`~mixle_pde.latent.PosteriorField3D` Gaussian artifacts and
:class:`~mixle_pde.latent.PosteriorFieldSamples3D` empirical artifacts; this module is the query and
compression layer over them -- the APIs a consumer uses to pull answers out of a posterior without
re-running the inversion, and to store a large posterior compactly.

Extraction (every result in the field's PHYSICAL units where the transform is monotone, exact where the
transform is the identity):

* :func:`marginal_at_points` -- per-point posterior mean / std / credible interval at chosen cells.
* :func:`section` -- a marginal summary over an axis-aligned plane (thin wrapper on
  :meth:`PosteriorField3D.slice`).
* :func:`region_summary` -- mean / std of every cell inside a boolean region mask.
* :func:`derived_quantity` -- the EXACT Gaussian posterior of a linear functional ``w . m`` of the field
  (mean ``w . mu``, variance ``w^T Sigma w``); the honest way to ask "total anomalous mass in this
  block" or "average grade in this zone" and get a mean AND an uncertainty, since a linear functional of
  a Gaussian is Gaussian in closed form. :func:`region_mass` is the common special case (sum of
  ``field * cell_volume`` over a region).
* :func:`sampled_derived_quantity` -- the empirical version for sampled posteriors, preserving
  non-Gaussian shape by carrying derived samples.

Compression (compact storage for large posteriors, each returning a valid ``PosteriorField3D``):

* :func:`compress_to_low_rank` -- dense covariance -> rank-``k`` factor + residual diagonal, with the
  per-cell marginal variances preserved EXACTLY (the residual diagonal is set to close the gap), so
  intervals/marginals are unchanged while storage drops from ``O(n^2)`` to ``O(nk)``.
* :func:`to_diagonal` -- keep only per-cell marginal variances (independent-marginal summary).
* :func:`to_ensemble` -- a fixed set of posterior sample vectors (ensemble storage), for downstream
  code that consumes draws rather than a covariance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.special import ndtri

from mixle_pde.field_assimilation import PosteriorField4D, PosteriorFieldSamples4D
from mixle_pde.latent import Field3D, PosteriorField3D, PosteriorFieldSamples3D

Posterior3D = PosteriorField3D | PosteriorFieldSamples3D
Posterior4D = PosteriorField4D | PosteriorFieldSamples4D


@dataclass
class MarginalSummary:
    """Per-point marginal posterior in physical units."""

    indices: np.ndarray
    coordinates: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    lower: np.ndarray
    upper: np.ndarray


@dataclass
class MarginalTimeSeries:
    """Per-point marginal posterior through time, in physical units."""

    indices: np.ndarray
    times: np.ndarray
    coordinates: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    lower: np.ndarray
    upper: np.ndarray


@dataclass
class DerivedTimeSeries:
    """Posterior of a derived scalar quantity through time."""

    times: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    samples: np.ndarray | None = None

    def credible_interval(self, alpha: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be in (0, 1).")
        if self.samples is not None:
            lo, hi = np.quantile(self.samples, [alpha / 2.0, 1.0 - alpha / 2.0], axis=0)
            return lo, hi
        z = ndtri(1.0 - alpha / 2.0)
        return self.mean - z * self.std, self.mean + z * self.std


def marginal_at_points(posterior: Posterior3D, indices, *, alpha: float = 0.1) -> MarginalSummary:
    """Per-point posterior mean / std / ``1-alpha`` credible interval at the given cell ``indices``."""
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1).")
    idx = np.atleast_1d(np.asarray(indices, dtype=int))
    grid = posterior.grid
    if isinstance(posterior, PosteriorFieldSamples3D):
        physical = posterior.physical_samples
        mean_phys = np.mean(physical, axis=0)
        std_phys = np.std(physical, axis=0, ddof=1) if physical.shape[0] > 1 else np.zeros(grid.n)
        lo, hi = posterior.credible_interval(alpha)
    else:
        mean_phys = grid.from_unconstrained(posterior.mean)
        std_u = posterior.marginal_std
        z = ndtri(1.0 - alpha / 2.0)
        lo = grid.from_unconstrained(posterior.mean - z * std_u)
        hi = grid.from_unconstrained(posterior.mean + z * std_u)
        # physical-space per-point std via the local monotone map (exact for identity transform)
        std_phys = np.abs(grid.from_unconstrained(posterior.mean + std_u) - mean_phys)
    lower = np.minimum(lo, hi)
    upper = np.maximum(lo, hi)
    return MarginalSummary(
        indices=idx,
        coordinates=grid.coordinates[idx],
        mean=mean_phys[idx],
        std=std_phys[idx],
        lower=lower[idx],
        upper=upper[idx],
    )


def section(
    posterior: Posterior3D,
    *,
    x: float | None = None,
    y: float | None = None,
    z: float | None = None,
    tol: float = 1e-9,
) -> dict[str, Any]:
    """A marginal summary over an axis-aligned plane -- thin wrapper on :meth:`PosteriorField3D.slice`."""
    return posterior.slice(x=x, y=y, z=z, tol=tol)


def time_slice(posterior: Posterior4D, time: float, *, interpolate: bool = False) -> Posterior3D:
    """Return the 3D posterior slice at ``time``."""
    if isinstance(posterior, PosteriorField4D):
        return posterior.at_time(float(time), interpolate=interpolate)
    if interpolate:
        raise ValueError("sampled 4D posteriors only support exact stored times.")
    return posterior.at_time(float(time))


def region_summary(posterior: Posterior3D, mask) -> dict[str, Any]:
    """Per-cell mean / std (physical units) restricted to a boolean region ``mask`` over the grid."""
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != (posterior.grid.n,):
        raise ValueError(f"mask must have shape ({posterior.grid.n},).")
    summary = marginal_at_points(posterior, np.flatnonzero(mask))
    return {
        "n_cells": int(mask.sum()),
        "coordinates": summary.coordinates,
        "mean": summary.mean,
        "std": summary.std,
    }


def marginal_time_series(posterior: Posterior4D, indices, *, alpha: float = 0.1) -> MarginalTimeSeries:
    """Per-point posterior marginals across all stored times."""
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1).")
    idx = np.atleast_1d(np.asarray(indices, dtype=int))
    grid = posterior.grid
    if isinstance(posterior, PosteriorFieldSamples4D):
        physical = posterior.physical_samples
        mean = np.mean(physical, axis=0)[:, idx]
        std = np.std(physical, axis=0, ddof=1)[:, idx] if posterior.n_samples > 1 else np.zeros_like(mean)
        lo_all, hi_all = posterior.credible_interval(alpha)
        lo = lo_all[:, idx]
        hi = hi_all[:, idx]
    else:
        mean = grid.from_unconstrained(posterior.mean_array)[:, idx]
        std_u = posterior.marginal_std
        z = ndtri(1.0 - alpha / 2.0)
        lo = grid.from_unconstrained(posterior.mean_array - z * std_u)[:, idx]
        hi = grid.from_unconstrained(posterior.mean_array + z * std_u)[:, idx]
        std = np.abs(grid.from_unconstrained(posterior.mean_array + std_u) - grid.from_unconstrained(posterior.mean_array))[
            :, idx
        ]
    return MarginalTimeSeries(
        indices=idx,
        times=posterior.times.copy(),
        coordinates=grid.coordinates[idx],
        mean=mean,
        std=std,
        lower=np.minimum(lo, hi),
        upper=np.maximum(lo, hi),
    )


def region_time_summary(posterior: Posterior4D, mask) -> dict[str, Any]:
    """Per-cell mean/std over a region through time."""
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != (posterior.grid.n,):
        raise ValueError(f"mask must have shape ({posterior.grid.n},).")
    summary = marginal_time_series(posterior, np.flatnonzero(mask))
    return {
        "n_cells": int(mask.sum()),
        "times": summary.times,
        "coordinates": summary.coordinates,
        "mean": summary.mean,
        "std": summary.std,
    }


@dataclass
class DerivedQuantity:
    """The exact Gaussian posterior of a linear functional of the field: ``value ~ N(mean, std^2)``."""

    mean: float
    std: float

    def credible_interval(self, alpha: float = 0.1) -> tuple[float, float]:
        z = ndtri(1.0 - alpha / 2.0)
        return self.mean - z * self.std, self.mean + z * self.std


@dataclass
class SampledDerivedQuantity:
    """An empirical posterior of a derived scalar quantity, preserving sampled posterior shape."""

    samples: np.ndarray

    def __post_init__(self) -> None:
        samples = np.asarray(self.samples, dtype=float).reshape(-1)
        if samples.size == 0:
            raise ValueError("samples must contain at least one derived draw.")
        if not np.all(np.isfinite(samples)):
            raise ValueError("samples must be finite.")
        self.samples = samples

    @property
    def mean(self) -> float:
        return float(np.mean(self.samples))

    @property
    def std(self) -> float:
        return float(np.std(self.samples, ddof=1)) if self.samples.size > 1 else 0.0

    def credible_interval(self, alpha: float = 0.1) -> tuple[float, float]:
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be in (0, 1).")
        lo, hi = np.quantile(self.samples, [alpha / 2.0, 1.0 - alpha / 2.0])
        return float(lo), float(hi)


def derived_quantity(posterior: PosteriorField3D, weights) -> DerivedQuantity:
    """Exact posterior of the linear functional ``w . m``: mean ``w . mu``, variance ``w^T Sigma w``.

    Acts on the field's (unconstrained) Gaussian variable -- exact in physical units when the transform
    is the identity (``bounds=None``), which is the case for the linear-Gaussian inversion path. A
    linear functional of a Gaussian is Gaussian, so this is closed-form, not sampled.
    """
    w = np.asarray(weights, dtype=float).reshape(-1)
    if w.shape != (posterior.grid.n,):
        raise ValueError(f"weights must have shape ({posterior.grid.n},).")
    mean = float(w @ posterior.mean)
    if posterior.cov is not None:
        var = float(w @ posterior.cov @ w)
    elif posterior.precision_factor is not None:
        cov_w = posterior.precision_factor.solve(w)
        var = float(w @ cov_w)
    elif posterior.low_rank is not None:
        proj = posterior.low_rank.T @ w
        var = float(proj @ proj + np.sum(posterior.diag_var * w**2))
    else:
        var = float(np.sum(posterior.diag_var * w**2))
    return DerivedQuantity(mean=mean, std=float(np.sqrt(max(var, 0.0))))


def sampled_derived_quantity(posterior: PosteriorFieldSamples3D, weights) -> SampledDerivedQuantity:
    """Empirical posterior of a linear functional over physical posterior samples."""
    w = np.asarray(weights, dtype=float).reshape(-1)
    if w.shape != (posterior.grid.n,):
        raise ValueError(f"weights must have shape ({posterior.grid.n},).")
    return SampledDerivedQuantity(posterior.physical_samples @ w)


def region_mass(posterior: Posterior3D, mask, cell_volumes) -> DerivedQuantity | SampledDerivedQuantity:
    """Posterior of total anomalous mass in a region: ``sum_{i in mask} field_i * volume_i``.

    The common derived quantity for a subsurface body -- a mean tonnage/mass AND its uncertainty, in one
    closed-form Gaussian for Gaussian posteriors or as derived samples for empirical posteriors.
    """
    mask = np.asarray(mask, dtype=bool)
    vols = np.broadcast_to(np.asarray(cell_volumes, dtype=float), (posterior.grid.n,))
    weights = np.where(mask, vols, 0.0)
    if isinstance(posterior, PosteriorFieldSamples3D):
        return sampled_derived_quantity(posterior, weights)
    return derived_quantity(posterior, weights)


def region_mass_time_series(posterior: Posterior4D, mask, cell_volumes) -> DerivedTimeSeries:
    """Posterior of region mass through time."""
    mask = np.asarray(mask, dtype=bool)
    vols = np.broadcast_to(np.asarray(cell_volumes, dtype=float), (posterior.grid.n,))
    if mask.shape != (posterior.grid.n,):
        raise ValueError(f"mask must have shape ({posterior.grid.n},).")
    weights = np.where(mask, vols, 0.0)
    if isinstance(posterior, PosteriorFieldSamples4D):
        samples = np.einsum("stn,n->st", posterior.physical_samples, weights)
        std = np.std(samples, axis=0, ddof=1) if samples.shape[0] > 1 else np.zeros(samples.shape[1])
        return DerivedTimeSeries(
            times=posterior.times.copy(),
            mean=np.mean(samples, axis=0),
            std=std,
            samples=samples,
        )
    quantities = [region_mass(posterior.at_time(float(time)), mask, weights) for time in posterior.times]
    return DerivedTimeSeries(
        times=posterior.times.copy(),
        mean=np.asarray([quantity.mean for quantity in quantities], dtype=float),
        std=np.asarray([quantity.std for quantity in quantities], dtype=float),
    )


def compress_to_low_rank(posterior: PosteriorField3D, rank: int) -> PosteriorField3D:
    """Compress a dense-covariance posterior to a rank-``k`` factor + residual diagonal.

    Truncated eigendecomposition of the covariance keeps the ``rank`` leading modes; the residual
    diagonal is set so every cell's MARGINAL variance is preserved exactly -- so credible intervals and
    per-cell marginals are unchanged, while storage drops from ``O(n^2)`` to ``O(n * rank)``.
    """
    if posterior.cov is None:
        raise ValueError("compress_to_low_rank needs a dense-covariance posterior.")
    n = posterior.grid.n
    if not 1 <= rank <= n:
        raise ValueError(f"rank must be in [1, {n}].")
    cov = posterior.cov
    vals, vecs = np.linalg.eigh(cov)  # ascending
    keep = slice(n - rank, n)
    lam = np.clip(vals[keep], 0.0, None)
    U = vecs[:, keep]
    low_rank = U * np.sqrt(lam)[None, :]  # (n, rank)
    resid = np.diag(cov) - np.sum(low_rank**2, axis=1)
    diag_var = np.clip(resid, 0.0, None)  # preserve marginal variance exactly (up to the clip floor)
    return PosteriorField3D(
        grid=posterior.grid,
        mean=posterior.mean.copy(),
        map=posterior.map.copy(),
        low_rank=low_rank,
        diag_var=diag_var,
    )


def to_diagonal(posterior: PosteriorField3D) -> PosteriorField3D:
    """Keep only per-cell marginal variances (independent-marginal summary storage)."""
    return PosteriorField3D(
        grid=posterior.grid,
        mean=posterior.mean.copy(),
        map=posterior.map.copy(),
        diag_var=posterior.marginal_variance,
    )


@dataclass
class PosteriorEnsemble:
    """Ensemble storage: a fixed set of posterior sample vectors (physical units) over a grid."""

    grid: Field3D
    samples: np.ndarray  # (n_samples, grid.n)

    def mean(self) -> np.ndarray:
        return self.samples.mean(axis=0)

    def std(self) -> np.ndarray:
        return self.samples.std(axis=0)


def to_ensemble(posterior: Posterior3D, n_samples: int, rng: np.random.Generator) -> PosteriorEnsemble:
    """Draw or resample a fixed posterior ensemble (physical units) for downstream consumers."""
    return PosteriorEnsemble(grid=posterior.grid, samples=posterior.sample(int(n_samples), rng))
