"""Latent 3D/4D field objects and posterior artifacts (workstream G1, first card).

Storage-agnostic containers for a gridded physical property: :class:`Field3D` fixes the grid
geometry, units, and (optional) bound constraints for a named property; :class:`PosteriorField3D`
carries a Gaussian posterior over that property (mean/MAP, marginal std, and dense, sparse-precision,
or compact covariance storage) with sampling, credible intervals, and axis-aligned slicing.

This card defines the objects only. Inversion lives in :mod:`mixle_pde.field_inversion`,
:mod:`mixle_pde.field_gauss_newton`, and :mod:`mixle_pde.field_assimilation`; forward mappings live in
:mod:`mixle_pde.observations` and the geophysics modules.

Bound transforms reuse the same log / logit convention as
:meth:`Field3D.to_unconstrained`: a property with both bounds gets a logit transform, one-sided bounds
get a log transform anchored at the bound, and an unbounded property is the identity. A
:class:`PosteriorField3D`'s Gaussian lives in this unconstrained space; samples and credible intervals
are mapped back through :meth:`Field3D.from_unconstrained` into physical units (monotone, so interval
endpoints and ordering survive the map).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.special import ndtri


@dataclass
class SparsePosteriorPrecision:
    """Sparse precision storage and factor solves for a Gaussian posterior.

    ``precision`` is the posterior precision matrix ``Lambda = Sigma^-1``. The LU factor is built
    lazily once and then reused for covariance actions ``Sigma v = Lambda^-1 v``. This stores the
    posterior in sparse precision form without materializing the dense covariance.
    """

    precision: Any
    _factor: Any | None = field(default=None, init=False, repr=False)
    _marginal_variance: np.ndarray | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        import scipy.sparse as sp

        mat = sp.csc_matrix(self.precision)
        if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
            raise ValueError("precision must be a square matrix.")
        self.precision = mat

    @property
    def n(self) -> int:
        return int(self.precision.shape[0])

    def factor(self):
        """Return the cached sparse LU factor."""
        if self._factor is None:
            from scipy.sparse.linalg import splu

            self._factor = splu(self.precision)
        return self._factor

    def solve(self, rhs) -> np.ndarray:
        """Apply the posterior covariance to ``rhs`` by solving ``precision x = rhs``."""
        arr = np.asarray(rhs, dtype=float)
        if arr.ndim == 1:
            if arr.shape != (self.n,):
                raise ValueError(f"rhs must have shape ({self.n},), got {arr.shape}.")
            return np.asarray(self.factor().solve(arr), dtype=float)
        if arr.ndim == 2:
            if arr.shape[0] != self.n:
                raise ValueError(f"rhs must have shape ({self.n}, k), got {arr.shape}.")
            return np.asarray(self.factor().solve(arr), dtype=float)
        raise ValueError("rhs must be a vector or a 2D matrix.")

    def marginal_variance(self) -> np.ndarray:
        """Diagonal of the dense covariance ``precision^-1`` computed by sparse solves."""
        if self._marginal_variance is None:
            eye = np.eye(self.n)
            inv_columns = self.solve(eye)
            self._marginal_variance = np.clip(np.diag(inv_columns), 0.0, None)
        return self._marginal_variance.copy()

    def covariance_dense(self) -> np.ndarray:
        """Materialize the dense covariance on demand for small reference sampling/export."""
        return self.solve(np.eye(self.n))


@dataclass
class Field3D:
    """Grid geometry and units for one named physical property, storage-agnostic w.r.t. any posterior."""

    coordinates: np.ndarray
    spacing: float | tuple[float, float, float]
    units: str
    property_name: str
    bounds: tuple[float | None, float | None] | None = None
    mask: np.ndarray | None = None
    provenance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        coords = np.asarray(self.coordinates, dtype=float)
        if coords.ndim != 2 or coords.shape[1] != 3:
            raise ValueError("coordinates must be an (n, 3) array of (x, y, z) grid points.")
        self.coordinates = coords
        if self.mask is not None:
            mask = np.asarray(self.mask, dtype=bool)
            if mask.shape != (coords.shape[0],):
                raise ValueError("mask must have shape (n,) matching coordinates.")
            self.mask = mask
        if self.bounds is not None:
            lo, hi = self.bounds
            if lo is not None and hi is not None and lo >= hi:
                raise ValueError("bounds must satisfy lo < hi.")
        self.provenance = dict(self.provenance)

    @property
    def n(self) -> int:
        return self.coordinates.shape[0]

    def to_unconstrained(self, values: Any, *, eps: float = 1.0e-12) -> np.ndarray:
        """Map physical (bounded) property values into an unconstrained real-valued space."""
        arr = np.asarray(values, dtype=float)
        if self.bounds is None:
            return arr.copy()
        lo, hi = self.bounds
        if lo is not None and hi is not None:
            scaled = np.clip((arr - lo) / (hi - lo), eps, 1.0 - eps)
            return np.log(scaled / (1.0 - scaled))
        if lo is not None:
            return np.log(np.maximum(arr - lo, eps))
        if hi is not None:
            return np.log(np.maximum(hi - arr, eps))
        return arr.copy()

    def from_unconstrained(self, values: Any) -> np.ndarray:
        """Inverse of :meth:`to_unconstrained`: unconstrained real values back to physical units."""
        arr = np.asarray(values, dtype=float)
        if self.bounds is None:
            return arr.copy()
        lo, hi = self.bounds
        if lo is not None and hi is not None:
            sigmoid = 1.0 / (1.0 + np.exp(-arr))
            return lo + (hi - lo) * sigmoid
        if lo is not None:
            return lo + np.exp(arr)
        if hi is not None:
            return hi - np.exp(arr)
        return arr.copy()


@dataclass
class PosteriorField3D:
    """A Gaussian posterior over a :class:`Field3D`, in the field's unconstrained space.

    Covariance storage is one of four modes, chosen by which of ``cov`` / ``precision_factor`` /
    ``low_rank`` + ``diag_var`` / ``diag_var`` alone is supplied:

    * dense: ``cov`` is the full ``(n, n)`` covariance.
    * sparse precision: ``precision_factor`` stores ``cov^-1`` and sparse solves.
    * low-rank + diagonal: ``cov = low_rank @ low_rank.T + diag(diag_var)`` with ``low_rank``
      shape ``(n, k)``.
    * diagonal only: ``diag_var`` alone, i.e. independent marginals.
    """

    grid: Field3D
    mean: np.ndarray
    map: np.ndarray | None = None
    cov: np.ndarray | None = None
    precision_factor: SparsePosteriorPrecision | None = None
    low_rank: np.ndarray | None = None
    diag_var: np.ndarray | None = None

    def __post_init__(self) -> None:
        n = self.grid.n
        mean = np.asarray(self.mean, dtype=float)
        if mean.shape != (n,):
            raise ValueError(f"mean must have shape ({n},) matching the grid.")
        self.mean = mean
        self.map = mean.copy() if self.map is None else np.asarray(self.map, dtype=float)
        if self.map.shape != (n,):
            raise ValueError(f"map must have shape ({n},) matching the grid.")

        modes = [
            self.cov is not None,
            self.precision_factor is not None,
            self.low_rank is not None or self.diag_var is not None,
        ]
        if sum(bool(mode) for mode in modes) != 1:
            raise ValueError("supply exactly one covariance mode: `cov`, `precision_factor`, or `low_rank`/`diag_var`.")
        if not any(modes):
            raise ValueError("supply a covariance mode.")

        if self.cov is not None:
            cov = np.asarray(self.cov, dtype=float)
            if cov.shape != (n, n):
                raise ValueError(f"cov must have shape ({n}, {n}).")
            self.cov = cov
        if self.precision_factor is not None and self.precision_factor.n != n:
            raise ValueError(f"precision_factor must have shape ({n}, {n}).")
        if self.low_rank is not None:
            low_rank = np.asarray(self.low_rank, dtype=float)
            if low_rank.ndim != 2 or low_rank.shape[0] != n:
                raise ValueError(f"low_rank must have shape ({n}, k).")
            self.low_rank = low_rank
            if self.diag_var is None:
                raise ValueError("low_rank requires diag_var (the residual diagonal variance).")
        if self.diag_var is not None:
            diag_var = np.asarray(self.diag_var, dtype=float)
            if diag_var.shape != (n,):
                raise ValueError(f"diag_var must have shape ({n},).")
            if np.any(diag_var < 0.0):
                raise ValueError("diag_var must be non-negative.")
            self.diag_var = diag_var

    @property
    def marginal_variance(self) -> np.ndarray:
        """Per-point posterior variance in unconstrained space, whichever storage mode is active."""
        if self.cov is not None:
            return np.diag(self.cov).copy()
        if self.precision_factor is not None:
            return self.precision_factor.marginal_variance()
        if self.low_rank is not None:
            return np.sum(self.low_rank**2, axis=1) + self.diag_var
        return self.diag_var.copy()

    @property
    def marginal_std(self) -> np.ndarray:
        return np.sqrt(self.marginal_variance)

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Draw ``n`` samples, mapped into physical units; shape ``(n, grid.n)``."""
        m = self.grid.n
        if self.cov is not None:
            chol = np.linalg.cholesky(self.cov + 1e-12 * np.eye(m))
            z = rng.standard_normal((n, m))
            unconstrained = self.mean[None, :] + z @ chol.T
        elif self.precision_factor is not None:
            cov = self.precision_factor.covariance_dense()
            chol = np.linalg.cholesky(cov + 1e-12 * np.eye(m))
            z = rng.standard_normal((n, m))
            unconstrained = self.mean[None, :] + z @ chol.T
        elif self.low_rank is not None:
            k = self.low_rank.shape[1]
            z = rng.standard_normal((n, k))
            eps = rng.standard_normal((n, m)) * np.sqrt(self.diag_var)[None, :]
            unconstrained = self.mean[None, :] + z @ self.low_rank.T + eps
        else:
            eps = rng.standard_normal((n, m)) * np.sqrt(self.diag_var)[None, :]
            unconstrained = self.mean[None, :] + eps
        return self.grid.from_unconstrained(unconstrained)

    def credible_interval(self, alpha: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
        """Per-point central credible interval covering ``1 - alpha`` mass, in physical units."""
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be in (0, 1).")
        z = ndtri(1.0 - alpha / 2.0)
        std = self.marginal_std
        lo_u = self.mean - z * std
        hi_u = self.mean + z * std
        return self.grid.from_unconstrained(lo_u), self.grid.from_unconstrained(hi_u)

    def slice(
        self, *, x: float | None = None, y: float | None = None, z: float | None = None, tol: float = 1e-9
    ) -> dict[str, Any]:
        """Select the grid points matching the given fixed axis coordinate(s), within ``tol``.

        Returns a dict with the restricted ``coordinates``, ``mean``, ``map``, and
        ``marginal_std`` (all physical units) plus the boolean ``index`` mask into the full grid.
        """
        fixed = [(axis, value) for axis, value in (("x", x), ("y", y), ("z", z)) if value is not None]
        if not fixed:
            raise ValueError("slice() requires at least one of x, y, z.")
        axis_index = {"x": 0, "y": 1, "z": 2}
        index = np.ones(self.grid.n, dtype=bool)
        for axis, value in fixed:
            index &= np.isclose(self.grid.coordinates[:, axis_index[axis]], value, atol=tol)
        mean_physical = self.grid.from_unconstrained(self.mean)
        map_physical = self.grid.from_unconstrained(self.map)
        std_physical_hi = self.grid.from_unconstrained(self.mean + self.marginal_std) - mean_physical
        return {
            "index": index,
            "coordinates": self.grid.coordinates[index],
            "mean": mean_physical[index],
            "map": map_physical[index],
            "marginal_std": np.abs(std_physical_hi)[index],
        }
