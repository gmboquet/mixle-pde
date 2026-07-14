"""Latent 3D/4D field objects and posterior artifacts (workstream G1, first card).

Storage-agnostic containers for a physical property: :class:`Field3D` fixes the 3D geometry, units, and
(optional) bound constraints for a named property on grid points or simplex-mesh nodes;
:class:`Field4D` adds a time axis and optional moving mesh geometry; :class:`PosteriorField3D` carries a
Gaussian posterior over that property (mean/MAP, marginal std, and dense, sparse-precision, or compact
covariance storage) with sampling, credible intervals, and axis-aligned slicing.

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

import json
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.sparse.linalg import LinearOperator
from scipy.special import ndtri

#: The frozen provenance key IC-2 (`mixle_pde.io.artifacts`) writes its saved-artifact digest under;
#: `Field3D.attach_content_hash` stamps a lineage edge onto the SAME key so a receipt walking
#: data -> inversion -> interpretation -> decision (E7) reads one consistent name at every hop.
PROVENANCE_HASH_KEY = "content_hash"


def _json_safe(value: Any) -> Any:
    """Coerce ``value`` into a JSON-serialisable equivalent.

    Numpy scalars/arrays become plain Python types; dicts/lists/tuples are walked recursively.
    Everything else passes through unchanged (so a value ``json.dumps`` cannot handle still raises,
    rather than being silently swallowed here -- see :meth:`Field3D.attach_content_hash`).
    """
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


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
    _marginal_variance_method: str | None = field(default=None, init=False, repr=False)

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

    def marginal_variance(self, *, method: str = "selected_inversion") -> np.ndarray:
        """Diagonal of the covariance ``precision^-1``; never forms the dense ``(n, n)`` inverse.

        ``method="selected_inversion"`` (default, C2) runs the Takahashi/Erisman-Tinney recursion over
        the sparse Cholesky factor's own nonzero pattern -- memory ``O(nnz(L))``, so a survey-scale
        precision (n ~ 10^5-10^6) never materializes an infeasible dense inverse.
        ``method="dense"`` keeps the original ``solve(eye)`` path (an explicit opt-in, for small
        problems or as a reference cross-check).
        """
        if method not in ("selected_inversion", "dense"):
            raise ValueError(f"unknown marginal_variance method {method!r}; expected 'selected_inversion' or 'dense'.")
        if self._marginal_variance is None or self._marginal_variance_method != method:
            if method == "dense":
                eye = np.eye(self.n)
                inv_columns = self.solve(eye)
                variance = np.diag(inv_columns)
            else:
                from mixle_pde.uq_lowrank import takahashi_selected_inversion

                variance = takahashi_selected_inversion(self.precision)
            self._marginal_variance = np.clip(np.asarray(variance, dtype=float), 0.0, None)
            self._marginal_variance_method = method
        return self._marginal_variance.copy()

    def covariance_dense(self) -> np.ndarray:
        """Materialize the dense covariance on demand for small reference sampling/export."""
        return self.solve(np.eye(self.n))


@dataclass
class Field3D:
    """3D geometry and units for one named physical property.

    ``coordinates`` are node or grid-point coordinates with shape ``(n, 3)``. ``mesh`` may be a
    :class:`mixle_pde.mesh.SimplexMesh` with matching 3D nodes, which lets the same latent-field object
    represent an unstructured tetrahedral domain without changing the posterior API.
    """

    coordinates: np.ndarray
    spacing: float | tuple[float, float, float]
    units: str
    property_name: str
    bounds: tuple[float | None, float | None] | None = None
    mask: np.ndarray | None = None
    provenance: dict[str, Any] = field(default_factory=dict)
    mesh: Any | None = None

    def __post_init__(self) -> None:
        coords = np.asarray(self.coordinates, dtype=float)
        if coords.ndim != 2 or coords.shape[1] != 3:
            raise ValueError("coordinates must be an (n, 3) array of (x, y, z) grid points.")
        self.coordinates = coords
        if self.mesh is not None:
            self._validate_mesh(self.mesh)
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

    @property
    def geometry_kind(self) -> str:
        """Return ``"simplex_mesh"`` for meshed fields, otherwise ``"point_grid"``."""
        return "simplex_mesh" if self.mesh is not None else "point_grid"

    def serialise_provenance(self) -> dict[str, Any]:
        """A JSON-safe copy of ``provenance`` (numpy arrays/scalars coerced to plain Python).

        This is the shape the sibling artifact header (IC-2's ``{path}.json``) writes to disk, so a
        caller's free-form in-memory ``provenance`` dict -- built up over an inversion run with whatever
        arrays/scalars it produced -- always round-trips through ``json.dumps`` once persisted.
        """
        return _json_safe(self.provenance)

    def attach_content_hash(self, content_hash: str, **extra: Any) -> None:
        """Stamp a content-hash lineage edge onto this field's ``provenance``, in place.

        ``content_hash`` is the IC-2 digest of the saved posterior artifact (the value
        ``mixle_pde.io.artifacts.save_posterior``/``content_hash`` computes over the artifact's
        arrays); ``extra`` keyword values (e.g. ``stage="inversion"``, ``parent=<upstream hash>``) are
        merged alongside it under their own keys. Replaces whatever a caller's free-form in-memory
        dict held for ``PROVENANCE_HASH_KEY`` (work-plan E7 algorithm step 1: "attach the IC-2
        content_hash ... to its provenance and serialise it -- no more free-form in-memory dict").

        The merged result is validated JSON-safe immediately (``json.dumps`` on
        :meth:`serialise_provenance`), so a value the free-form dict cannot survive is caught here, at
        attach time, rather than failing silently later when an artifact header is actually written.
        """
        self.provenance = {**self.provenance, PROVENANCE_HASH_KEY: str(content_hash), **extra}
        json.dumps(self.serialise_provenance())

    @classmethod
    def from_mesh(
        cls,
        mesh: Any,
        *,
        spacing: float | tuple[float, float, float],
        units: str,
        property_name: str,
        bounds: tuple[float | None, float | None] | None = None,
        mask: np.ndarray | None = None,
        provenance: dict[str, Any] | None = None,
    ) -> Field3D:
        """Construct a field over the nodes of a 3D simplex mesh."""
        return cls(
            coordinates=np.asarray(mesh.nodes, dtype=float),
            spacing=spacing,
            units=units,
            property_name=property_name,
            bounds=bounds,
            mask=mask,
            provenance={} if provenance is None else provenance,
            mesh=mesh,
        )

    @property
    def cell_measures(self) -> np.ndarray | None:
        """Simplex lengths/areas/volumes for meshed fields, otherwise ``None``."""
        if self.mesh is None:
            return None
        return self.mesh.simplex_measures()

    def _validate_mesh(self, mesh: Any) -> None:
        if getattr(mesh, "dim", None) != 3:
            raise ValueError("Field3D mesh must be three-dimensional.")
        nodes = np.asarray(getattr(mesh, "nodes", None), dtype=float)
        if nodes.shape != self.coordinates.shape:
            raise ValueError("Field3D mesh nodes must match coordinates shape.")
        if not np.allclose(nodes, self.coordinates):
            raise ValueError("Field3D mesh nodes must match coordinates.")

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
class Field4D:
    """A time-indexed 3D field object with optional moving-domain geometry."""

    spatial: Field3D
    times: np.ndarray
    provenance: dict[str, Any] = field(default_factory=dict)
    moving_mesh: Any | None = None

    def __post_init__(self) -> None:
        times = np.asarray(self.times, dtype=float).reshape(-1)
        if times.size == 0:
            raise ValueError("times must be a non-empty 1-D array.")
        if np.any(np.diff(times) <= 0.0):
            raise ValueError("times must be strictly increasing.")
        if self.moving_mesh is not None:
            self._validate_moving_mesh(self.moving_mesh, times)
        self.times = times
        self.provenance = dict(self.provenance)

    @property
    def n_times(self) -> int:
        return int(self.times.size)

    @property
    def n_per_time(self) -> int:
        return int(self.spatial.n)

    @property
    def n(self) -> int:
        return int(self.n_times * self.n_per_time)

    @property
    def property_name(self) -> str:
        return self.spatial.property_name

    @property
    def units(self) -> str:
        return self.spatial.units

    @property
    def bounds(self) -> tuple[float | None, float | None] | None:
        return self.spatial.bounds

    @property
    def geometry_kind(self) -> str:
        if self.moving_mesh is not None:
            return "moving_simplex_mesh"
        return f"static_{self.spatial.geometry_kind}"

    @property
    def coordinates(self) -> np.ndarray:
        """Space-time coordinates with shape ``(n_times * n_per_time, 4)``."""
        layers = [np.column_stack([self.coordinates_at_time(t), np.full(self.n_per_time, t)]) for t in self.times]
        return np.vstack(layers)

    def coordinates_at_time(self, time: float, *, interpolate: bool = True) -> np.ndarray:
        """3D coordinates at ``time`` using moving geometry when available."""
        if self.moving_mesh is None:
            return self.spatial.coordinates.copy()
        if interpolate:
            return self.moving_mesh.at_time(float(time), clamp=False).nodes
        idx = self._index_of(float(time))
        return self.moving_mesh.at_step(idx).nodes

    def mesh_at_time(self, time: float, *, interpolate: bool = True) -> Any | None:
        """Return the simplex mesh at ``time`` when moving or static mesh geometry is available."""
        if self.moving_mesh is not None:
            return (
                self.moving_mesh.at_time(float(time), clamp=False)
                if interpolate
                else self.moving_mesh.at_step(self._index_of(float(time)))
            )
        return self.spatial.mesh

    def values_at_time(self, values: Any, time: float) -> np.ndarray:
        """Extract one time slice from a ``(n_times, n_per_time)`` or flattened value array."""
        arr = self.reshape_values(values)
        return arr[self._index_of(float(time))].copy()

    def reshape_values(self, values: Any) -> np.ndarray:
        """Return values as ``(n_times, n_per_time)`` and validate their size."""
        arr = np.asarray(values, dtype=float)
        if arr.shape == (self.n_times, self.n_per_time):
            return arr.copy()
        if arr.shape == (self.n,):
            return arr.reshape(self.n_times, self.n_per_time)
        raise ValueError(f"values must have shape ({self.n_times}, {self.n_per_time}) or ({self.n},).")

    def to_unconstrained(self, values: Any) -> np.ndarray:
        """Map physical time-indexed values to unconstrained space."""
        return self.spatial.to_unconstrained(self.reshape_values(values))

    def from_unconstrained(self, values: Any) -> np.ndarray:
        """Map unconstrained time-indexed values to physical units."""
        return self.spatial.from_unconstrained(self.reshape_values(values))

    def _index_of(self, time: float, tol: float = 1e-9) -> int:
        matches = np.flatnonzero(np.isclose(self.times, time, atol=tol))
        if matches.size == 0:
            raise KeyError(f"time {time!r} is not in this Field4D; have {self.times.tolist()}.")
        return int(matches[0])

    def _validate_moving_mesh(self, moving_mesh: Any, times: np.ndarray) -> None:
        if getattr(moving_mesh, "dim", None) != 3:
            raise ValueError("Field4D moving_mesh must contain 3D spatial meshes.")
        if getattr(moving_mesh, "n_nodes", None) != self.spatial.n:
            raise ValueError("Field4D moving_mesh node count must match the spatial field.")
        if len(getattr(moving_mesh, "times", [])) != times.size:
            raise ValueError("Field4D moving_mesh must have the same number of times.")
        if not np.allclose(moving_mesh.times, times):
            raise ValueError("Field4D moving_mesh times must match Field4D times.")


_DENSE_COV_MAX_N = 2048
"""Grid size at/below which `PosteriorField3D.cov`/`PosteriorFieldSamples3D.cov` materialize a
dense array; above it, they return a matrix-free `LinearOperator` (IC-1; work-plan Sec.C2)."""


class _PosteriorDerivedQuantity:
    """A pushforward of a posterior through a functional (IC-1 `DerivedQuantity`): the draws, a
    central credible interval, and the `prior_dominated` honesty flag. `prior_dominated` is set
    from A2's variance-reduction hook once that lands (work-plan A2); it defaults to False here,
    since that hook is out of scope for this task.
    """

    def __init__(self, samples: np.ndarray, *, prior_dominated: bool = False) -> None:
        self.samples = np.asarray(samples)
        self.prior_dominated = bool(prior_dominated)

    def credible_interval(self, level: float) -> tuple[np.ndarray, np.ndarray]:
        """Central ``level`` interval (e.g. 0.9) of the derived quantity."""
        if not 0.0 < level < 1.0:
            raise ValueError("level must be in (0, 1).")
        alpha = 1.0 - level
        lo = np.quantile(self.samples, alpha / 2.0, axis=0)
        hi = np.quantile(self.samples, 1.0 - alpha / 2.0, axis=0)
        return lo, hi


@dataclass
class PosteriorField3D:
    """A Gaussian posterior over a :class:`Field3D`, in the field's unconstrained space.

    Covariance storage is one of four modes, chosen by which of ``dense_cov`` / ``precision_factor``
    / ``low_rank`` + ``diag_var`` / ``diag_var`` alone is supplied:

    * dense: ``dense_cov`` is the full ``(n, n)`` covariance.
    * sparse precision: ``precision_factor`` stores ``cov^-1`` and sparse solves.
    * low-rank + diagonal: ``cov = low_rank @ low_rank.T + diag(diag_var)`` with ``low_rank``
      shape ``(n, k)``.
    * diagonal only: ``diag_var`` alone, i.e. independent marginals.

    ``dense_cov`` is the raw, mode-discriminating storage slot (``None`` unless the dense mode was
    explicitly supplied at construction -- callers use ``dense_cov is None`` to detect the other
    three modes). The IC-1 `cov` **property** below is the always-populated read surface: it returns
    ``dense_cov`` unchanged in dense mode, and otherwise materializes a dense array (small grids) or
    a matrix-free `LinearOperator` (large grids) from whichever mode is active -- it is a *view*, not
    a fourth storage mode.
    """

    grid: Field3D
    mean: np.ndarray
    map: np.ndarray | None = None
    dense_cov: np.ndarray | None = None
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
            self.dense_cov is not None,
            self.precision_factor is not None,
            self.low_rank is not None or self.diag_var is not None,
        ]
        if sum(bool(mode) for mode in modes) != 1:
            raise ValueError(
                "supply exactly one covariance mode: `dense_cov`, `precision_factor`, or `low_rank`/`diag_var`."
            )
        if not any(modes):
            raise ValueError("supply a covariance mode.")

        if self.dense_cov is not None:
            dense_cov = np.asarray(self.dense_cov, dtype=float)
            if dense_cov.shape != (n, n):
                raise ValueError(f"dense_cov must have shape ({n}, {n}).")
            self.dense_cov = dense_cov
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
        if self.dense_cov is not None:
            return np.diag(self.dense_cov).copy()
        if self.precision_factor is not None:
            return self.precision_factor.marginal_variance()
        if self.low_rank is not None:
            return np.sum(self.low_rank**2, axis=1) + self.diag_var
        return self.diag_var.copy()

    @property
    def marginal_std(self) -> np.ndarray:
        return np.sqrt(self.marginal_variance)

    @property
    def cov(self) -> np.ndarray | LinearOperator:
        """IC-1 `Posterior.cov`: covariance in the field's unconstrained space (the same space as
        `mean`) -- a dense `(grid.n, grid.n)` array when explicitly supplied (`dense_cov`) or small
        enough to materialize cheaply, otherwise a matrix-free `LinearOperator` backed by whichever
        storage mode (`precision_factor` or `low_rank`/`diag_var`) is active. Never materializes a
        dense array beyond `_DENSE_COV_MAX_N` (work-plan Sec.C2).
        """
        if self.dense_cov is not None:
            return self.dense_cov
        n = self.grid.n
        if self.precision_factor is not None:
            if n <= _DENSE_COV_MAX_N:
                return self.precision_factor.covariance_dense()
            solve = self.precision_factor.solve
            return LinearOperator((n, n), matvec=solve, matmat=solve, dtype=float)
        low_rank, diag_var = self.low_rank, self.diag_var
        if low_rank is not None:
            if n <= _DENSE_COV_MAX_N:
                return low_rank @ low_rank.T + np.diag(diag_var)

            def _low_rank_action(v: np.ndarray) -> np.ndarray:
                v = np.asarray(v, dtype=float)
                residual = diag_var[:, None] * v if v.ndim == 2 else diag_var * v
                return low_rank @ (low_rank.T @ v) + residual

            return LinearOperator((n, n), matvec=_low_rank_action, matmat=_low_rank_action, dtype=float)
        if n <= _DENSE_COV_MAX_N:
            return np.diag(diag_var)

        def _diag_action(v: np.ndarray) -> np.ndarray:
            v = np.asarray(v, dtype=float)
            return diag_var[:, None] * v if v.ndim == 2 else diag_var * v

        return LinearOperator((n, n), matvec=_diag_action, matmat=_diag_action, dtype=float)

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Draw ``n`` samples, mapped into physical units; shape ``(n, grid.n)``."""
        m = self.grid.n
        if self.dense_cov is not None:
            chol = np.linalg.cholesky(self.dense_cov + 1e-12 * np.eye(m))
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

    def samples(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """IC-1 `Posterior.samples`: draw ``n`` samples in physical units; shape ``(n, grid.n)``.

        A thin alias over :meth:`sample` (kept unchanged, so no existing caller of the singular
        name breaks).
        """
        return self.sample(n, rng)

    def posterior_predictive_draws(
        self,
        registry: Any,
        observation: Any,
        *,
        n: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Draw posterior-predictive model values for one registered observation."""
        draws = self.sample(n, rng)
        op = registry.get(observation.kind)
        return np.vstack([op.predict_observation(self.grid, draw, observation) for draw in draws])

    def credible_interval(self, level: float = 0.9, *, alpha: float | None = None) -> tuple[np.ndarray, np.ndarray]:
        """IC-1 `Posterior.credible_interval`: per-point central interval covering ``level`` mass,
        in physical units.

        ``alpha`` (the former public parameter, the miss-coverage ``1 - level``) is accepted as a
        deprecated keyword alias for one release: passing it overrides ``level`` and emits a
        ``DeprecationWarning``.
        """
        if alpha is not None:
            warnings.warn(
                "PosteriorField3D.credible_interval(alpha=...) is deprecated; pass level=... "
                "(level = 1 - alpha) instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            level = 1.0 - alpha
        if not 0.0 < level < 1.0:
            raise ValueError("level must be in (0, 1).")
        z = ndtri(0.5 + level / 2.0)
        std = self.marginal_std
        lo_u = self.mean - z * std
        hi_u = self.mean + z * std
        return self.grid.from_unconstrained(lo_u), self.grid.from_unconstrained(hi_u)

    def derived_quantity(
        self, fn: Callable[[np.ndarray], np.ndarray], n: int, rng: np.random.Generator
    ) -> _PosteriorDerivedQuantity:
        """IC-1 `Posterior.derived_quantity`: pushforward ``fn`` over ``n`` posterior draws
        (physical units) into a `DerivedQuantity` (samples + a central credible interval + the
        `prior_dominated` honesty flag; work-plan A2 sets that flag once its variance-reduction
        hook lands -- out of scope here, so it defaults to False).
        """
        draws = self.samples(n, rng)
        return _PosteriorDerivedQuantity(fn(draws), prior_dominated=False)

    def physical_mean(self, *, n: int = 4096, rng: np.random.Generator | None = None) -> np.ndarray:
        """Posterior mean in physical units, i.e. ``E[g(X)]`` where ``g = grid.from_unconstrained``.

        For an unbounded field ``g`` is the identity, so this agrees with ``grid.from_unconstrained(mean)``
        up to Monte-Carlo noise. For a bounded field ``g`` is a log/logit transform, and
        ``grid.from_unconstrained(mean)`` is ``g(median)``, not ``E[g(X)]`` -- Jensen's inequality makes the
        two diverge once the unconstrained-space variance is appreciable. This draws ``n`` physical-unit
        samples (:meth:`sample` already maps through ``grid.from_unconstrained``) and averages them, which
        is unbiased regardless of the transform.
        """
        if rng is None:
            rng = np.random.default_rng()
        draws = self.sample(n, rng)
        return draws.mean(axis=0)

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


@dataclass
class PosteriorFieldSamples3D:
    """An empirical posterior over a :class:`Field3D`, stored as unconstrained samples.

    This is the small-reference posterior artifact for non-Gaussian or nonlinear checks where a Gaussian
    covariance summary would hide posterior shape. It deliberately mirrors the core query surface of
    :class:`PosteriorField3D`: ``mean`` / ``map`` in unconstrained space, marginal variance/std,
    physical-unit ``sample`` draws, physical-unit credible intervals, and axis-aligned slices.

    Note (IC-1 conformance): the empirical draws are stored under the ``samples`` **attribute**
    (read directly by several existing consumers), which collides with the plural ``samples(n,
    rng)`` **method** IC-1's `Posterior` protocol expects -- the same collision :class:`PosteriorField3D`
    had with its old ``sample``/``samples`` naming, and that ``PosteriorEnsemble`` had with its
    ``samples`` attribute. Unlike those two (each with a handful of call sites), the stored
    ``samples`` array here is read directly across many modules; renaming it is a larger, separate
    change. This class therefore adds `credible_interval(level)`, `derived_quantity`, and `cov`, but
    does not (yet) satisfy `isinstance(x, Posterior)` -- it keeps `sample` (singular) as its draw
    method.
    """

    grid: Field3D
    samples: np.ndarray
    log_posterior: np.ndarray | None = None
    map: np.ndarray | None = None
    provenance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        n = self.grid.n
        samples = np.asarray(self.samples, dtype=float)
        if samples.ndim != 2 or samples.shape[1] != n:
            raise ValueError(f"samples must have shape (n_samples, {n}).")
        if samples.shape[0] == 0:
            raise ValueError("samples must contain at least one posterior draw.")
        if not np.all(np.isfinite(samples)):
            raise ValueError("samples must be finite.")
        self.samples = samples

        if self.log_posterior is not None:
            logp = np.asarray(self.log_posterior, dtype=float)
            if logp.shape != (samples.shape[0],):
                raise ValueError(f"log_posterior must have shape ({samples.shape[0]},).")
            self.log_posterior = logp

        if self.map is None:
            if self.log_posterior is None:
                self.map = np.mean(samples, axis=0)
            else:
                self.map = samples[int(np.argmax(self.log_posterior))].copy()
        else:
            self.map = np.asarray(self.map, dtype=float)
        if self.map.shape != (n,):
            raise ValueError(f"map must have shape ({n},).")
        self.provenance = dict(self.provenance)

    @property
    def mean(self) -> np.ndarray:
        """Empirical posterior mean in the field's unconstrained space."""
        return np.mean(self.samples, axis=0)

    @property
    def marginal_variance(self) -> np.ndarray:
        """Empirical per-point posterior variance in unconstrained space."""
        if self.samples.shape[0] == 1:
            return np.zeros(self.grid.n)
        return np.var(self.samples, axis=0, ddof=1)

    @property
    def marginal_std(self) -> np.ndarray:
        return np.sqrt(self.marginal_variance)

    @property
    def cov(self) -> np.ndarray | LinearOperator:
        """IC-1 `Posterior.cov`: empirical covariance of the stored (unconstrained-space) draws --
        a dense ``(grid.n, grid.n)`` array when small enough to materialize cheaply, otherwise a
        matrix-free `LinearOperator` computed from the centered sample matrix (never a materialized
        dense array beyond `_DENSE_COV_MAX_N`).
        """
        n = self.grid.n
        if self.samples.shape[0] == 1:
            return np.zeros((n, n))
        if n <= _DENSE_COV_MAX_N:
            return np.atleast_2d(np.cov(self.samples, rowvar=False))
        centered = self.samples - self.mean[None, :]
        denom = max(self.samples.shape[0] - 1, 1)

        def _action(v: np.ndarray) -> np.ndarray:
            v = np.asarray(v, dtype=float)
            return (centered.T @ (centered @ v)) / denom

        return LinearOperator((n, n), matvec=_action, matmat=_action, dtype=float)

    @property
    def physical_samples(self) -> np.ndarray:
        """Stored posterior samples mapped into physical units."""
        return self.grid.from_unconstrained(self.samples)

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Resample ``n`` empirical draws with replacement, mapped into physical units."""
        if int(n) <= 0:
            raise ValueError("n must be positive.")
        idx = rng.integers(0, self.samples.shape[0], size=int(n))
        return self.grid.from_unconstrained(self.samples[idx])

    def credible_interval(self, level: float = 0.9, *, alpha: float | None = None) -> tuple[np.ndarray, np.ndarray]:
        """IC-1 `Posterior.credible_interval`: per-point empirical central interval covering
        ``level`` mass, in physical units. ``alpha`` (``1 - level``) is a deprecated keyword alias
        for one release.
        """
        if alpha is not None:
            warnings.warn(
                "PosteriorFieldSamples3D.credible_interval(alpha=...) is deprecated; pass level=... "
                "(level = 1 - alpha) instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            level = 1.0 - alpha
        if not 0.0 < level < 1.0:
            raise ValueError("level must be in (0, 1).")
        alpha_internal = 1.0 - level
        physical = self.physical_samples
        lo, hi = np.quantile(physical, [alpha_internal / 2.0, 1.0 - alpha_internal / 2.0], axis=0)
        return lo, hi

    def derived_quantity(
        self, fn: Callable[[np.ndarray], np.ndarray], n: int, rng: np.random.Generator
    ) -> _PosteriorDerivedQuantity:
        """IC-1 `Posterior.derived_quantity`: pushforward ``fn`` over ``n`` resampled draws
        (physical units) into a `DerivedQuantity` (samples + a central credible interval +
        `prior_dominated`, which defaults to False -- work-plan A2's variance-reduction hook is out
        of scope here).
        """
        draws = self.sample(n, rng)
        return _PosteriorDerivedQuantity(fn(draws), prior_dominated=False)

    def posterior_predictive_draws(
        self,
        registry: Any,
        observation: Any,
        *,
        n: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Draw posterior-predictive model values for one registered observation."""
        draws = self.sample(n, rng)
        op = registry.get(observation.kind)
        return np.vstack([op.predict_observation(self.grid, draw, observation) for draw in draws])

    def slice(
        self, *, x: float | None = None, y: float | None = None, z: float | None = None, tol: float = 1e-9
    ) -> dict[str, Any]:
        """Select grid points matching fixed axis coordinate(s), within ``tol``."""
        fixed = [(axis, value) for axis, value in (("x", x), ("y", y), ("z", z)) if value is not None]
        if not fixed:
            raise ValueError("slice() requires at least one of x, y, z.")
        axis_index = {"x": 0, "y": 1, "z": 2}
        index = np.ones(self.grid.n, dtype=bool)
        for axis, value in fixed:
            index &= np.isclose(self.grid.coordinates[:, axis_index[axis]], value, atol=tol)
        physical = self.physical_samples
        mean_physical = np.mean(physical, axis=0)
        map_physical = self.grid.from_unconstrained(self.map)
        return {
            "index": index,
            "coordinates": self.grid.coordinates[index],
            "mean": mean_physical[index],
            "map": map_physical[index],
            "marginal_std": (
                np.std(physical[:, index], axis=0, ddof=1) if physical.shape[0] > 1 else np.zeros(index.sum())
            ),
        }
