"""Geochemistry and paleontology/biostratigraphy observation models (workstream G3).

Workstream G2 (:mod:`mixle_pde.observations`) gives every observation ONE Gaussian likelihood. Real
geoscience data breaks that assumption in two specific, common ways this module handles honestly:

* **Geochemistry -- detection limits (left-censoring).** An assay below the instrument's detection
  limit is not "zero" and not a point measurement: it only says *the true concentration is below the
  limit*. :func:`assay_log_likelihood` scores a detected element by its Gaussian density and a censored
  element by ``log P(predicted < detection_limit)`` -- so a model predicting a low value is REWARDED,
  not penalised, for a below-limit assay. :class:`MultiElementAssay` extends the same idea to lab panels
  with element covariance and batch offsets; detected elements use a covariance-aware Gaussian and
  censored elements use marginal left-tail probabilities. Compositional assays (parts summing to a
  constant) are handled through the additive-log-ratio transform (:func:`additive_log_ratio`), which maps
  a simplex-constrained composition to an unconstrained vector the Gaussian machinery can model.

* **Paleontology / biostratigraphy -- range zones and absence.** A fossil occurrence constrains the
  stratigraphic age to the taxon's known range; an ABSENCE is not evidence of a specific age but a soft
  one-sided bound (the taxon had not yet appeared / had already gone). :func:`biostrat_log_likelihood`
  scores a predicted age against a taxon's ``[first_appearance, last_appearance]`` range for an
  occurrence, and as a one-sided censored bound for an absence.

These are OBSERVATION LIKELIHOODS tied to provenance, units, and uncertainty -- not process simulators.
A full geochemical reaction path or a sedimentary-basin model is a separate validated kernel; nothing
here hides one inside a likelihood. Each observation dataclass carries ``units`` and ``provenance`` so a
posterior artifact can trace where every constraint came from.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.special import log_ndtr


def _gaussian_log_density(residual: np.ndarray, covariance: np.ndarray) -> float:
    residual = np.atleast_1d(np.asarray(residual, dtype=float))
    covariance = np.asarray(covariance, dtype=float)
    sign, logdet = np.linalg.slogdet(covariance)
    if sign <= 0.0:
        raise ValueError("covariance must be positive definite.")
    solved = np.linalg.solve(covariance, residual)
    return float(-0.5 * (residual @ solved + logdet + residual.size * np.log(2.0 * np.pi)))


def _coerce_sample_element_matrix(value: np.ndarray | float, *, shape: tuple[int, int], name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.shape == shape:
        return arr.copy()
    if arr.shape == (shape[1],):
        return np.broadcast_to(arr[None, :], shape).copy()
    if arr.shape == ():
        return np.full(shape, float(arr))
    raise ValueError(f"{name} must have shape {shape}, ({shape[1]},), or be scalar.")


@dataclass
class GeochemAssay:
    """A multi-point assay of ONE element's concentration, with per-point detection-limit censoring.

    ``censored[i]`` True means point ``i`` is below ``detection_limit[i]`` (left-censored): ``value[i]``
    then records the detection limit, not a measured concentration. ``noise_std`` is the (positive)
    measurement error in concentration units.
    """

    element: str
    location: np.ndarray
    value: np.ndarray
    noise_std: np.ndarray
    detection_limit: np.ndarray | None = None
    censored: np.ndarray | None = None
    units: str = ""
    provenance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        loc = np.atleast_2d(np.asarray(self.location, dtype=float))
        if loc.ndim != 2 or loc.shape[1] != 3:
            raise ValueError("location must be (n, 3).")
        self.location = loc
        n = loc.shape[0]
        self.value = np.atleast_1d(np.asarray(self.value, dtype=float))
        self.noise_std = np.atleast_1d(np.asarray(self.noise_std, dtype=float))
        if self.value.shape != (n,) or self.noise_std.shape != (n,):
            raise ValueError("value and noise_std must have shape (n,) matching location.")
        if np.any(self.noise_std <= 0.0):
            raise ValueError("noise_std must be strictly positive.")
        self.censored = np.zeros(n, dtype=bool) if self.censored is None else np.asarray(self.censored, dtype=bool)
        if self.censored.shape != (n,):
            raise ValueError("censored must have shape (n,).")
        if self.detection_limit is not None:
            self.detection_limit = np.atleast_1d(np.asarray(self.detection_limit, dtype=float))
            if self.detection_limit.shape != (n,):
                raise ValueError("detection_limit must have shape (n,).")
        elif np.any(self.censored):
            raise ValueError("detection_limit is required when any point is censored.")
        self.provenance = dict(self.provenance)

    @property
    def n(self) -> int:
        return self.value.shape[0]


def assay_log_likelihood(assay: GeochemAssay, predicted: np.ndarray) -> float:
    """``log p(assay | predicted concentrations)`` with left-censoring at the detection limit.

    Detected points score by Gaussian density; censored points score by ``log Phi((dl - pred)/sigma)``
    -- the probability the true (noisy) concentration fell below the detection limit, so predicting a
    low value is rewarded for a below-limit assay rather than penalised toward the limit value.
    """
    predicted = np.atleast_1d(np.asarray(predicted, dtype=float))
    if predicted.shape != assay.value.shape:
        raise ValueError(f"predicted must have shape {assay.value.shape}.")
    sigma = assay.noise_std
    total = 0.0
    detected = ~assay.censored
    if np.any(detected):
        resid = assay.value[detected] - predicted[detected]
        s = sigma[detected]
        total += float(np.sum(-0.5 * (resid**2 / s**2 + np.log(2.0 * np.pi * s**2))))
    if np.any(assay.censored):
        dl = assay.detection_limit[assay.censored]
        p = predicted[assay.censored]
        s = sigma[assay.censored]
        total += float(np.sum(log_ndtr((dl - p) / s)))
    return total


@dataclass
class MultiElementAssay:
    """A multi-point, multi-element geochemical assay with correlated analytical uncertainty.

    ``value[i, j]`` is the measured concentration for sample ``i`` and element ``j``. ``noise_cov`` can
    be a shared diagonal ``(k,)`` variance vector, a shared full ``(k, k)`` element covariance, or a
    per-sample ``(n, k, k)`` covariance stack. ``censored[i, j]`` marks below-detection measurements;
    those entries score by their marginal detection-limit probability while detected entries use the
    covariance submatrix for the observed elements. ``batch_offset`` is an additive lab/batch bias in
    assay units, so the expected reported value is ``predicted + batch_offset``.
    """

    elements: Sequence[str]
    location: np.ndarray
    value: np.ndarray
    noise_cov: np.ndarray
    detection_limit: np.ndarray | None = None
    censored: np.ndarray | None = None
    batch_offset: np.ndarray | float | None = None
    units: str = ""
    provenance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.elements = tuple(str(element) for element in self.elements)
        if not self.elements:
            raise ValueError("elements must not be empty.")
        if len(set(self.elements)) != len(self.elements):
            raise ValueError("elements must be unique.")
        self.location = np.atleast_2d(np.asarray(self.location, dtype=float))
        if self.location.ndim != 2 or self.location.shape[1] != 3:
            raise ValueError("location must be (n, 3).")
        n = self.location.shape[0]
        k = len(self.elements)
        self.value = np.asarray(self.value, dtype=float)
        if self.value.shape != (n, k):
            raise ValueError("value must have shape (n, k) matching location and elements.")

        cov = np.asarray(self.noise_cov, dtype=float)
        if cov.shape == (k,):
            if np.any(cov <= 0.0):
                raise ValueError("diagonal noise_cov entries must be positive variances.")
            cov_stack = np.broadcast_to(np.diag(cov)[None, :, :], (n, k, k)).copy()
        elif cov.shape == (k, k):
            cov_stack = np.broadcast_to(cov[None, :, :], (n, k, k)).copy()
        elif cov.shape == (n, k, k):
            cov_stack = cov.copy()
        else:
            raise ValueError("noise_cov must have shape (k,), (k, k), or (n, k, k).")
        for sample_cov in cov_stack:
            if not np.allclose(sample_cov, sample_cov.T):
                raise ValueError("noise_cov must be symmetric.")
            sign, _ = np.linalg.slogdet(sample_cov)
            if sign <= 0.0:
                raise ValueError("noise_cov must be positive definite.")
        self.noise_cov = cov_stack

        self.censored = (
            np.zeros((n, k), dtype=bool) if self.censored is None else np.asarray(self.censored, dtype=bool)
        )
        if self.censored.shape != (n, k):
            raise ValueError("censored must have shape (n, k).")
        if self.detection_limit is not None:
            self.detection_limit = _coerce_sample_element_matrix(
                self.detection_limit, shape=(n, k), name="detection_limit"
            )
        elif np.any(self.censored):
            raise ValueError("detection_limit is required when any element is censored.")
        if self.batch_offset is None:
            self.batch_offset = np.zeros((n, k), dtype=float)
        else:
            self.batch_offset = _coerce_sample_element_matrix(self.batch_offset, shape=(n, k), name="batch_offset")
        self.provenance = dict(self.provenance)

    @property
    def n(self) -> int:
        return self.value.shape[0]

    @property
    def k(self) -> int:
        return self.value.shape[1]


def multi_element_assay_log_likelihood(assay: MultiElementAssay, predicted: np.ndarray) -> float:
    """``log p(assay | predicted element concentrations)`` for correlated multi-element assays.

    Detected elements at each sample use the appropriate covariance submatrix. Censored elements are
    below-detection observations and score by the marginal left-tail probability from the corresponding
    covariance diagonal. This keeps common lab covariance and batch effects in the likelihood while
    staying explicit about the approximation used for mixed detected/censored rows.
    """
    predicted = np.asarray(predicted, dtype=float)
    if predicted.shape != assay.value.shape:
        raise ValueError(f"predicted must have shape {assay.value.shape}.")
    expected = predicted + assay.batch_offset
    total = 0.0
    for i in range(assay.n):
        detected = ~assay.censored[i]
        if np.any(detected):
            cov = assay.noise_cov[i][np.ix_(detected, detected)]
            residual = assay.value[i, detected] - expected[i, detected]
            total += _gaussian_log_density(residual, cov)
        if np.any(assay.censored[i]):
            censored = assay.censored[i]
            std = np.sqrt(np.diag(assay.noise_cov[i])[censored])
            dl = assay.detection_limit[i, censored]
            p = expected[i, censored]
            total += float(np.sum(log_ndtr((dl - p) / std)))
    return total


def additive_log_ratio(composition: np.ndarray, *, denom_index: int = -1, eps: float = 1e-12) -> np.ndarray:
    """Additive log-ratio (ALR) of a compositional vector/array: map a simplex point (parts that sum to
    a constant) to an unconstrained ``(d-1)``-vector ``log(x_i / x_denom)`` the Gaussian machinery can
    model. Operates on the last axis."""
    comp = np.asarray(composition, dtype=float)
    comp = np.clip(comp, eps, None)
    denom = comp[..., denom_index][..., None]
    keep = [i for i in range(comp.shape[-1]) if i % comp.shape[-1] != denom_index % comp.shape[-1]]
    return np.log(comp[..., keep] / denom)


def inverse_additive_log_ratio(alr: np.ndarray, *, total: float = 1.0) -> np.ndarray:
    """Inverse ALR: unconstrained ``(d-1)``-vector back to a ``d``-part composition summing to ``total``
    (the denominator part restored as the reference). Operates on the last axis."""
    alr = np.asarray(alr, dtype=float)
    exp = np.concatenate([np.exp(alr), np.ones(alr.shape[:-1] + (1,))], axis=-1)
    return total * exp / np.sum(exp, axis=-1, keepdims=True)


@dataclass
class BiostratConstraint:
    """A biostratigraphic age constraint at one location from a taxon occurrence or absence.

    An occurrence (``present=True``) constrains the age to the taxon's range zone
    ``[first_appearance, last_appearance]`` (ages increasing into the past). An absence
    (``present=False``) is a one-sided soft bound: the sample age is on the young side of
    ``absence_bound`` (the taxon had not appeared yet) -- weak evidence, scored as censoring.
    ``tolerance`` softens the range edges (measurement/correlation slack).
    """

    location: np.ndarray
    taxon: str
    present: bool
    first_appearance: float | None = None
    last_appearance: float | None = None
    absence_bound: float | None = None
    tolerance: float = 0.5
    units: str = "Ma"
    provenance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.location = np.atleast_2d(np.asarray(self.location, dtype=float))
        if self.tolerance <= 0.0:
            raise ValueError("tolerance must be positive.")
        if self.present:
            if self.first_appearance is None or self.last_appearance is None:
                raise ValueError("an occurrence needs first_appearance and last_appearance.")
            if self.first_appearance < self.last_appearance:
                raise ValueError("first_appearance (older) must be >= last_appearance (younger).")
        elif self.absence_bound is None:
            raise ValueError("an absence needs absence_bound.")
        self.provenance = dict(self.provenance)


def biostrat_log_likelihood(constraint: BiostratConstraint, predicted_age: float) -> float:
    """``log p(constraint | predicted age)`` for a biostrat occurrence or absence.

    Occurrence: a plateau of log-density 0 inside the range zone, decaying as a Gaussian with scale
    ``tolerance`` outside it (so an age just outside the zone is penalised, not forbidden). Absence: a
    one-sided censored bound ``log Phi((absence_bound - age)/tolerance)`` -- ages younger than the bound
    are consistent, older ages increasingly unlikely.
    """
    age = float(predicted_age)
    tol = constraint.tolerance
    if constraint.present:
        if constraint.last_appearance <= age <= constraint.first_appearance:
            return 0.0
        gap = (
            age - constraint.first_appearance if age > constraint.first_appearance else constraint.last_appearance - age
        )
        return float(-0.5 * (gap / tol) ** 2)
    return float(log_ndtr((constraint.absence_bound - age) / tol))


@dataclass
class GeochronologyAge:
    """An isotopic/geochronology age measurement at one location.

    ``analytical_std`` is the laboratory measurement uncertainty. ``systematic_std`` captures shared
    calibration/decay-constant uncertainty when available; the likelihood treats them as independent and
    combines them in quadrature. This is a measurement likelihood over an ``age_ma`` field, not a thermal
    history or isotope-system closure simulator.
    """

    location: np.ndarray
    age: float
    analytical_std: float
    systematic_std: float = 0.0
    method: str = ""
    units: str = "Ma"
    provenance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.location = np.atleast_2d(np.asarray(self.location, dtype=float))
        if self.location.ndim != 2 or self.location.shape[1] != 3:
            raise ValueError("location must be (n, 3).")
        if self.location.shape[0] != 1:
            raise ValueError("GeochronologyAge records one dated sample location.")
        if self.analytical_std <= 0.0:
            raise ValueError("analytical_std must be positive.")
        if self.systematic_std < 0.0:
            raise ValueError("systematic_std must be non-negative.")
        self.age = float(self.age)
        self.analytical_std = float(self.analytical_std)
        self.systematic_std = float(self.systematic_std)
        self.provenance = dict(self.provenance)

    @property
    def total_std(self) -> float:
        return float(np.hypot(self.analytical_std, self.systematic_std))


def geochronology_log_likelihood(observation: GeochronologyAge, predicted_age: float) -> float:
    """Gaussian age likelihood for an isotopic/geochronology measurement."""
    sigma = observation.total_std
    residual = observation.age - float(predicted_age)
    return float(-0.5 * (residual**2 / sigma**2 + np.log(2.0 * np.pi * sigma**2)))


@dataclass
class StratigraphicCorrelation:
    """A relative-age / horizon-correlation constraint between two locations.

    ``age_difference`` is ``age_a - age_b`` in ``units``. A value of zero says the two positions share a
    correlated horizon. Positive values say location A is older by that amount.
    """

    location_a: np.ndarray
    location_b: np.ndarray
    age_difference: float = 0.0
    std: float = 1.0
    units: str = "Ma"
    provenance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.location_a = np.atleast_2d(np.asarray(self.location_a, dtype=float))
        self.location_b = np.atleast_2d(np.asarray(self.location_b, dtype=float))
        if self.location_a.shape != (1, 3) or self.location_b.shape != (1, 3):
            raise ValueError("location_a and location_b must each be one (1, 3) coordinate.")
        if self.std <= 0.0:
            raise ValueError("std must be positive.")
        self.age_difference = float(self.age_difference)
        self.std = float(self.std)
        self.provenance = dict(self.provenance)


def stratigraphic_correlation_log_likelihood(
    constraint: StratigraphicCorrelation, predicted_age_a: float, predicted_age_b: float
) -> float:
    """Gaussian likelihood for a relative-age / horizon-correlation constraint."""
    predicted = float(predicted_age_a) - float(predicted_age_b)
    residual = constraint.age_difference - predicted
    sigma = constraint.std
    return float(-0.5 * (residual**2 / sigma**2 + np.log(2.0 * np.pi * sigma**2)))


@dataclass
class FaciesIntervalConstraint:
    """A facies, environment, palynology, or microfossil interval constraint over a numeric proxy field.

    ``present=True`` gives a plateau inside ``[lower, upper]`` with Gaussian decay outside. ``present=False``
    is an absence constraint: values outside the interval are consistent, while values inside the excluded
    interval are penalized by distance to the nearest interval boundary.
    """

    location: np.ndarray
    label: str
    property_name: str
    lower: float
    upper: float
    tolerance: float = 0.5
    present: bool = True
    units: str = ""
    provenance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.location = np.atleast_2d(np.asarray(self.location, dtype=float))
        if self.location.shape != (1, 3):
            raise ValueError("location must be one (1, 3) coordinate.")
        if self.lower >= self.upper:
            raise ValueError("lower must be < upper.")
        if self.tolerance <= 0.0:
            raise ValueError("tolerance must be positive.")
        self.lower = float(self.lower)
        self.upper = float(self.upper)
        self.tolerance = float(self.tolerance)
        self.provenance = dict(self.provenance)


def facies_interval_log_likelihood(constraint: FaciesIntervalConstraint, predicted_value: float) -> float:
    """Soft interval likelihood for a facies/environment indicator or proxy field."""
    value = float(predicted_value)
    tol = constraint.tolerance
    inside = constraint.lower <= value <= constraint.upper
    if constraint.present:
        if inside:
            return 0.0
        gap = value - constraint.upper if value > constraint.upper else constraint.lower - value
        return float(-0.5 * (gap / tol) ** 2)
    if not inside:
        return 0.0
    gap_to_boundary = min(value - constraint.lower, constraint.upper - value)
    return float(-0.5 * (gap_to_boundary / tol) ** 2)


def assay_posterior_predictive(assay: GeochemAssay, grid: Any, posterior_mean: np.ndarray) -> np.ndarray:
    """Posterior-predictive concentrations for an assay's locations from a fitted element field (nearest
    grid cell) -- the input to a posterior-predictive check against held-out assays."""
    coords = np.asarray(grid.coordinates, dtype=float)
    diffs = coords[None, :, :] - assay.location[:, None, :]
    idx = np.argmin(np.sum(diffs**2, axis=2), axis=1)
    return np.asarray(posterior_mean, dtype=float)[idx]


def multi_element_assay_posterior_predictive(
    assay: MultiElementAssay, grid: Any, posterior_mean: np.ndarray | Mapping[str, np.ndarray]
) -> np.ndarray:
    """Nearest-cell posterior-predictive concentrations for every element in a multi-element assay.

    ``posterior_mean`` may be a dense ``(grid.n, k)`` array in ``assay.elements`` order or a mapping from
    element name to one posterior mean vector per grid cell.
    """
    coords = np.asarray(grid.coordinates, dtype=float)
    diffs = coords[None, :, :] - assay.location[:, None, :]
    idx = np.argmin(np.sum(diffs**2, axis=2), axis=1)
    if isinstance(posterior_mean, Mapping):
        columns = []
        for element in assay.elements:
            if element not in posterior_mean:
                raise ValueError(f"posterior_mean is missing element {element!r}.")
            values = np.asarray(posterior_mean[element], dtype=float)
            if values.shape[0] != coords.shape[0]:
                raise ValueError("posterior_mean element vectors must match the grid coordinate count.")
            columns.append(values[idx])
        return np.stack(columns, axis=1)
    values = np.asarray(posterior_mean, dtype=float)
    if values.shape != (coords.shape[0], assay.k):
        raise ValueError(f"posterior_mean must have shape {(coords.shape[0], assay.k)}.")
    return values[idx, :]
