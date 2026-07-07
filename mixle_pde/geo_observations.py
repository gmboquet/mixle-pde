"""Geochemistry and paleontology/biostratigraphy observation models (workstream G3).

Workstream G2 (:mod:`mixle_pde.observations`) gives every observation ONE Gaussian likelihood. Real
geoscience data breaks that assumption in two specific, common ways this module handles honestly:

* **Geochemistry -- detection limits (left-censoring).** An assay below the instrument's detection
  limit is not "zero" and not a point measurement: it only says *the true concentration is below the
  limit*. :func:`assay_log_likelihood` scores a detected element by its Gaussian density and a censored
  element by ``log P(predicted < detection_limit)`` -- so a model predicting a low value is REWARDED,
  not penalised, for a below-limit assay. Compositional assays (parts summing to a constant) are
  handled through the additive-log-ratio transform (:func:`additive_log_ratio`), which maps a
  simplex-constrained composition to an unconstrained vector the Gaussian machinery can model.

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

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.special import log_ndtr


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


def assay_posterior_predictive(assay: GeochemAssay, grid: Any, posterior_mean: np.ndarray) -> np.ndarray:
    """Posterior-predictive concentrations for an assay's locations from a fitted element field (nearest
    grid cell) -- the input to a posterior-predictive check against held-out assays."""
    coords = np.asarray(grid.coordinates, dtype=float)
    diffs = coords[None, :, :] - assay.location[:, None, :]
    idx = np.argmin(np.sum(diffs**2, axis=2), axis=1)
    return np.asarray(posterior_mean, dtype=float)[idx]
