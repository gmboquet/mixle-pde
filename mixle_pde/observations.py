"""Typed observations and a common likelihood interface over 3D/4D latent fields (workstream G2).

:mod:`mixle_pde.latent` (workstream G1) defines the latent object; this module defines what an
OBSERVATION of it looks like and how it is scored -- the second concrete step of workstream G, still
"no inversion yet": geometry, noise model, provenance, and a forward-operator registry, so a later
inversion card fits a :class:`~mixle_pde.latent.PosteriorField3D` against a mixed batch of
observations without a per-kind branch anywhere in the fitting code.

An :class:`Observation` fixes ONE common shape (location, value, noise covariance, time, units,
provenance) that every observation kind -- a gravity station, a magnetics reading, a borehole sample,
a geochemical assay -- conforms to. :func:`gaussian_log_likelihood` is the ONE likelihood function
every kind shares: only the :class:`ForwardOperator` (the physics mapping a field to a predicted
observation) differs by kind, resolved through :class:`ForwardOperatorRegistry` by
``observation.kind``. This card wires two potential-field kinds already present in
:mod:`mixle_pde.geophysics` (gravity, magnetics) plus a direct point-sample operator for
borehole/sensor observations; geochemistry and paleontology/biostratigraphy likelihoods are a
separate, later card (workstream G step 3), not assumed here.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from mixle_pde.latent import Field3D


@dataclass
class Observation:
    """One typed observation of a latent field: geometry, value, noise model, time, provenance.

    ``noise_cov`` is either ``(n,)`` (diagonal variances -- independent noise) or ``(n, n)`` (a full
    covariance, e.g. correlated instrument drift). ``time`` is ``None`` for a static (time-invariant)
    observation, or the observation time for a 4D time-lapse field.
    """

    kind: str
    location: np.ndarray
    value: np.ndarray
    noise_cov: np.ndarray
    time: float | None = None
    units: str = ""
    provenance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        loc = np.atleast_2d(np.asarray(self.location, dtype=float))
        if loc.ndim != 2 or loc.shape[1] != 3:
            raise ValueError("location must be an (n, 3) array of (x, y, z) points.")
        self.location = loc

        value = np.atleast_1d(np.asarray(self.value, dtype=float))
        if value.shape != (loc.shape[0],):
            raise ValueError(f"value must have shape ({loc.shape[0]},) matching location, got {value.shape}.")
        self.value = value

        n = value.shape[0]
        noise_cov = np.asarray(self.noise_cov, dtype=float)
        if noise_cov.shape == (n,):
            if np.any(noise_cov <= 0.0):
                raise ValueError("diagonal noise_cov entries must be strictly positive.")
        elif noise_cov.shape == (n, n):
            if not np.allclose(noise_cov, noise_cov.T):
                raise ValueError("full noise_cov must be symmetric.")
        else:
            raise ValueError(
                f"noise_cov must have shape ({n},) (diagonal) or ({n}, {n}) (full), got {noise_cov.shape}."
            )
        self.noise_cov = noise_cov
        self.provenance = dict(self.provenance)

    @property
    def n(self) -> int:
        return self.value.shape[0]

    @property
    def is_diagonal(self) -> bool:
        return self.noise_cov.ndim == 1


def gaussian_log_likelihood(observation: Observation, predicted: np.ndarray) -> float:
    """``log p(observation.value | predicted)`` under the observation's OWN noise model.

    The one common likelihood every observation kind shares regardless of the physics that produced
    ``predicted`` -- exact for the declared noise model (diagonal or full covariance), never assuming
    independence when a full covariance was supplied.
    """
    predicted = np.atleast_1d(np.asarray(predicted, dtype=float))
    if predicted.shape != observation.value.shape:
        raise ValueError(f"predicted must have shape {observation.value.shape}, got {predicted.shape}.")
    residual = observation.value - predicted
    n = observation.n
    if observation.is_diagonal:
        var = observation.noise_cov
        return float(-0.5 * np.sum(residual**2 / var + np.log(2.0 * np.pi * var)))
    prec = np.linalg.inv(observation.noise_cov)
    _, logdet = np.linalg.slogdet(observation.noise_cov)
    return float(-0.5 * (residual @ prec @ residual + logdet + n * np.log(2.0 * np.pi)))


@dataclass
class ForwardOperator:
    """Binds an observation ``kind`` to the physics mapping a latent field to a predicted observation.

    ``predict(grid, field_values, obs_locations) -> (n,)`` is required. ``jacobian(grid,
    obs_locations) -> (n, grid.n)``, if supplied, declares the operator as having an exact adjoint
    (linear-in-field-values physics); ``differentiable`` separately records whether ``predict`` is
    torch-differentiable (a finite-difference fallback is always available by definition of
    ``predict`` being callable, so this contract never blocks on missing gradients -- it only makes
    the actual capability explicit).
    """

    kind: str
    predict: Callable[[Field3D, np.ndarray, np.ndarray], np.ndarray]
    jacobian: Callable[[Field3D, np.ndarray], np.ndarray] | None = None
    differentiable: bool = False

    def has_adjoint(self) -> bool:
        return self.jacobian is not None

    def predict_observation(self, grid: Field3D, field_values: np.ndarray, observation: Observation) -> np.ndarray:
        if observation.kind != self.kind:
            raise ValueError(f"operator kind {self.kind!r} does not match observation kind {observation.kind!r}.")
        return np.atleast_1d(np.asarray(self.predict(grid, field_values, observation.location), dtype=float))


class ForwardOperatorRegistry:
    """A registry of :class:`ForwardOperator` by observation kind.

    Resolves an :class:`Observation`'s ``kind`` to the physics that predicts it, so likelihood and
    (later) inversion code never needs a per-kind branch -- new observation kinds register here, they
    do not require touching the fitting machinery.
    """

    def __init__(self) -> None:
        self._ops: dict[str, ForwardOperator] = {}

    def register(self, op: ForwardOperator) -> None:
        self._ops[op.kind] = op

    def get(self, kind: str) -> ForwardOperator:
        if kind not in self._ops:
            raise KeyError(
                f"no forward operator registered for observation kind {kind!r}; registered: {sorted(self._ops)}"
            )
        return self._ops[kind]

    def __contains__(self, kind: str) -> bool:
        return kind in self._ops

    def log_likelihood(self, grid: Field3D, field_values: np.ndarray, observation: Observation) -> float:
        """Resolve ``observation.kind`` to its operator, predict, and score -- the one-call path."""
        op = self.get(observation.kind)
        predicted = op.predict_observation(grid, field_values, observation)
        return gaussian_log_likelihood(observation, predicted)

    def total_log_likelihood(self, grid: Field3D, field_values: np.ndarray, observations: list[Observation]) -> float:
        """Sum of :meth:`log_likelihood` over a (possibly multi-kind) batch of observations -- the
        one-call path a later inversion optimizes."""
        return float(sum(self.log_likelihood(grid, field_values, obs) for obs in observations))


def gravity_forward_operator(cells: np.ndarray, volumes: np.ndarray) -> ForwardOperator:
    """Wire :func:`mixle_pde.geophysics.gravity_point_sensitivity` as a G2 gravity operator:
    predicted vertical gravity anomaly (mGal) from a density-contrast field."""
    from mixle_pde.geophysics import gravity_point_sensitivity

    cells = np.asarray(cells, dtype=float)

    def predict(grid: Field3D, field_values: np.ndarray, obs_locations: np.ndarray) -> np.ndarray:
        G = gravity_point_sensitivity(obs_locations, cells, volumes)
        return G @ np.asarray(field_values, dtype=float)

    def jacobian(grid: Field3D, obs_locations: np.ndarray) -> np.ndarray:
        return gravity_point_sensitivity(obs_locations, cells, volumes)

    return ForwardOperator("gravity", predict, jacobian=jacobian, differentiable=False)


def magnetics_forward_operator(
    cells: np.ndarray, volumes: np.ndarray, *, inclination: float, declination: float, field_nt: float = 50000.0
) -> ForwardOperator:
    """Wire :func:`mixle_pde.geophysics.magnetic_dipole_sensitivity` as a G2 magnetics operator:
    predicted total-field anomaly (nT) from a susceptibility field."""
    from mixle_pde.geophysics import magnetic_dipole_sensitivity

    cells = np.asarray(cells, dtype=float)

    def _sensitivity(obs_locations: np.ndarray) -> np.ndarray:
        return magnetic_dipole_sensitivity(
            obs_locations, cells, volumes, inclination=inclination, declination=declination, field_nt=field_nt
        )

    def predict(grid: Field3D, field_values: np.ndarray, obs_locations: np.ndarray) -> np.ndarray:
        return _sensitivity(obs_locations) @ np.asarray(field_values, dtype=float)

    def jacobian(grid: Field3D, obs_locations: np.ndarray) -> np.ndarray:
        return _sensitivity(obs_locations)

    return ForwardOperator("magnetics", predict, jacobian=jacobian, differentiable=False)


def borehole_forward_operator() -> ForwardOperator:
    """A direct point-sample operator: predicted value is the field value at the NEAREST grid cell to
    each observation location -- the simplest borehole/sensor forward map, exact when observation
    locations coincide with grid points. Linear (a 0/1 selection matrix), so it declares a Jacobian.
    """

    def _nearest_index(grid: Field3D, obs_locations: np.ndarray) -> np.ndarray:
        obs_locations = np.atleast_2d(np.asarray(obs_locations, dtype=float))
        diffs = grid.coordinates[None, :, :] - obs_locations[:, None, :]
        return np.argmin(np.sum(diffs**2, axis=2), axis=1)

    def predict(grid: Field3D, field_values: np.ndarray, obs_locations: np.ndarray) -> np.ndarray:
        idx = _nearest_index(grid, obs_locations)
        return np.asarray(field_values, dtype=float)[idx]

    def jacobian(grid: Field3D, obs_locations: np.ndarray) -> np.ndarray:
        idx = _nearest_index(grid, obs_locations)
        J = np.zeros((len(idx), grid.n))
        J[np.arange(len(idx)), idx] = 1.0
        return J

    return ForwardOperator("borehole", predict, jacobian=jacobian, differentiable=True)
