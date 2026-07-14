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
``observation.kind``. This card wires potential fields, direct point samples, nonlinear DC/ERT, and a
layered-MT sounding operator into the same registry path. Geochemistry and
paleontology/biostratigraphy likelihoods live in :mod:`mixle_pde.geo_observations` because they are
measurement likelihoods rather than field-to-data PDE/geophysics operators.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from mixle_pde.latent import Field3D
from mixle_pde.model_error import inflated_noise_cov


@dataclass
class Observation:
    """One typed observation of a latent field: geometry, value, noise model, time, provenance.

    ``noise_cov`` is either ``(n,)`` (diagonal variances -- independent noise) or ``(n, n)`` (a full
    covariance, e.g. correlated instrument drift). ``time`` is ``None`` for a static (time-invariant)
    observation, or the observation time for a 4D time-lapse field. ``crs`` is an EPSG string (e.g.
    ``"EPSG:32613"``) naming the coordinate reference system ``location`` is expressed in, or ``None``
    when ``location`` is already mesh-local; see :mod:`mixle_pde.geospatial.crs` to reproject it.
    ``modality`` is the coarse sensor family (``"gravity"``, ``"seismic"``, ``"assay"``, ...) used for
    routing/UI, distinct from the fine-grained ``kind`` a :class:`ForwardOperator` dispatches on.
    """

    kind: str
    location: np.ndarray
    value: np.ndarray
    noise_cov: np.ndarray
    time: float | None = None
    units: str = ""
    provenance: dict[str, Any] = field(default_factory=dict)
    crs: str | None = None
    modality: str = ""

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


@dataclass
class SurveyGeometry:
    """Maps real acquisition XYZ (electrodes/shots/receivers/flight lines) onto discretisation indices.

    ``points`` is the ``(m, 3)`` real-world geometry in ``crs``; ``node_index``/``edge_index`` are the
    resolved mesh handles the forward operators consume, so an operator is built from XYZ with no
    hand-authored indices (workstream B7) -- replacing the manual quadrupole/edge tables the DC (around
    :func:`dc_resistivity_forward_operator`) and CSEM (around :func:`csem_3d_forward_operator`) operators
    used to require by hand. The node/edge-numbering rules themselves live in
    :mod:`mixle_pde.geometry_to_mesh` (`nearest_node_indices`, `electrodes_to_schedule`,
    `yee_edge_index`); this dataclass is the IC-4 handle that carries a survey's real geometry
    through to the point it is resolved onto a specific grid.
    """

    points: np.ndarray
    crs: str | None = None
    node_index: np.ndarray | None = None
    edge_index: np.ndarray | None = None

    def __post_init__(self) -> None:
        pts = np.atleast_2d(np.asarray(self.points, dtype=float))
        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError("points must be an (m, 3) array of (x, y, z) points.")
        self.points = pts
        if self.node_index is not None:
            node_index = np.atleast_1d(np.asarray(self.node_index))
            if node_index.shape != (pts.shape[0],):
                raise ValueError(f"node_index must have shape ({pts.shape[0]},), got {node_index.shape}.")
            self.node_index = node_index.astype(int)
        if self.edge_index is not None:
            edge_index = np.atleast_1d(np.asarray(self.edge_index))
            if edge_index.shape != (pts.shape[0],):
                raise ValueError(f"edge_index must have shape ({pts.shape[0]},), got {edge_index.shape}.")
            self.edge_index = edge_index.astype(int)

    def resolve(self, grid: Any) -> SurveyGeometry:
        """Return a copy with ``node_index`` filled by snapping ``points`` onto ``grid`` (workstream B7).

        ``grid`` is a :class:`~mixle_pde.latent.Field3D` (or anything exposing an ``(n, 3)``
        ``coordinates`` array): each point is snapped onto its nearest grid node via
        :func:`mixle_pde.geometry_to_mesh.nearest_node_indices` (a KD-tree query, never a hand-built
        table). ``edge_index`` is left as-is -- resolving a Yee-edge id additionally needs a per-point
        dipole axis that a bare ``grid`` does not carry; call
        :func:`mixle_pde.geometry_to_mesh.yee_edge_index` directly for that (one point + axis at a
        time), then assign the result onto a copy of this geometry.
        """
        from mixle_pde.geometry_to_mesh import nearest_node_indices

        node_index = nearest_node_indices(self.points, grid)
        return SurveyGeometry(points=self.points, crs=self.crs, node_index=node_index, edge_index=self.edge_index)


def gaussian_log_likelihood(
    observation: Observation, predicted: np.ndarray, *, model_error_cov: np.ndarray | None = None
) -> float:
    """``log p(observation.value | predicted)`` under the observation's noise model.

    The one common likelihood every observation kind shares regardless of the physics that produced
    ``predicted`` -- exact for the declared noise model (diagonal or full covariance), never assuming
    independence when a full covariance was supplied.

    ``model_error_cov``, when given, is a theory-error (model-discrepancy) covariance -- diagonal
    ``(n,)`` or full ``(n, n)`` -- that is added to ``observation.noise_cov`` (see
    :func:`mixle_pde.model_error.inflated_noise_cov`) before scoring, so a mis-specified forward
    operator no longer buys an overconfident posterior. Default ``None`` reproduces today's result
    exactly (score against the instrument noise alone).
    """
    predicted = np.atleast_1d(np.asarray(predicted, dtype=float))
    if predicted.shape != observation.value.shape:
        raise ValueError(f"predicted must have shape {observation.value.shape}, got {predicted.shape}.")
    residual = observation.value - predicted
    n = observation.n
    if model_error_cov is None:
        total_cov = observation.noise_cov
    else:
        total_cov = inflated_noise_cov(observation.noise_cov, np.asarray(model_error_cov, dtype=float))
    if total_cov.ndim == 1:
        var = total_cov
        return float(-0.5 * np.sum(residual**2 / var + np.log(2.0 * np.pi * var)))
    prec = np.linalg.inv(total_cov)
    _, logdet = np.linalg.slogdet(total_cov)
    return float(-0.5 * (residual @ prec @ residual + logdet + n * np.log(2.0 * np.pi)))


@dataclass
class ForwardOperator:
    """Binds an observation ``kind`` to the physics mapping a latent field to a predicted observation.

    ``predict(grid, field_values, obs_locations) -> (n,)`` is required. ``jacobian(grid,
    obs_locations) -> (n, grid.n)``, if supplied, declares a linear-in-field operator with a fixed
    adjoint. ``jacobian_at_values(grid, field_values, obs_locations) -> (n, grid.n)`` declares a
    nonlinear operator that can still expose a local linearization for Gauss-Newton. Linear Gaussian
    inference accepts only the fixed-Jacobian form; nonlinear MAP/Laplace inference can use either.

    ``jacobian_kind='finite_difference'`` (with ``finite_difference_step`` recorded) means the local
    Jacobian above is central-difference: O(2n) forward solves per evaluation, one per model parameter,
    versus a single solve for a true adjoint. Every nonlinear DC/EM operator in this module (DC
    resistivity, layered MT, AEM, 2-D/3-D MT, CSEM) currently uses this reference path; adjoint
    sensitivities are C1.
    """

    kind: str
    predict: Callable[[Field3D, np.ndarray, np.ndarray], np.ndarray]
    jacobian: Callable[[Field3D, np.ndarray], np.ndarray] | None = None
    jacobian_at_values: Callable[[Field3D, np.ndarray, np.ndarray], np.ndarray] | None = None
    differentiable: bool = False
    jacobian_kind: str | None = None
    has_true_adjoint: bool = False
    finite_difference_step: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.jacobian is not None and self.jacobian_at_values is not None:
            raise ValueError("provide either jacobian or jacobian_at_values, not both.")
        if self.jacobian_kind is None:
            if self.jacobian is not None:
                self.jacobian_kind = "fixed"
            elif self.jacobian_at_values is not None:
                self.jacobian_kind = "state_dependent"
            else:
                self.jacobian_kind = "none"
        allowed = {"none", "fixed", "state_dependent", "finite_difference", "autodiff", "adjoint"}
        if self.jacobian_kind not in allowed:
            raise ValueError(f"jacobian_kind must be one of {sorted(allowed)}.")
        if self.jacobian_kind == "none" and self.has_adjoint():
            raise ValueError("jacobian_kind='none' is inconsistent with a supplied Jacobian.")
        if self.finite_difference_step is not None and self.finite_difference_step <= 0.0:
            raise ValueError("finite_difference_step must be positive when supplied.")
        if self.jacobian_kind == "finite_difference" and self.finite_difference_step is None:
            raise ValueError("finite_difference Jacobians must record finite_difference_step.")
        self.metadata = dict(self.metadata)

    def has_adjoint(self) -> bool:
        return self.jacobian is not None or self.jacobian_at_values is not None

    @property
    def is_linear(self) -> bool:
        return self.jacobian is not None and self.jacobian_at_values is None

    def capability_report(self) -> dict[str, Any]:
        """Machine-readable declaration of derivative and fallback support."""
        return {
            "kind": self.kind,
            "is_linear": self.is_linear,
            "has_fixed_jacobian": self.jacobian is not None,
            "has_local_jacobian": self.has_adjoint(),
            "jacobian_kind": self.jacobian_kind,
            "has_true_adjoint": bool(self.has_true_adjoint),
            "differentiable": bool(self.differentiable),
            "finite_difference_step": self.finite_difference_step,
            "metadata": dict(self.metadata),
        }

    def predict_observation(self, grid: Field3D, field_values: np.ndarray, observation: Observation) -> np.ndarray:
        if observation.kind != self.kind:
            raise ValueError(f"operator kind {self.kind!r} does not match observation kind {observation.kind!r}.")
        return np.atleast_1d(np.asarray(self.predict(grid, field_values, observation.location), dtype=float))

    def local_jacobian(self, grid: Field3D, field_values: np.ndarray | None, observation: Observation) -> np.ndarray:
        """Return the fixed or state-dependent Jacobian for ``observation``.

        ``field_values`` is required only for nonlinear operators. The return shape is always
        ``(observation.n, grid.n)``.
        """
        if observation.kind != self.kind:
            raise ValueError(f"operator kind {self.kind!r} does not match observation kind {observation.kind!r}.")
        if self.jacobian_at_values is not None:
            if field_values is None:
                raise ValueError(f"operator {self.kind!r} needs field_values for its state-dependent Jacobian.")
            jac = self.jacobian_at_values(grid, np.asarray(field_values, dtype=float), observation.location)
        elif self.jacobian is not None:
            jac = self.jacobian(grid, observation.location)
        else:
            raise ValueError(f"operator {self.kind!r} has no Jacobian.")
        jac = np.atleast_2d(np.asarray(jac, dtype=float))
        if jac.shape != (observation.n, grid.n):
            raise ValueError(f"operator {self.kind!r} Jacobian shape {jac.shape} != ({observation.n}, {grid.n}).")
        return jac


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

    def capability_report(self) -> dict[str, dict[str, Any]]:
        """Return derivative/fallback declarations for all registered operators."""
        return {kind: op.capability_report() for kind, op in sorted(self._ops.items())}

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


def dc_resistivity_forward_operator(
    shape: Sequence[int],
    schedule: Sequence[Sequence[int | None]],
    *,
    spacing=1.0,
    sigma_ref: float = 1.0,
    log_data: bool = True,
    clamp: float | None = 12.0,
    finite_difference_step: float = 1.0e-4,
    adjoint: bool = False,
) -> ForwardOperator:
    """Wrap :func:`mixle_pde.geophysics.dc_resistivity` as a nonlinear observation operator.

    The latent field is log-conductivity contrast over a structured grid whose flattened size must match
    ``prod(shape)``. ``Observation.location`` is still required by the common observation contract; for
    ERT it is only metadata with one row per quadrupole (for example, survey midpoints). The electrode
    geometry lives in ``schedule`` as node-index quadrupoles ``(a, b, m, n)``.

    The local Jacobian is central finite-difference by default. That is intentionally a small/medium
    reference path for posterior construction and validation. Pass ``adjoint=True`` for production
    scale: the Jacobian is then computed by :func:`mixle_pde.adjoint.torch_adjoint_jacobian` -- one
    differentiable forward (reusing :func:`mixle_pde.geophysics.dc_resistivity`'s existing
    ``pde_solve.sparse_solve`` factorization per unique current injection) plus one O(1) adjoint solve
    per quadrupole, in place of ``2 * n_model`` finite-difference evaluations. Numerically equivalent to
    the finite-difference Jacobian (same ``local_jacobian`` contract); only the cost differs.
    """
    if finite_difference_step <= 0.0:
        raise ValueError("finite_difference_step must be positive.")
    shape = tuple(int(s) for s in shape)
    n_model = int(np.prod(shape))
    sched = tuple(tuple(None if item is None else int(item) for item in row) for row in schedule)
    if not sched:
        raise ValueError("schedule must contain at least one quadrupole.")

    def _check(grid: Field3D, obs_locations: np.ndarray) -> None:
        if grid.n != n_model:
            raise ValueError(f"grid.n={grid.n} does not match prod(shape)={n_model}.")
        if len(obs_locations) != len(sched):
            raise ValueError(f"observation locations must have one row per quadrupole ({len(sched)}).")

    def _predict_array(field_values: np.ndarray) -> np.ndarray:
        import torch

        from mixle_pde.geophysics import dc_resistivity

        log_sigma = torch.as_tensor(np.asarray(field_values, dtype=float), dtype=torch.float64)
        with torch.no_grad():
            out = dc_resistivity(
                log_sigma,
                shape,
                sched,
                spacing=spacing,
                sigma_ref=sigma_ref,
                log_data=log_data,
                clamp=clamp,
            )
        return np.asarray(out.detach().cpu().numpy(), dtype=float)

    def _predict_torch(log_sigma: Any) -> Any:
        from mixle_pde.geophysics import dc_resistivity

        return dc_resistivity(
            log_sigma,
            shape,
            sched,
            spacing=spacing,
            sigma_ref=sigma_ref,
            log_data=log_data,
            clamp=clamp,
        )

    def predict(grid: Field3D, field_values: np.ndarray, obs_locations: np.ndarray) -> np.ndarray:
        _check(grid, np.atleast_2d(np.asarray(obs_locations, dtype=float)))
        return _predict_array(np.asarray(field_values, dtype=float))

    def jacobian_at_values_fd(grid: Field3D, field_values: np.ndarray, obs_locations: np.ndarray) -> np.ndarray:
        _check(grid, np.atleast_2d(np.asarray(obs_locations, dtype=float)))
        x = np.asarray(field_values, dtype=float).reshape(-1)
        if x.shape != (n_model,):
            raise ValueError(f"field_values must have shape ({n_model},).")
        base_step = float(finite_difference_step)
        jac = np.empty((len(sched), n_model), dtype=float)
        for i in range(n_model):
            step = base_step * max(1.0, abs(float(x[i])))
            plus = x.copy()
            minus = x.copy()
            plus[i] += step
            minus[i] -= step
            jac[:, i] = (_predict_array(plus) - _predict_array(minus)) / (2.0 * step)
        return jac

    def jacobian_at_values_adjoint(grid: Field3D, field_values: np.ndarray, obs_locations: np.ndarray) -> np.ndarray:
        from mixle_pde.adjoint import torch_adjoint_jacobian

        _check(grid, np.atleast_2d(np.asarray(obs_locations, dtype=float)))
        x = np.asarray(field_values, dtype=float).reshape(-1)
        if x.shape != (n_model,):
            raise ValueError(f"field_values must have shape ({n_model},).")
        return torch_adjoint_jacobian(_predict_torch, x, n_obs=len(sched))

    if adjoint:
        return ForwardOperator(
            "dc_resistivity",
            predict,
            jacobian_at_values=jacobian_at_values_adjoint,
            differentiable=True,
            jacobian_kind="adjoint",
            has_true_adjoint=True,
        )
    return ForwardOperator(
        "dc_resistivity",
        predict,
        jacobian_at_values=jacobian_at_values_fd,
        differentiable=True,
        jacobian_kind="finite_difference",
        finite_difference_step=finite_difference_step,
    )


def layered_mt_forward_operator(
    frequencies: Sequence[float],
    thicknesses: Sequence[float],
    *,
    component: str = "log_apparent_resistivity",
    sigma_ref: float = 1.0,
    mu: float | None = None,
    finite_difference_step: float = 1.0e-4,
) -> ForwardOperator:
    """Wrap the 1-D layered magnetotelluric forward as a nonlinear posterior observation operator.

    The latent field is log-conductivity contrast for one vertical layer stack, with ``grid.n`` equal to
    ``len(thicknesses) + 1``. ``frequencies`` fixes the sounding frequencies; ``Observation.location`` is
    metadata with one row per frequency so the common observation contract still has matching
    ``location/value/noise_cov`` lengths. Supported components are ``"apparent_resistivity"``,
    ``"log_apparent_resistivity"``, and ``"phase"``.

    The local Jacobian is central finite-difference in the log-conductivity parameters. This is a
    deterministic reference path for Gauss-Newton/Laplace posterior construction; large production MT
    inversions should replace it with an adjoint sensitivity. (finite-difference Jacobian: O(2n) forward
    solves; adjoint sensitivities are C1)
    """
    if finite_difference_step <= 0.0:
        raise ValueError("finite_difference_step must be positive.")
    freqs = np.atleast_1d(np.asarray(frequencies, dtype=float))
    if freqs.ndim != 1 or freqs.size == 0 or np.any(freqs <= 0.0):
        raise ValueError("frequencies must be a non-empty positive 1D sequence.")
    thick = np.atleast_1d(np.asarray(thicknesses, dtype=float))
    if thick.ndim != 1:
        raise ValueError("thicknesses must be a 1D sequence.")
    if np.any(thick <= 0.0):
        raise ValueError("thicknesses must be positive.")
    if sigma_ref <= 0.0:
        raise ValueError("sigma_ref must be positive.")
    component = str(component)
    allowed = {"apparent_resistivity", "log_apparent_resistivity", "phase"}
    if component not in allowed:
        raise ValueError(f"component must be one of {sorted(allowed)}.")
    n_model = int(thick.size + 1)
    kind = f"layered_mt_{component}"

    def _check(grid: Field3D, obs_locations: np.ndarray) -> None:
        if grid.n != n_model:
            raise ValueError(f"grid.n={grid.n} does not match len(thicknesses)+1={n_model}.")
        if len(obs_locations) != freqs.size:
            raise ValueError(f"observation locations must have one row per frequency ({freqs.size}).")

    def _predict_array(field_values: np.ndarray) -> np.ndarray:
        import torch

        from mixle_pde.em_diffusion import MU0, layered_mt_impedance

        log_sigma = torch.as_tensor(np.asarray(field_values, dtype=float), dtype=torch.float64)
        if log_sigma.shape != (n_model,):
            raise ValueError(f"field_values must have shape ({n_model},).")
        permeability = MU0 if mu is None else float(mu)
        with torch.no_grad():
            sigma = float(sigma_ref) * torch.exp(log_sigma)
            rho_a, phase, _ = layered_mt_impedance(sigma, thick, freqs, mu=permeability)
            if component == "apparent_resistivity":
                out = rho_a
            elif component == "log_apparent_resistivity":
                out = torch.log(rho_a)
            else:
                out = phase
        return np.asarray(out.detach().cpu().numpy(), dtype=float)

    def predict(grid: Field3D, field_values: np.ndarray, obs_locations: np.ndarray) -> np.ndarray:
        _check(grid, np.atleast_2d(np.asarray(obs_locations, dtype=float)))
        return _predict_array(np.asarray(field_values, dtype=float))

    def jacobian_at_values(grid: Field3D, field_values: np.ndarray, obs_locations: np.ndarray) -> np.ndarray:
        _check(grid, np.atleast_2d(np.asarray(obs_locations, dtype=float)))
        x = np.asarray(field_values, dtype=float).reshape(-1)
        if x.shape != (n_model,):
            raise ValueError(f"field_values must have shape ({n_model},).")
        base_step = float(finite_difference_step)
        jac = np.empty((freqs.size, n_model), dtype=float)
        for i in range(n_model):
            step = base_step * max(1.0, abs(float(x[i])))
            plus = x.copy()
            minus = x.copy()
            plus[i] += step
            minus[i] -= step
            jac[:, i] = (_predict_array(plus) - _predict_array(minus)) / (2.0 * step)
        return jac

    return ForwardOperator(
        kind,
        predict,
        jacobian_at_values=jacobian_at_values,
        differentiable=True,
        jacobian_kind="finite_difference",
        finite_difference_step=finite_difference_step,
    )


def aem_layered_forward_operator(
    frequencies: Sequence[float],
    thicknesses: Sequence[float],
    *,
    component: str = "log_apparent_conductivity",
    sigma_ref: float = 1.0,
    mu: float | None = None,
    finite_difference_step: float = 1.0e-4,
) -> ForwardOperator:
    """Wrap the 1-D layered frequency-domain AEM forward as a nonlinear posterior observation operator.

    This uses the package's layered diffusive-EM impedance recursion and reports AEM-style apparent
    conductivity. The latent field is log-conductivity contrast for one vertical layer stack, with
    ``grid.n == len(thicknesses) + 1``. ``Observation.location`` carries one metadata row per frequency.
    Supported components are ``"apparent_conductivity"``, ``"log_apparent_conductivity"``, and
    ``"phase"``.

    The local Jacobian is central finite-difference in the log-conductivity parameters. This is the
    validated 1-D frequency-domain AEM reference path; production flight-line/loop geometry should use a
    dedicated airborne-source adjoint. (finite-difference Jacobian: O(2n) forward solves; adjoint
    sensitivities are C1)
    """
    if finite_difference_step <= 0.0:
        raise ValueError("finite_difference_step must be positive.")
    freqs = np.atleast_1d(np.asarray(frequencies, dtype=float))
    if freqs.ndim != 1 or freqs.size == 0 or np.any(freqs <= 0.0):
        raise ValueError("frequencies must be a non-empty positive 1D sequence.")
    thick = np.atleast_1d(np.asarray(thicknesses, dtype=float))
    if thick.ndim != 1:
        raise ValueError("thicknesses must be a 1D sequence.")
    if np.any(thick <= 0.0):
        raise ValueError("thicknesses must be positive.")
    if sigma_ref <= 0.0:
        raise ValueError("sigma_ref must be positive.")
    component = str(component)
    allowed = {"apparent_conductivity", "log_apparent_conductivity", "phase"}
    if component not in allowed:
        raise ValueError(f"component must be one of {sorted(allowed)}.")
    n_model = int(thick.size + 1)
    kind = f"aem_layered_{component}"

    def _check(grid: Field3D, obs_locations: np.ndarray) -> None:
        if grid.n != n_model:
            raise ValueError(f"grid.n={grid.n} does not match len(thicknesses)+1={n_model}.")
        if len(obs_locations) != freqs.size:
            raise ValueError(f"observation locations must have one row per frequency ({freqs.size}).")

    def _predict_array(field_values: np.ndarray) -> np.ndarray:
        import torch

        from mixle_pde.em_diffusion import MU0, layered_mt_impedance

        log_sigma = torch.as_tensor(np.asarray(field_values, dtype=float), dtype=torch.float64)
        if log_sigma.shape != (n_model,):
            raise ValueError(f"field_values must have shape ({n_model},).")
        permeability = MU0 if mu is None else float(mu)
        with torch.no_grad():
            sigma = float(sigma_ref) * torch.exp(log_sigma)
            rho_a, phase, _ = layered_mt_impedance(sigma, thick, freqs, mu=permeability)
            sigma_app = 1.0 / rho_a
            if component == "apparent_conductivity":
                out = sigma_app
            elif component == "log_apparent_conductivity":
                out = torch.log(sigma_app)
            else:
                out = phase
        return np.asarray(out.detach().cpu().numpy(), dtype=float)

    def predict(grid: Field3D, field_values: np.ndarray, obs_locations: np.ndarray) -> np.ndarray:
        _check(grid, np.atleast_2d(np.asarray(obs_locations, dtype=float)))
        return _predict_array(np.asarray(field_values, dtype=float))

    def jacobian_at_values(grid: Field3D, field_values: np.ndarray, obs_locations: np.ndarray) -> np.ndarray:
        _check(grid, np.atleast_2d(np.asarray(obs_locations, dtype=float)))
        x = np.asarray(field_values, dtype=float).reshape(-1)
        if x.shape != (n_model,):
            raise ValueError(f"field_values must have shape ({n_model},).")
        base_step = float(finite_difference_step)
        jac = np.empty((freqs.size, n_model), dtype=float)
        for i in range(n_model):
            step = base_step * max(1.0, abs(float(x[i])))
            plus = x.copy()
            minus = x.copy()
            plus[i] += step
            minus[i] -= step
            jac[:, i] = (_predict_array(plus) - _predict_array(minus)) / (2.0 * step)
        return jac

    return ForwardOperator(
        kind,
        predict,
        jacobian_at_values=jacobian_at_values,
        differentiable=True,
        jacobian_kind="finite_difference",
        finite_difference_step=finite_difference_step,
    )


def mt_2d_te_forward_operator(
    shape: Sequence[int],
    freq: float,
    *,
    component: str = "log_apparent_resistivity",
    spacing=1.0,
    sigma_ref: float = 1.0,
    mu: float | None = None,
    finite_difference_step: float = 1.0e-4,
) -> ForwardOperator:
    """Wrap the 2-D magnetotelluric TE forward as a nonlinear posterior observation operator.

    The latent field is flattened log-conductivity over a structured ``(nx, nz)`` grid. The observation
    has one row per surface site along ``nx``; ``Observation.location`` is metadata for those surface
    sites, while ``shape`` and ``freq`` define the MT solve. Supported components are
    ``"apparent_resistivity"``, ``"log_apparent_resistivity"``, and ``"phase"``.

    The local Jacobian is central finite-difference in the log-conductivity parameters. This is a
    small/medium reference path for posterior construction; production 2-D MT inversion should replace
    it with an adjoint sensitivity. (finite-difference Jacobian: O(2n) forward solves; adjoint
    sensitivities are C1)
    """
    if finite_difference_step <= 0.0:
        raise ValueError("finite_difference_step must be positive.")
    shape = tuple(int(s) for s in shape)
    if len(shape) != 2 or any(s < 2 for s in shape):
        raise ValueError("shape must be a 2D grid shape (nx, nz) with both dimensions >= 2.")
    if freq <= 0.0:
        raise ValueError("freq must be positive.")
    if sigma_ref <= 0.0:
        raise ValueError("sigma_ref must be positive.")
    component = str(component)
    allowed = {"apparent_resistivity", "log_apparent_resistivity", "phase"}
    if component not in allowed:
        raise ValueError(f"component must be one of {sorted(allowed)}.")
    nx, _ = shape
    n_model = int(np.prod(shape))
    kind = f"mt_2d_te_{component}"

    def _check(grid: Field3D, obs_locations: np.ndarray) -> None:
        if grid.n != n_model:
            raise ValueError(f"grid.n={grid.n} does not match prod(shape)={n_model}.")
        if len(obs_locations) != nx:
            raise ValueError(f"observation locations must have one row per surface site ({nx}).")

    def _predict_array(field_values: np.ndarray) -> np.ndarray:
        import torch

        from mixle_pde.em_diffusion import MU0, mt_2d_te

        log_sigma = torch.as_tensor(np.asarray(field_values, dtype=float), dtype=torch.float64)
        if log_sigma.shape != (n_model,):
            raise ValueError(f"field_values must have shape ({n_model},).")
        permeability = MU0 if mu is None else float(mu)
        with torch.no_grad():
            rho_a, phase = mt_2d_te(
                log_sigma,
                shape,
                float(freq),
                spacing=spacing,
                sigma_ref=float(sigma_ref),
                mu=permeability,
            )
            if component == "apparent_resistivity":
                out = rho_a
            elif component == "log_apparent_resistivity":
                out = torch.log(rho_a)
            else:
                out = phase
        return np.asarray(out.detach().cpu().numpy(), dtype=float)

    def predict(grid: Field3D, field_values: np.ndarray, obs_locations: np.ndarray) -> np.ndarray:
        _check(grid, np.atleast_2d(np.asarray(obs_locations, dtype=float)))
        return _predict_array(np.asarray(field_values, dtype=float))

    def jacobian_at_values(grid: Field3D, field_values: np.ndarray, obs_locations: np.ndarray) -> np.ndarray:
        _check(grid, np.atleast_2d(np.asarray(obs_locations, dtype=float)))
        x = np.asarray(field_values, dtype=float).reshape(-1)
        if x.shape != (n_model,):
            raise ValueError(f"field_values must have shape ({n_model},).")
        base_step = float(finite_difference_step)
        jac = np.empty((nx, n_model), dtype=float)
        for i in range(n_model):
            step = base_step * max(1.0, abs(float(x[i])))
            plus = x.copy()
            minus = x.copy()
            plus[i] += step
            minus[i] -= step
            jac[:, i] = (_predict_array(plus) - _predict_array(minus)) / (2.0 * step)
        return jac

    return ForwardOperator(
        kind,
        predict,
        jacobian_at_values=jacobian_at_values,
        differentiable=True,
        jacobian_kind="finite_difference",
        finite_difference_step=finite_difference_step,
    )


def mt_3d_forward_operator(
    shape: Sequence[int],
    frequencies: Sequence[float],
    *,
    component: str = "log_apparent_resistivity",
    polarization: str = "x",
    spacing=1.0,
    sigma_ref: float = 1.0,
    mu: float | None = None,
    gauge: float = 1.0,
    finite_difference_step: float = 1.0e-4,
    adjoint: bool = False,
) -> ForwardOperator:
    """Wrap the 3-D curl-curl MT forward as a nonlinear posterior observation operator.

    The latent field is flattened log-conductivity over a structured ``(nx, ny, nz)`` node grid.
    The observation has one row per sounding frequency; ``Observation.location`` is metadata for the
    shared observation contract, while ``shape`` and ``frequencies`` define the 3-D EM solves.
    Supported components are ``"apparent_resistivity"``, ``"log_apparent_resistivity"``, and
    ``"phase"``.

    The local Jacobian is central finite-difference in the log-conductivity parameters by default.
    Pass ``adjoint=True`` for production 3-D MT inversion: the Jacobian is then computed by
    :func:`mixle_pde.adjoint.torch_adjoint_jacobian` from a grad-enabled forward -- one curl-curl
    factorization per sounding frequency (reusing :func:`mixle_pde.em_diffusion_3d.mt_3d`'s existing
    ``pde_solve.sparse_solve`` call) plus one O(1) adjoint solve per frequency, instead of
    ``2 * n_model`` finite-difference evaluations.
    """
    if finite_difference_step <= 0.0:
        raise ValueError("finite_difference_step must be positive.")
    shape = tuple(int(s) for s in shape)
    if len(shape) != 3 or any(s < 2 for s in shape) or shape[2] < 4:
        raise ValueError("shape must be a 3D grid shape (nx, ny, nz) with nx, ny >= 2 and nz >= 4.")
    freqs = np.atleast_1d(np.asarray(frequencies, dtype=float))
    if freqs.ndim != 1 or freqs.size == 0 or np.any(freqs <= 0.0):
        raise ValueError("frequencies must be a non-empty positive 1D sequence.")
    if sigma_ref <= 0.0:
        raise ValueError("sigma_ref must be positive.")
    component = str(component)
    allowed = {"apparent_resistivity", "log_apparent_resistivity", "phase"}
    if component not in allowed:
        raise ValueError(f"component must be one of {sorted(allowed)}.")
    polarization = str(polarization)
    if polarization not in {"x", "y"}:
        raise ValueError("polarization must be 'x' or 'y'.")
    n_model = int(np.prod(shape))
    kind = f"mt_3d_{component}"

    def _check(grid: Field3D, obs_locations: np.ndarray) -> None:
        if grid.n != n_model:
            raise ValueError(f"grid.n={grid.n} does not match prod(shape)={n_model}.")
        if len(obs_locations) != freqs.size:
            raise ValueError(f"observation locations must have one row per frequency ({freqs.size}).")

    def _predict_array(field_values: np.ndarray) -> np.ndarray:
        import torch

        from mixle_pde.em_diffusion_3d import MU0, mt_3d

        log_sigma = torch.as_tensor(np.asarray(field_values, dtype=float), dtype=torch.float64)
        if log_sigma.shape != (n_model,):
            raise ValueError(f"field_values must have shape ({n_model},).")
        permeability = MU0 if mu is None else float(mu)
        out = []
        with torch.no_grad():
            for freq in freqs:
                rho_a, phase, _ = mt_3d(
                    log_sigma,
                    shape,
                    float(freq),
                    polarization=polarization,
                    spacing=spacing,
                    sigma_ref=float(sigma_ref),
                    mu=permeability,
                    gauge=float(gauge),
                )
                if component == "apparent_resistivity":
                    out.append(rho_a)
                elif component == "log_apparent_resistivity":
                    out.append(torch.log(rho_a))
                else:
                    out.append(phase)
            predicted = torch.stack(out)
        return np.asarray(predicted.detach().cpu().numpy(), dtype=float)

    def _predict_torch(log_sigma: Any) -> Any:
        import torch

        from mixle_pde.em_diffusion_3d import MU0, mt_3d

        if log_sigma.shape != (n_model,):
            raise ValueError(f"field_values must have shape ({n_model},).")
        permeability = MU0 if mu is None else float(mu)
        out = []
        for freq in freqs:
            rho_a, phase, _ = mt_3d(
                log_sigma,
                shape,
                float(freq),
                polarization=polarization,
                spacing=spacing,
                sigma_ref=float(sigma_ref),
                mu=permeability,
                gauge=float(gauge),
            )
            if component == "apparent_resistivity":
                out.append(rho_a)
            elif component == "log_apparent_resistivity":
                out.append(torch.log(rho_a))
            else:
                out.append(phase)
        return torch.stack(out)

    def predict(grid: Field3D, field_values: np.ndarray, obs_locations: np.ndarray) -> np.ndarray:
        _check(grid, np.atleast_2d(np.asarray(obs_locations, dtype=float)))
        return _predict_array(np.asarray(field_values, dtype=float))

    def jacobian_at_values_fd(grid: Field3D, field_values: np.ndarray, obs_locations: np.ndarray) -> np.ndarray:
        _check(grid, np.atleast_2d(np.asarray(obs_locations, dtype=float)))
        x = np.asarray(field_values, dtype=float).reshape(-1)
        if x.shape != (n_model,):
            raise ValueError(f"field_values must have shape ({n_model},).")
        base_step = float(finite_difference_step)
        jac = np.empty((freqs.size, n_model), dtype=float)
        for i in range(n_model):
            step = base_step * max(1.0, abs(float(x[i])))
            plus = x.copy()
            minus = x.copy()
            plus[i] += step
            minus[i] -= step
            jac[:, i] = (_predict_array(plus) - _predict_array(minus)) / (2.0 * step)
        return jac

    def jacobian_at_values_adjoint(grid: Field3D, field_values: np.ndarray, obs_locations: np.ndarray) -> np.ndarray:
        from mixle_pde.adjoint import torch_adjoint_jacobian

        _check(grid, np.atleast_2d(np.asarray(obs_locations, dtype=float)))
        x = np.asarray(field_values, dtype=float).reshape(-1)
        if x.shape != (n_model,):
            raise ValueError(f"field_values must have shape ({n_model},).")
        return torch_adjoint_jacobian(_predict_torch, x, n_obs=int(freqs.size))

    if adjoint:
        return ForwardOperator(
            kind,
            predict,
            jacobian_at_values=jacobian_at_values_adjoint,
            differentiable=True,
            jacobian_kind="adjoint",
            has_true_adjoint=True,
        )
    return ForwardOperator(
        kind,
        predict,
        jacobian_at_values=jacobian_at_values_fd,
        differentiable=True,
        jacobian_kind="finite_difference",
        finite_difference_step=finite_difference_step,
    )


def csem_3d_forward_operator(
    shape: Sequence[int],
    freq: float,
    source_edges: Sequence[int],
    receiver_edges: Sequence[int],
    *,
    component: str = "log_amplitude",
    source_amp: float | complex = 1.0,
    spacing=1.0,
    sigma_ref: float = 1.0,
    mu: float | None = None,
    gauge: float = 1.0,
    amplitude_floor: float = 1.0e-30,
    finite_difference_step: float = 1.0e-4,
    adjoint: bool = False,
) -> ForwardOperator:
    """Wrap the 3-D curl-curl CSEM forward as a nonlinear posterior observation operator.

    The latent field is flattened log-conductivity over a structured ``(nx, ny, nz)`` node grid.
    ``source_edges`` and ``receiver_edges`` are flat Yee-edge indices in the ordering used by
    :func:`mixle_pde.em_diffusion_3d.csem_3d`. ``Observation.location`` remains human-readable
    receiver metadata with one row per receiver edge. Supported real-valued components are
    ``"real"``, ``"imag"``, ``"amplitude"``, and ``"log_amplitude"``.

    The local Jacobian is central finite-difference in log-conductivity by default. Pass
    ``adjoint=True`` for production CSEM inversion: the Jacobian is then computed by
    :func:`mixle_pde.adjoint.torch_adjoint_jacobian` from a grad-enabled forward -- one curl-curl
    factorization (reusing :func:`mixle_pde.em_diffusion_3d.csem_3d`'s existing
    ``pde_solve.sparse_solve`` call) plus one O(1) adjoint solve per receiver edge, instead of
    ``2 * n_model`` finite-difference evaluations.
    """
    if finite_difference_step <= 0.0:
        raise ValueError("finite_difference_step must be positive.")
    shape = tuple(int(s) for s in shape)
    if len(shape) != 3 or any(s < 2 for s in shape):
        raise ValueError("shape must be a 3D grid shape (nx, ny, nz) with all dimensions >= 2.")
    if freq <= 0.0:
        raise ValueError("freq must be positive.")
    if sigma_ref <= 0.0:
        raise ValueError("sigma_ref must be positive.")
    if amplitude_floor <= 0.0:
        raise ValueError("amplitude_floor must be positive.")
    component = str(component)
    allowed = {"real", "imag", "amplitude", "log_amplitude"}
    if component not in allowed:
        raise ValueError(f"component must be one of {sorted(allowed)}.")

    nx, ny, nz = shape
    n_edge = (nx - 1) * ny * nz + nx * (ny - 1) * nz + nx * ny * (nz - 1)
    src = tuple(int(edge) for edge in source_edges)
    rcv = tuple(int(edge) for edge in receiver_edges)
    if not src:
        raise ValueError("source_edges must contain at least one edge index.")
    if not rcv:
        raise ValueError("receiver_edges must contain at least one edge index.")
    if min(src) < 0 or max(src) >= n_edge:
        raise ValueError(f"source_edges must be in [0, {n_edge}).")
    if min(rcv) < 0 or max(rcv) >= n_edge:
        raise ValueError(f"receiver_edges must be in [0, {n_edge}).")

    n_model = int(np.prod(shape))
    kind = f"csem_3d_{component}"

    def _check(grid: Field3D, obs_locations: np.ndarray) -> None:
        if grid.n != n_model:
            raise ValueError(f"grid.n={grid.n} does not match prod(shape)={n_model}.")
        if len(obs_locations) != len(rcv):
            raise ValueError(f"observation locations must have one row per receiver edge ({len(rcv)}).")

    def _predict_array(field_values: np.ndarray) -> np.ndarray:
        import torch

        from mixle_pde.em_diffusion_3d import MU0, csem_3d

        log_sigma = torch.as_tensor(np.asarray(field_values, dtype=float), dtype=torch.float64)
        if log_sigma.shape != (n_model,):
            raise ValueError(f"field_values must have shape ({n_model},).")
        permeability = MU0 if mu is None else float(mu)
        with torch.no_grad():
            field = csem_3d(
                log_sigma,
                shape,
                float(freq),
                source_edges=src,
                source_amp=source_amp,
                spacing=spacing,
                sigma_ref=float(sigma_ref),
                mu=permeability,
                gauge=float(gauge),
            )[list(rcv)]
            if component == "real":
                out = field.real
            elif component == "imag":
                out = field.imag
            elif component == "amplitude":
                out = field.abs()
            else:
                out = torch.log(torch.clamp(field.abs(), min=float(amplitude_floor)))
        return np.asarray(out.detach().cpu().numpy(), dtype=float)

    def _predict_torch(log_sigma: Any) -> Any:
        import torch

        from mixle_pde.em_diffusion_3d import MU0, csem_3d

        if log_sigma.shape != (n_model,):
            raise ValueError(f"field_values must have shape ({n_model},).")
        permeability = MU0 if mu is None else float(mu)
        field = csem_3d(
            log_sigma,
            shape,
            float(freq),
            source_edges=src,
            source_amp=source_amp,
            spacing=spacing,
            sigma_ref=float(sigma_ref),
            mu=permeability,
            gauge=float(gauge),
        )[list(rcv)]
        if component == "real":
            return field.real
        if component == "imag":
            return field.imag
        if component == "amplitude":
            return field.abs()
        return torch.log(torch.clamp(field.abs(), min=float(amplitude_floor)))

    def predict(grid: Field3D, field_values: np.ndarray, obs_locations: np.ndarray) -> np.ndarray:
        _check(grid, np.atleast_2d(np.asarray(obs_locations, dtype=float)))
        return _predict_array(np.asarray(field_values, dtype=float))

    def jacobian_at_values_fd(grid: Field3D, field_values: np.ndarray, obs_locations: np.ndarray) -> np.ndarray:
        _check(grid, np.atleast_2d(np.asarray(obs_locations, dtype=float)))
        x = np.asarray(field_values, dtype=float).reshape(-1)
        if x.shape != (n_model,):
            raise ValueError(f"field_values must have shape ({n_model},).")
        base_step = float(finite_difference_step)
        jac = np.empty((len(rcv), n_model), dtype=float)
        for i in range(n_model):
            step = base_step * max(1.0, abs(float(x[i])))
            plus = x.copy()
            minus = x.copy()
            plus[i] += step
            minus[i] -= step
            jac[:, i] = (_predict_array(plus) - _predict_array(minus)) / (2.0 * step)
        return jac

    def jacobian_at_values_adjoint(grid: Field3D, field_values: np.ndarray, obs_locations: np.ndarray) -> np.ndarray:
        from mixle_pde.adjoint import torch_adjoint_jacobian

        _check(grid, np.atleast_2d(np.asarray(obs_locations, dtype=float)))
        x = np.asarray(field_values, dtype=float).reshape(-1)
        if x.shape != (n_model,):
            raise ValueError(f"field_values must have shape ({n_model},).")
        return torch_adjoint_jacobian(_predict_torch, x, n_obs=len(rcv))

    if adjoint:
        return ForwardOperator(
            kind,
            predict,
            jacobian_at_values=jacobian_at_values_adjoint,
            differentiable=True,
            jacobian_kind="adjoint",
            has_true_adjoint=True,
        )
    return ForwardOperator(
        kind,
        predict,
        jacobian_at_values=jacobian_at_values_fd,
        differentiable=True,
        jacobian_kind="finite_difference",
        finite_difference_step=finite_difference_step,
    )
