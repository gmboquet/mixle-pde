"""Cross-modal subsurface reasoning: fuse geophysical modalities into a belief with UQ.

The domain-neutral machinery lives in core ``mixle`` -- :func:`mixle.reason.reason` folds
linear-Gaussian evidence into a belief state by exact Kalman assimilation, and reports per-modality
information gain and epistemic/aleatoric prediction splits. This module supplies the *physics*: it
turns the geophysical forward operators in :mod:`mixle_pde.geophysics` into the ``(H, y, R)``
evidence that front door consumes.

:class:`JointPotentialField` reasons jointly about subsurface **density contrast** ``rho`` and
**magnetic susceptibility** ``kappa`` over a shared mesh, from gravity and magnetic total-field
anomaly data. The latent is ``z = [rho, kappa]``; gravity reads the ``rho`` block, magnetic the
``kappa`` block. A per-cell petrophysical correlation in the prior lets each modality inform the
other -- so a gravity survey sharpens the susceptibility belief, and vice versa, exactly the
cross-modal information flow the reasoning layer is built to express.

This is the application layer: general reasoning stays in ``mixle.reason``; only the forward models
and the density/susceptibility semantics live here.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from mixle.reason import (
    CrossModalStore,
    Evidence,
    GaussianBelief,
    Latent,
    ReasonedAnswer,
    block_selector,
    reason,
)

from mixle_pde.dynamics import DynamicsOperator
from mixle_pde.field_priors import FaciesMixturePrior
from mixle_pde.geophysics import gravity_point_sensitivity, magnetic_dipole_sensitivity
from mixle_pde.latent import Field3D


class JointPotentialField:
    """Joint Bayesian reasoning about subsurface density (``rho``) and susceptibility (``kappa``).

    Args:
        cells: ``(n_cells, 3)`` cell-centre coordinates (east, north, up), metres.
        volumes: ``(n_cells,)`` cell volumes (m^3; scalar broadcast allowed).
        rho_sd: prior std of the per-cell density contrast (kg/m^3).
        kappa_sd: prior std of the per-cell susceptibility (SI).
        correlation: per-cell prior correlation between ``rho`` and ``kappa`` (in ``[-1, 1]``). A
            positive value (dense rock tends to be more magnetic) couples the modalities so gravity
            informs ``kappa`` and magnetic informs ``rho``. Ignored when ``facies`` is given.
        facies: an optional :class:`~mixle_pde.field_priors.FaciesMixturePrior` (C7) over ``(rho, kappa)``;
            when given, it REPLACES the single-``correlation`` coupling with the responsibility-weighted
            facies-mixture precision, so a bimodal rock-physics cloud (e.g. two rock units, each its own
            density/susceptibility trend) is representable. Call :meth:`refresh_facies_prior` after a
            `reason` pass to re-fit the facies assignment (EM) to the posterior mean and rebuild the prior.
    """

    def __init__(
        self,
        cells: Any,
        volumes: Any = 1.0e6,
        *,
        rho_sd: float = 200.0,
        kappa_sd: float = 0.05,
        correlation: float = 0.0,
        facies: FaciesMixturePrior | None = None,
    ) -> None:
        self.cells = np.asarray(cells, dtype=float)
        self.n = len(self.cells)
        self.volumes = np.broadcast_to(np.asarray(volumes, dtype=float), (self.n,)).copy()
        if not -1.0 <= correlation <= 1.0:
            raise ValueError("correlation must be in [-1, 1]")
        self.rho_sd = float(rho_sd)
        self.kappa_sd = float(kappa_sd)
        self.correlation = float(correlation)
        self.facies = facies
        self.prior = self._build_prior()

    # -- latent layout ------------------------------------------------------------------------
    @property
    def rho_index(self) -> np.ndarray:
        """Latent coordinate indices of the density block."""
        return np.arange(0, self.n)

    @property
    def kappa_index(self) -> np.ndarray:
        """Latent coordinate indices of the susceptibility block."""
        return np.arange(self.n, 2 * self.n)

    def _build_prior(self) -> GaussianBelief:
        n = self.n
        if self.facies is not None:
            return self._build_facies_prior()
        cov = np.zeros((2 * n, 2 * n))
        cov[:n, :n] = self.rho_sd**2 * np.eye(n)
        cov[n:, n:] = self.kappa_sd**2 * np.eye(n)
        cross = self.correlation * self.rho_sd * self.kappa_sd * np.eye(n)
        cov[:n, n:] = cross
        cov[n:, :n] = cross
        return GaussianBelief(np.zeros(2 * n), cov)

    def _build_facies_prior(self) -> GaussianBelief:
        """C7 wiring: build the joint ``[rho; kappa]`` covariance from the facies-mixture precision instead
        of the single-``correlation`` coupling. With no ``(rho, kappa)`` point estimate yet at construction
        time, the initial per-cell responsibilities are the mixture's global facies weights broadcast over
        every cell; :meth:`refresh_facies_prior` re-fits them (EM) once a posterior mean is available."""
        n = self.n
        grid = Field3D(coordinates=self.cells, spacing=1.0, units="", property_name="rho_kappa")
        k = self.facies.means.shape[0]
        responsibilities = np.broadcast_to(self.facies.weights, (n, k)).copy()
        precision = self.facies.precision_sparse(grid, responsibilities=responsibilities).toarray()
        cov = np.linalg.inv(precision + 1.0e-9 * np.eye(2 * n))
        return GaussianBelief(np.zeros(2 * n), cov)

    def refresh_facies_prior(self, answer: ReasonedAnswer) -> None:
        """Re-fit the facies responsibilities (EM) to a joint ``answer``'s posterior mean and rebuild
        ``self.prior`` from the updated mixture -- alternates the Gauss-Newton/EM loop of DR-ALG C7 with the
        reasoning layer's belief update. Requires the model to have been built with a ``facies`` prior."""
        if self.facies is None:
            raise ValueError("refresh_facies_prior requires a facies-mixture prior (construct with facies=...).")
        mean = np.asarray(answer.mean, dtype=float)
        self.facies.em_update(mean[self.rho_index], mean[self.kappa_index])
        self.prior = self._build_prior()

    # -- modality evidence --------------------------------------------------------------------
    def gravity(self, obs: Any, data: Any, *, noise_sd: float, name: str = "gravity") -> Evidence:
        """Gravity-anomaly evidence: ``d = G_grav @ rho + noise`` (mGal). Reads the density block."""
        G = gravity_point_sensitivity(obs, self.cells, self.volumes)  # (n_obs, n)
        H = np.hstack([G, np.zeros_like(G)])  # reads rho, ignores kappa
        y = np.atleast_1d(np.asarray(data, dtype=float))
        return Evidence(H, y, float(noise_sd) ** 2 * np.eye(H.shape[0]), name)

    def magnetic(
        self,
        obs: Any,
        data: Any,
        *,
        inclination: float,
        declination: float,
        noise_sd: float,
        field_nt: float = 50000.0,
        name: str = "magnetic",
    ) -> Evidence:
        """Magnetic total-field evidence: ``d = G_mag @ kappa + noise`` (nT). Reads the susceptibility block."""
        G = magnetic_dipole_sensitivity(
            obs, self.cells, self.volumes, inclination=inclination, declination=declination, field_nt=field_nt
        )  # (n_obs, n)
        H = np.hstack([np.zeros_like(G), G])  # reads kappa, ignores rho
        y = np.atleast_1d(np.asarray(data, dtype=float))
        return Evidence(H, y, float(noise_sd) ** 2 * np.eye(H.shape[0]), name)

    # -- reasoning ----------------------------------------------------------------------------
    def reason(self, evidence: Any) -> ReasonedAnswer:
        """Fuse a list of modality evidence into the joint posterior belief over ``[rho, kappa]``."""
        return reason(self.prior, list(evidence))

    def density(self, answer: ReasonedAnswer) -> ReasonedAnswer:
        """The marginal belief over the density field ``rho`` from a joint answer."""
        return answer.marginal(self.rho_index)

    def susceptibility(self, answer: ReasonedAnswer) -> ReasonedAnswer:
        """The marginal belief over the susceptibility field ``kappa`` from a joint answer."""
        return answer.marginal(self.kappa_index)


class SpatialFieldStore:
    """Cross-modal RAG over ONE spatially-indexed volume -- the seismic-block pattern.

    A large property volume is too lossy to compress into a single embedding; instead it is tiled
    into local sub-volumes indexed by their location. A location-anchored query ("what is the
    property here?") retrieves the nearby tiles and conditions the field belief on their **raw**
    per-cell measurements (a precise sub-volume observation), falling back to a cheap tile-average
    ("embedding") only where that already suffices. So no lossy compression sits between the data
    and the answer, and only the local neighborhood a query touches is ever conditioned on -- the
    point of RAG-over-raw-data for terabyte volumes.

    This is the application wrapper: it builds the tile index and the coarse/fine evidence, then
    hands off to the domain-neutral :class:`mixle.reason.CrossModalStore`.

    Args:
        cells: ``(n, dim)`` cell-centre coordinates of the volume.
        data: ``(n,)`` noisy per-cell measurement of the scalar property.
        tile_radius: cells within this distance of a tile centre form the tile (the sub-volume).
        coarse_sd: noise std of a tile's cheap average summary ("embedding" fidelity).
        fine_sd: noise std of the raw per-cell readout ("raw" fidelity).
    """

    def __init__(
        self,
        cells: Any,
        data: Any,
        *,
        tile_radius: float,
        coarse_sd: float = 1.0,
        fine_sd: float = 0.05,
    ) -> None:
        self.cells = np.asarray(cells, dtype=float)
        self.data = np.asarray(data, dtype=float).reshape(-1)
        self.n = len(self.cells)
        if len(self.data) != self.n:
            raise ValueError(f"{self.n} cells but {len(self.data)} data values")
        self.coarse_sd = float(coarse_sd)
        self.fine_sd = float(fine_sd)
        # one tile per cell: its spatial neighborhood within tile_radius (the sub-volume).
        self.centroids = self.cells.copy()
        self.tiles: list[np.ndarray] = [
            np.where(np.linalg.norm(self.cells - c, axis=1) <= tile_radius)[0] for c in self.cells
        ]

    def prior(self, sd: float = 1.0) -> GaussianBelief:
        """An isotropic Gaussian prior ``N(0, sd^2 I)`` over the per-cell property field."""
        return GaussianBelief(np.zeros(self.n), (float(sd) ** 2) * np.eye(self.n))

    def _fine(self, members: np.ndarray) -> Evidence:
        # raw sub-volume: read every cell in the tile individually (precise).
        H = np.eye(self.n)[members]
        return Evidence(H, self.data[members], self.fine_sd**2 * np.eye(len(members)), "raw")

    def _coarse(self, members: np.ndarray) -> Evidence:
        # lossy summary: one averaged readout over the tile (an "embedding" of the sub-volume).
        H = np.zeros((1, self.n))
        H[0, members] = 1.0 / len(members)
        return Evidence(H, [float(self.data[members].mean())], [[self.coarse_sd**2]], "embedding")

    def store(self) -> CrossModalStore:
        """The :class:`mixle.reason.CrossModalStore` over this volume (keys = tile centroids)."""
        return CrossModalStore(self.centroids, list(self.tiles), coarse=self._coarse, fine=self._fine)

    def nearest_cell(self, location: Any) -> int:
        """Index of the cell nearest ``location`` (the target of a location-anchored query)."""
        return int(np.argmin(np.linalg.norm(self.cells - np.asarray(location, dtype=float), axis=1)))


class MechanisticFieldReasoner:
    """Reconstruct a spatiotemporal PDE field from sparse sensors, with the PDE itself as the prior.

    A field ``u(x, t)`` on an ``n``-cell grid follows a linear PDE (diffusion / advection / ...). Its
    discretized one-step transition ``A = operator.transition_matrix(dt)`` drives a mechanistic
    trajectory prior (:meth:`mixle.reason.Latent.mechanistic`) over the whole space-time state
    ``u_0 .. u_{T-1}``. Sparse sensor readings at arbitrary ``(cell, step)`` are fused via
    :func:`mixle.reason.reason` -- exact Kalman smoothing -- so the entire field is reconstructed
    from a few sensors, the physics fills the gaps, and every cell/time carries honest uncertainty.

    Application layer: the PDE operators live in :mod:`mixle_pde.dynamics`; the trajectory prior,
    fusion, and smoothing come from the domain-neutral core.

    Args:
        operator: a :class:`mixle_pde.dynamics.DynamicsOperator` (diffusion, advection, ...).
        dt: time step between trajectory states.
        steps: number of time steps ``T``.
        x0_mean: prior mean of the initial field (default zeros, ``n``-vector).
        x0_sd: prior std of the initial field.
        process_sd: process-noise std per step (0 = deterministic dynamics).
    """

    def __init__(
        self,
        operator: DynamicsOperator,
        *,
        dt: float,
        steps: int,
        x0_mean: Any = None,
        x0_sd: float = 1.0,
        process_sd: float = 0.0,
    ) -> None:
        self.operator = operator
        self.n = int(operator.n)
        self.dt = float(dt)
        self.steps = int(steps)
        A = operator.transition_matrix(self.dt)
        self.prior = Latent.mechanistic(
            A,
            self.steps,
            x0_mean=None if x0_mean is None else np.asarray(x0_mean, dtype=float),
            x0_cov=(float(x0_sd) ** 2) * np.eye(self.n),
            process_cov=(float(process_sd) ** 2) * np.eye(self.n),
        )

    def sensor(self, *, cell: int, step: int, value: float, noise_sd: float, name: str = "") -> Evidence:
        """A sensor reading of the field at one ``cell`` and one time ``step``."""
        one_hot = np.eye(self.n)[[int(cell)]]  # (1, n) read a single cell
        H = block_selector(int(step), self.steps, self.n, within=one_hot)
        return Evidence(H, [float(value)], [[float(noise_sd) ** 2]], name or f"u[{cell}]@{step}")

    def reason(self, sensors: Any) -> ReasonedAnswer:
        """Fuse sensor readings into the smoothed posterior over the whole space-time field."""
        return reason(self.prior, list(sensors))

    def field(self, answer: ReasonedAnswer) -> np.ndarray:
        """Posterior-mean field as a ``(steps, n)`` array ``u[t, x]``."""
        return np.asarray(answer.mean).reshape(self.steps, self.n)

    def uncertainty(self, answer: ReasonedAnswer) -> np.ndarray:
        """Posterior std of the field as a ``(steps, n)`` array."""
        return answer.belief.sd().reshape(self.steps, self.n)
