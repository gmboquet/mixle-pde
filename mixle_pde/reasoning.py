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
from mixle.reason import Evidence, GaussianBelief, ReasonedAnswer, reason

from mixle_pde.geophysics import gravity_point_sensitivity, magnetic_dipole_sensitivity


class JointPotentialField:
    """Joint Bayesian reasoning about subsurface density (``rho``) and susceptibility (``kappa``).

    Args:
        cells: ``(n_cells, 3)`` cell-centre coordinates (east, north, up), metres.
        volumes: ``(n_cells,)`` cell volumes (m^3; scalar broadcast allowed).
        rho_sd: prior std of the per-cell density contrast (kg/m^3).
        kappa_sd: prior std of the per-cell susceptibility (SI).
        correlation: per-cell prior correlation between ``rho`` and ``kappa`` (in ``[-1, 1]``). A
            positive value (dense rock tends to be more magnetic) couples the modalities so gravity
            informs ``kappa`` and magnetic informs ``rho``.
    """

    def __init__(
        self,
        cells: Any,
        volumes: Any = 1.0e6,
        *,
        rho_sd: float = 200.0,
        kappa_sd: float = 0.05,
        correlation: float = 0.0,
    ) -> None:
        self.cells = np.asarray(cells, dtype=float)
        self.n = len(self.cells)
        self.volumes = np.broadcast_to(np.asarray(volumes, dtype=float), (self.n,)).copy()
        if not -1.0 <= correlation <= 1.0:
            raise ValueError("correlation must be in [-1, 1]")
        self.rho_sd = float(rho_sd)
        self.kappa_sd = float(kappa_sd)
        self.correlation = float(correlation)
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
        cov = np.zeros((2 * n, 2 * n))
        cov[:n, :n] = self.rho_sd**2 * np.eye(n)
        cov[n:, n:] = self.kappa_sd**2 * np.eye(n)
        cross = self.correlation * self.rho_sd * self.kappa_sd * np.eye(n)
        cov[:n, n:] = cross
        cov[n:, :n] = cross
        return GaussianBelief(np.zeros(2 * n), cov)

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
