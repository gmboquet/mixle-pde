"""Small-reference MCMC inversion for latent 3D fields.

The production ladder should prefer exact linear-Gaussian, sparse precision, MAP/Laplace, ensemble, or
variational routes when they apply. This module is the deliberately small Random-Walk Metropolis
reference path: it samples the posterior over a :class:`~mixle_pde.latent.Field3D` in unconstrained
coordinates using any registered observation likelihood, including nonlinear operators that do not expose
a Jacobian. The result is a :class:`~mixle_pde.latent.PosteriorFieldSamples3D` empirical posterior.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mixle_pde.field_inversion import FieldGaussianPrior
from mixle_pde.latent import Field3D, PosteriorFieldSamples3D
from mixle_pde.observations import ForwardOperatorRegistry, Observation


@dataclass(frozen=True)
class MCMCReport:
    """Diagnostics for a Random-Walk Metropolis field posterior sample."""

    iterations: int
    burn_in: int
    thin: int
    proposed: int
    accepted: int
    acceptance_rate: float
    stored_samples: int
    final_log_posterior: float
    best_log_posterior: float


def _proposal_scale(step_scale: float | np.ndarray, n: int) -> np.ndarray:
    scale = np.asarray(step_scale, dtype=float)
    if scale.ndim == 0:
        if float(scale) <= 0.0:
            raise ValueError("step_scale must be positive.")
        return np.full(n, float(scale))
    if scale.shape != (n,):
        raise ValueError(f"step_scale must be a positive scalar or have shape ({n},).")
    if np.any(scale <= 0.0):
        raise ValueError("step_scale entries must be positive.")
    return scale.copy()


def _prior_log_density_kernel(u: np.ndarray, mean: np.ndarray, precision: np.ndarray) -> float:
    delta = u - mean
    return float(-0.5 * delta @ precision @ delta)


def field_log_posterior_kernel(
    grid: Field3D,
    observations: list[Observation],
    registry: ForwardOperatorRegistry,
    prior: FieldGaussianPrior,
    unconstrained_values: np.ndarray,
    *,
    prior_mean: np.ndarray | None = None,
    prior_precision: np.ndarray | None = None,
) -> float:
    """Unnormalized ``log p(u | observations)`` for a field in unconstrained coordinates.

    The Gaussian prior is evaluated in unconstrained coordinates. Observation likelihoods see the
    physical field values produced by ``grid.from_unconstrained(u)``.
    """
    u = np.asarray(unconstrained_values, dtype=float)
    if u.shape != (grid.n,):
        raise ValueError(f"unconstrained_values must have shape ({grid.n},).")
    if not np.all(np.isfinite(u)):
        return -np.inf
    mean = prior.mean_vector(grid) if prior_mean is None else np.asarray(prior_mean, dtype=float)
    precision = prior.precision(grid) if prior_precision is None else np.asarray(prior_precision, dtype=float)
    if mean.shape != (grid.n,):
        raise ValueError(f"prior_mean must have shape ({grid.n},).")
    if precision.shape != (grid.n, grid.n):
        raise ValueError(f"prior_precision must have shape ({grid.n}, {grid.n}).")
    physical = grid.from_unconstrained(u)
    if not np.all(np.isfinite(physical)):
        return -np.inf
    return _prior_log_density_kernel(u, mean, precision) + registry.total_log_likelihood(grid, physical, observations)


def metropolis_field_invert(
    grid: Field3D,
    observations: list[Observation],
    registry: ForwardOperatorRegistry,
    prior: FieldGaussianPrior,
    *,
    n_samples: int,
    burn_in: int = 1000,
    thin: int = 1,
    step_scale: float | np.ndarray = 1.0,
    initial_unconstrained: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[PosteriorFieldSamples3D, MCMCReport]:
    """Sample a small field posterior with Random-Walk Metropolis.

    This is a validation/reference engine, not the scalable production smoother. It supports nonlinear
    and non-Gaussian observation likelihoods because it only requires the registry's predictive
    likelihood path; it does not require Jacobians or adjoints.
    """
    if not observations:
        raise ValueError("need at least one observation to invert.")
    n_samples = int(n_samples)
    burn_in = int(burn_in)
    thin = int(thin)
    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")
    if burn_in < 0:
        raise ValueError("burn_in must be non-negative.")
    if thin <= 0:
        raise ValueError("thin must be positive.")

    rng = np.random.default_rng() if rng is None else rng
    n = grid.n
    scale = _proposal_scale(step_scale, n)
    prior_mean = prior.mean_vector(grid)
    precision = np.asarray(prior.precision(grid), dtype=float)
    precision = 0.5 * (precision + precision.T)

    if initial_unconstrained is None:
        current = prior_mean.copy()
    else:
        current = np.asarray(initial_unconstrained, dtype=float)
        if current.shape != (n,):
            raise ValueError(f"initial_unconstrained must have shape ({n},).")
        current = current.copy()

    def logp(u: np.ndarray) -> float:
        return field_log_posterior_kernel(
            grid,
            observations,
            registry,
            prior,
            u,
            prior_mean=prior_mean,
            prior_precision=precision,
        )

    current_logp = logp(current)
    if not np.isfinite(current_logp):
        raise ValueError("initial_unconstrained has non-finite log posterior.")

    total_steps = burn_in + n_samples * thin
    accepted = 0
    best = current.copy()
    best_logp = float(current_logp)
    draws: list[np.ndarray] = []
    draw_logp: list[float] = []

    for step in range(1, total_steps + 1):
        proposal = current + rng.normal(size=n) * scale
        proposal_logp = logp(proposal)
        if np.isfinite(proposal_logp) and np.log(rng.random()) < proposal_logp - current_logp:
            current = proposal
            current_logp = float(proposal_logp)
            accepted += 1
            if current_logp > best_logp:
                best = current.copy()
                best_logp = float(current_logp)
        if step > burn_in and (step - burn_in) % thin == 0:
            draws.append(current.copy())
            draw_logp.append(float(current_logp))

    posterior = PosteriorFieldSamples3D(
        grid=grid,
        samples=np.asarray(draws, dtype=float),
        log_posterior=np.asarray(draw_logp, dtype=float),
        map=best,
        provenance={
            "method": "random_walk_metropolis",
            "small_reference": True,
            "burn_in": burn_in,
            "thin": thin,
            "step_scale": np.asarray(step_scale, dtype=float).tolist(),
        },
    )
    report = MCMCReport(
        iterations=total_steps,
        burn_in=burn_in,
        thin=thin,
        proposed=total_steps,
        accepted=accepted,
        acceptance_rate=accepted / total_steps if total_steps else 0.0,
        stored_samples=len(draws),
        final_log_posterior=float(current_logp),
        best_log_posterior=best_logp,
    )
    return posterior, report
