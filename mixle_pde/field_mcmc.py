"""Small-reference MCMC inversion for latent 3D fields.

The production ladder should prefer exact linear-Gaussian, sparse precision, MAP/Laplace, ensemble, or
variational routes when they apply. This module is the deliberately small Random-Walk Metropolis
reference path: it samples the posterior over a :class:`~mixle_pde.latent.Field3D` in unconstrained
coordinates using any registered observation likelihood, including nonlinear operators that do not expose
a Jacobian. The result is a :class:`~mixle_pde.latent.PosteriorFieldSamples3D` empirical posterior.

:func:`metropolis_field_invert`'s isotropic random-walk proposal is a poor fit for two situations:
survey-scale grids (the proposal has to shrink with ``grid.n`` to keep a usable acceptance rate) and
well-separated multimodal posteriors (a small local step almost never crosses a wide low-likelihood
gap). :func:`pcn_field_invert` fixes the first with a preconditioned Crank-Nicolson proposal that stays
dimension-robust; :func:`mala_field_invert` and :func:`hmc_field_invert` use the gradient of
:func:`field_log_posterior_kernel` (via each registered operator's Jacobian -- see
:func:`field_log_posterior_grad_kernel`) preconditioned by the prior precision. All three remain a
small/medium-scale reference path, not a replacement for the exact linear-Gaussian or Laplace routes;
their job is the multimodal / non-Gaussian fallback those exact paths cannot represent.
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


def _prior_precision_and_mean(grid: Field3D, prior: FieldGaussianPrior) -> tuple[np.ndarray, np.ndarray]:
    prior_mean = prior.mean_vector(grid)
    precision = np.asarray(prior.precision(grid), dtype=float)
    precision = 0.5 * (precision + precision.T)
    return prior_mean, precision


def _unconstrained_from_derivative(grid: Field3D, u: np.ndarray) -> np.ndarray:
    """Elementwise ``d(grid.from_unconstrained(u)) / du``, matching :meth:`Field3D.from_unconstrained`.

    The bound transform is applied cell-by-cell, so the Jacobian of the physical map is diagonal; this
    returns that diagonal (a length-``grid.n`` vector), which is all the chain rule needs to pull a
    gradient computed in physical units back into the unconstrained space the samplers walk in.
    """
    u = np.asarray(u, dtype=float)
    if grid.bounds is None:
        return np.ones_like(u)
    lo, hi = grid.bounds
    if lo is not None and hi is not None:
        sigmoid = 1.0 / (1.0 + np.exp(-u))
        return (hi - lo) * sigmoid * (1.0 - sigmoid)
    if lo is not None:
        return np.exp(u)
    if hi is not None:
        return -np.exp(u)
    return np.ones_like(u)


def field_log_posterior_grad_kernel(
    grid: Field3D,
    observations: list[Observation],
    registry: ForwardOperatorRegistry,
    prior: FieldGaussianPrior,
    unconstrained_values: np.ndarray,
    *,
    prior_mean: np.ndarray | None = None,
    prior_precision: np.ndarray | None = None,
) -> np.ndarray:
    """Gradient of :func:`field_log_posterior_kernel` w.r.t. ``unconstrained_values``.

    There is no reverse-mode autodiff engine in :mod:`mixle_pde`; this is the registry-based stand-in
    ``mala_field_invert``/``hmc_field_invert`` use for "autograd through the registry" -- it chains the
    Gaussian prior's exact gradient with each observation's registered Jacobian (fixed or
    state-dependent via :meth:`~mixle_pde.observations.ForwardOperator.local_jacobian`), pulled back
    through the field's bound transform by :func:`_unconstrained_from_derivative`. Every observation
    kind used with a gradient sampler must expose a Jacobian; one that does not raises ``ValueError``.
    """
    u = np.asarray(unconstrained_values, dtype=float)
    if u.shape != (grid.n,):
        raise ValueError(f"unconstrained_values must have shape ({grid.n},).")
    mean = prior.mean_vector(grid) if prior_mean is None else np.asarray(prior_mean, dtype=float)
    precision = np.asarray(prior.precision(grid) if prior_precision is None else prior_precision, dtype=float)
    physical = grid.from_unconstrained(u)
    grad_physical = np.zeros(grid.n, dtype=float)
    for obs in observations:
        op = registry.get(obs.kind)
        if not op.has_adjoint():
            raise ValueError(
                f"gradient-based field samplers require a Jacobian for observation kind {obs.kind!r}; "
                "register one via `jacobian` or `jacobian_at_values`."
            )
        predicted = op.predict_observation(grid, physical, obs)
        residual = obs.value - predicted
        weighted = residual / obs.noise_cov if obs.is_diagonal else np.linalg.solve(obs.noise_cov, residual)
        jac = op.local_jacobian(grid, physical, obs)
        grad_physical += jac.T @ weighted
    grad_prior = -(precision @ (u - mean))
    return grad_prior + _unconstrained_from_derivative(grid, u) * grad_physical


def pcn_field_invert(
    grid: Field3D,
    observations: list[Observation],
    registry: ForwardOperatorRegistry,
    prior: FieldGaussianPrior,
    *,
    n_samples: int,
    beta_pcn: float = 0.3,
    burn_in: int = 1000,
    thin: int = 1,
    initial_unconstrained: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[PosteriorFieldSamples3D, MCMCReport]:
    """Sample a field posterior with preconditioned Crank-Nicolson (pCN).

    The pCN proposal ``sqrt(1 - beta_pcn**2) * (u - prior_mean) + prior_mean + beta_pcn * xi`` (``xi ~
    N(0, prior_cov)``) is exactly reversible with respect to the Gaussian prior itself, so its
    Metropolis ratio only needs the likelihood -- unlike :func:`metropolis_field_invert`'s isotropic
    step, it never has to shrink as ``grid.n`` grows (dimension-robust), and every proposal injects a
    fresh prior-scaled draw rather than only a small local perturbation. That fresh-draw component is
    what lets pCN occasionally land near a second, well-separated posterior mode that an isotropic
    random walk essentially never reaches; it is the recommended multimodal fallback here.
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
    if not 0.0 < beta_pcn <= 1.0:
        raise ValueError("beta_pcn must be in (0, 1].")

    rng = np.random.default_rng() if rng is None else rng
    n = grid.n
    prior_mean, precision = _prior_precision_and_mean(grid, prior)
    chol_precision = np.linalg.cholesky(precision)

    if initial_unconstrained is None:
        current = prior_mean.copy()
    else:
        current = np.asarray(initial_unconstrained, dtype=float)
        if current.shape != (n,):
            raise ValueError(f"initial_unconstrained must have shape ({n},).")
        current = current.copy()

    def loglik(u: np.ndarray) -> float:
        physical = grid.from_unconstrained(u)
        if not np.all(np.isfinite(physical)):
            return -np.inf
        return registry.total_log_likelihood(grid, physical, observations)

    def logpost(u: np.ndarray, current_loglik: float) -> float:
        return float(_prior_log_density_kernel(u, prior_mean, precision) + current_loglik)

    current_loglik = loglik(current)
    if not np.isfinite(current_loglik):
        raise ValueError("initial_unconstrained has non-finite log likelihood.")

    shrink = float(np.sqrt(1.0 - beta_pcn**2))
    total_steps = burn_in + n_samples * thin
    accepted = 0
    best = current.copy()
    best_logp = logpost(current, current_loglik)
    draws: list[np.ndarray] = []
    draw_logp: list[float] = []

    for step in range(1, total_steps + 1):
        z = rng.standard_normal(n)
        xi = np.linalg.solve(chol_precision.T, z)  # xi ~ N(0, precision^-1) = N(0, prior_cov)
        proposal = prior_mean + shrink * (current - prior_mean) + beta_pcn * xi
        proposal_loglik = loglik(proposal)
        if np.isfinite(proposal_loglik) and np.log(rng.random()) < proposal_loglik - current_loglik:
            current = proposal
            current_loglik = float(proposal_loglik)
            accepted += 1
            candidate_logp = logpost(current, current_loglik)
            if candidate_logp > best_logp:
                best = current.copy()
                best_logp = candidate_logp
        if step > burn_in and (step - burn_in) % thin == 0:
            draws.append(current.copy())
            draw_logp.append(logpost(current, current_loglik))

    posterior = PosteriorFieldSamples3D(
        grid=grid,
        samples=np.asarray(draws, dtype=float),
        log_posterior=np.asarray(draw_logp, dtype=float),
        map=best,
        provenance={
            "method": "pcn",
            "small_reference": True,
            "burn_in": burn_in,
            "thin": thin,
            "beta_pcn": float(beta_pcn),
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
        final_log_posterior=logpost(current, current_loglik),
        best_log_posterior=best_logp,
    )
    return posterior, report


def mala_field_invert(
    grid: Field3D,
    observations: list[Observation],
    registry: ForwardOperatorRegistry,
    prior: FieldGaussianPrior,
    *,
    n_samples: int,
    step_size: float,
    burn_in: int = 1000,
    thin: int = 1,
    initial_unconstrained: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[PosteriorFieldSamples3D, MCMCReport]:
    """Sample a field posterior with prior-preconditioned MALA (Metropolis-adjusted Langevin).

    The Langevin drift/diffusion step uses the prior covariance (``prior.precision(grid)^-1``) as the
    preconditioner, so a correlated or anisotropic prior does not force an overly conservative isotropic
    step size. The drift uses :func:`field_log_posterior_grad_kernel`, so every observation used here
    must expose a registered Jacobian. The asymmetric-proposal Metropolis-Hastings correction is applied
    exactly (both the forward and reverse transition densities), not dropped as in unadjusted Langevin.
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
    if step_size <= 0.0:
        raise ValueError("step_size must be positive.")

    rng = np.random.default_rng() if rng is None else rng
    n = grid.n
    prior_mean, precision = _prior_precision_and_mean(grid, prior)
    prior_cov = np.linalg.inv(precision)
    chol_cov = np.linalg.cholesky(prior_cov + 1.0e-12 * np.eye(n))

    if initial_unconstrained is None:
        current = prior_mean.copy()
    else:
        current = np.asarray(initial_unconstrained, dtype=float)
        if current.shape != (n,):
            raise ValueError(f"initial_unconstrained must have shape ({n},).")
        current = current.copy()

    def logp(u: np.ndarray) -> float:
        return field_log_posterior_kernel(
            grid, observations, registry, prior, u, prior_mean=prior_mean, prior_precision=precision
        )

    def grad(u: np.ndarray) -> np.ndarray:
        return field_log_posterior_grad_kernel(
            grid, observations, registry, prior, u, prior_mean=prior_mean, prior_precision=precision
        )

    current_logp = logp(current)
    if not np.isfinite(current_logp):
        raise ValueError("initial_unconstrained has non-finite log posterior.")
    current_grad = grad(current)

    half_step2 = 0.5 * step_size**2
    total_steps = burn_in + n_samples * thin
    accepted = 0
    best = current.copy()
    best_logp = float(current_logp)
    draws: list[np.ndarray] = []
    draw_logp: list[float] = []

    for step in range(1, total_steps + 1):
        drift = half_step2 * (prior_cov @ current_grad)
        z = rng.standard_normal(n)
        proposal = current + drift + step_size * (chol_cov @ z)
        proposal_logp = logp(proposal)
        accept = False
        proposal_grad = None
        if np.isfinite(proposal_logp):
            proposal_grad = grad(proposal)
            f_fwd = proposal - current - drift
            f_back = current - proposal - half_step2 * (prior_cov @ proposal_grad)
            log_q_fwd = -0.5 / step_size**2 * float(f_fwd @ precision @ f_fwd)
            log_q_back = -0.5 / step_size**2 * float(f_back @ precision @ f_back)
            log_alpha = (proposal_logp - current_logp) + (log_q_back - log_q_fwd)
            accept = np.isfinite(log_alpha) and np.log(rng.random()) < log_alpha
        if accept:
            current = proposal
            current_logp = float(proposal_logp)
            current_grad = proposal_grad
            accepted += 1
            if current_logp > best_logp:
                best = current.copy()
                best_logp = current_logp
        if step > burn_in and (step - burn_in) % thin == 0:
            draws.append(current.copy())
            draw_logp.append(float(current_logp))

    posterior = PosteriorFieldSamples3D(
        grid=grid,
        samples=np.asarray(draws, dtype=float),
        log_posterior=np.asarray(draw_logp, dtype=float),
        map=best,
        provenance={
            "method": "mala",
            "small_reference": True,
            "burn_in": burn_in,
            "thin": thin,
            "step_size": float(step_size),
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


def hmc_field_invert(
    grid: Field3D,
    observations: list[Observation],
    registry: ForwardOperatorRegistry,
    prior: FieldGaussianPrior,
    *,
    n_samples: int,
    step_size: float,
    n_leapfrog: int = 10,
    burn_in: int = 1000,
    thin: int = 1,
    initial_unconstrained: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[PosteriorFieldSamples3D, MCMCReport]:
    """Sample a field posterior with Hamiltonian Monte Carlo, mass-preconditioned by the prior precision.

    Momentum is drawn from ``N(0, prior.precision(grid))`` (the mass matrix), so its natural scale
    matches the prior's; ``n_leapfrog`` leapfrog steps of size ``step_size`` simulate the Hamiltonian
    dynamics using :func:`field_log_posterior_grad_kernel`. A fresh momentum draw every iteration
    occasionally carries enough kinetic energy for the (near-)deterministic leapfrog trajectory to cross
    a likelihood barrier that an isotropic random-walk step essentially never crosses -- this is what
    lets HMC discover a second well-separated posterior mode :func:`metropolis_field_invert` misses.
    """
    if not observations:
        raise ValueError("need at least one observation to invert.")
    n_samples = int(n_samples)
    burn_in = int(burn_in)
    thin = int(thin)
    n_leapfrog = int(n_leapfrog)
    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")
    if burn_in < 0:
        raise ValueError("burn_in must be non-negative.")
    if thin <= 0:
        raise ValueError("thin must be positive.")
    if step_size <= 0.0:
        raise ValueError("step_size must be positive.")
    if n_leapfrog <= 0:
        raise ValueError("n_leapfrog must be positive.")

    rng = np.random.default_rng() if rng is None else rng
    n = grid.n
    prior_mean, precision = _prior_precision_and_mean(grid, prior)
    mass_chol = np.linalg.cholesky(precision)  # mass matrix M = prior precision
    mass_cov = np.linalg.inv(precision)  # M^-1, needed for the position update and kinetic energy

    if initial_unconstrained is None:
        current = prior_mean.copy()
    else:
        current = np.asarray(initial_unconstrained, dtype=float)
        if current.shape != (n,):
            raise ValueError(f"initial_unconstrained must have shape ({n},).")
        current = current.copy()

    def logp(u: np.ndarray) -> float:
        return field_log_posterior_kernel(
            grid, observations, registry, prior, u, prior_mean=prior_mean, prior_precision=precision
        )

    def grad(u: np.ndarray) -> np.ndarray:
        return field_log_posterior_grad_kernel(
            grid, observations, registry, prior, u, prior_mean=prior_mean, prior_precision=precision
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
        p0 = mass_chol @ rng.standard_normal(n)
        u = current.copy()
        p = p0.copy()
        g = grad(u)
        p = p + 0.5 * step_size * g
        for leap in range(n_leapfrog):
            u = u + step_size * (mass_cov @ p)
            if not np.all(np.isfinite(u)):
                break
            g = grad(u)
            p = p + (step_size if leap != n_leapfrog - 1 else 0.5 * step_size) * g
        proposal_logp = logp(u) if np.all(np.isfinite(u)) else -np.inf
        accept = False
        if np.isfinite(proposal_logp):
            kinetic0 = 0.5 * float(p0 @ (mass_cov @ p0))
            kinetic1 = 0.5 * float(p @ (mass_cov @ p))
            log_alpha = (proposal_logp - kinetic1) - (current_logp - kinetic0)
            accept = np.isfinite(log_alpha) and np.log(rng.random()) < log_alpha
        if accept:
            current = u
            current_logp = float(proposal_logp)
            accepted += 1
            if current_logp > best_logp:
                best = current.copy()
                best_logp = current_logp
        if step > burn_in and (step - burn_in) % thin == 0:
            draws.append(current.copy())
            draw_logp.append(float(current_logp))

    posterior = PosteriorFieldSamples3D(
        grid=grid,
        samples=np.asarray(draws, dtype=float),
        log_posterior=np.asarray(draw_logp, dtype=float),
        map=best,
        provenance={
            "method": "hmc",
            "small_reference": True,
            "burn_in": burn_in,
            "thin": thin,
            "step_size": float(step_size),
            "n_leapfrog": n_leapfrog,
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
