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
dimension-robust; :func:`mala_field_invert`, :func:`hmc_field_invert`, and :func:`nuts_field_invert` use
the gradient of :func:`field_log_posterior_kernel` (via each registered operator's Jacobian -- see
:func:`field_log_posterior_grad_kernel`) preconditioned by the prior precision. All four remain a
small/medium-scale reference path, not a replacement for the exact linear-Gaussian or Laplace routes;
their job is the multimodal / non-Gaussian fallback those exact paths cannot represent.

:func:`hmc_field_invert` simulates a *fixed* number of leapfrog steps (``n_leapfrog``) at a *fixed*,
caller-chosen ``step_size`` every iteration -- both have to be hand-tuned per problem, and a fixed
trajectory length either wastes gradient evaluations (too long: the trajectory u-turns and doubles
back before ``n_leapfrog`` is reached) or under-explores (too short). :func:`nuts_field_invert` is the
No-U-Turn Sampler (Hoffman & Gelman, 2014): it reuses exactly the same leapfrog integrator and
prior-precision mass matrix :func:`hmc_field_invert` already uses (both call the shared
:func:`_leapfrog_step` primitive), but replaces the fixed trajectory length with a recursive
binary-tree doubling procedure that stops itself the moment continuing would double back on its own
path (the "no-U-turn" criterion), and replaces the fixed step size with dual-averaging adaptation
during a warmup phase that automatically targets a caller-chosen acceptance probability.
"""

from __future__ import annotations

from collections.abc import Callable
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


def _leapfrog_step(
    u: np.ndarray,
    p: np.ndarray,
    grad_u: np.ndarray,
    step_size: float,
    mass_cov: np.ndarray,
    grad_fn: Callable[[np.ndarray], np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One leapfrog (Stormer-Verlet) step of Hamiltonian dynamics, shared by :func:`hmc_field_invert`
    and :func:`nuts_field_invert` -- the "same leapfrog-integrator/momentum machinery" both samplers
    build on, factored out once rather than reimplemented a second time.

    Half-kick / drift / half-kick: ``p_half = p + 0.5*step_size*grad_u``, ``u' = u + step_size*(mass_cov
    @ p_half)``, ``p' = p_half + 0.5*step_size*grad_fn(u')``. ``step_size`` may be negative -- NUTS
    integrates backward in "fictitious time" by negating it; leapfrog is time-reversible, so this is
    exact, not an approximation. Chaining calls (feeding each call's returned gradient back in as the
    next call's ``grad_u``) reproduces exactly the merged-interior-half-step trajectory a naive
    per-step "half-kick, drift, half-kick" loop would give, at one fresh gradient evaluation per call.

    If the drift lands on a non-finite position (outside the field's bound-transform domain, or an
    overflowing physical value), ``grad_fn`` is never evaluated there -- the caller sees a non-finite
    ``u'`` back and must treat the step as invalid rather than trust the returned momentum/gradient for
    anything beyond that check.
    """
    p_half = p + 0.5 * step_size * grad_u
    u_new = u + step_size * (mass_cov @ p_half)
    if not np.all(np.isfinite(u_new)):
        return u_new, p_half, grad_u
    grad_new = grad_fn(u_new)
    p_new = p_half + 0.5 * step_size * grad_new
    return u_new, p_new, grad_new


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
        for _leap in range(n_leapfrog):
            u, p, g = _leapfrog_step(u, p, g, step_size, mass_cov, grad)
            if not np.all(np.isfinite(u)):
                break
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


# ---------------------------------------------------------------------------------------------------
# No-U-Turn Sampler (Hoffman & Gelman, 2014, JMLR 15: "The No-U-Turn Sampler: Adaptively Setting Path
# Lengths in Hamiltonian Monte Carlo"). Implements the paper's efficient, recursive-doubling Algorithm 3
# (slice-sampling based trajectory construction with the no-U-turn stopping rule) combined with
# Algorithm 6 (dual-averaging step-size adaptation during warmup). Both reuse hmc_field_invert's own
# leapfrog integrator (:func:`_leapfrog_step`) and prior-precision mass matrix unchanged.
# ---------------------------------------------------------------------------------------------------

# Hoffman & Gelman section 3.2's divergence threshold: a subtree is abandoned (treated as divergent) once
# the joint log-density has fallen more than this far below its value at the trajectory's start.
_NUTS_DELTA_MAX = 1000.0
# Dual-averaging constants (Hoffman & Gelman, Algorithm 6 / section 3.2.1); not exposed as public
# tunables because the paper treats them as fixed algorithmic constants, not per-problem knobs.
_NUTS_GAMMA = 0.05
_NUTS_T0 = 10.0
_NUTS_KAPPA = 0.75


@dataclass(frozen=True)
class NUTSReport:
    """Diagnostics for a No-U-Turn Sampler field posterior sample (Hoffman & Gelman, 2014).

    Unlike :class:`MCMCReport` (fixed proposal count, single acceptance rate), NUTS's cost and
    behavior per iteration are themselves adaptive, so this reports the adaptation outcome directly:
    the dual-averaged ``step_size`` frozen at the end of ``warmup``, how deep the trajectory tree
    typically grew before self-terminating, how often it was truncated by ``max_tree_depth`` rather
    than stopping itself, how many sampling-phase transitions were flagged divergent, and the total
    number of gradient evaluations spent (warmup and sampling combined) -- the natural cost unit for
    comparing NUTS's sampling efficiency against :func:`hmc_field_invert`'s fixed ``n_leapfrog``.
    """

    iterations: int
    warmup: int
    thin: int
    stored_samples: int
    final_log_posterior: float
    best_log_posterior: float
    step_size: float
    target_accept: float
    mean_accept_stat: float
    mean_tree_depth: float
    max_tree_depth: int
    max_tree_depth_hits: int
    n_divergences: int
    gradient_evaluations: int


def _no_u_turn(
    u_plus: np.ndarray, u_minus: np.ndarray, p_plus: np.ndarray, p_minus: np.ndarray, mass_cov: np.ndarray
) -> bool:
    """Generalized no-U-turn criterion: ``True`` while the trajectory should keep extending.

    Hoffman & Gelman's original criterion (their section 3, identity-mass-matrix case) dots the
    endpoint separation directly against each endpoint's momentum. With a non-identity mass matrix --
    :func:`hmc_field_invert` and this sampler both precondition by the prior precision -- the
    physically meaningful quantity is *velocity* (``mass_cov @ p``, since Hamilton's equation gives
    ``d(position)/dt = mass_cov @ momentum``), not raw momentum; this is the same generalization
    Stan's NUTS implementation uses for a non-identity mass matrix. A U-turn is detected (return
    ``False``) the moment either endpoint's velocity now points back toward the other endpoint.
    """
    delta = u_plus - u_minus
    return bool(delta @ (mass_cov @ p_minus) >= 0.0 and delta @ (mass_cov @ p_plus) >= 0.0)


@dataclass
class _BuildTreeResult:
    """One :func:`_build_tree` call's output: the new subtree's two endpoints, its proposal, and the
    bookkeeping (candidate count, continue/stop flag, dual-averaging accept statistic, divergence flag,
    and gradient-evaluation count) Algorithm 3/6 thread through the recursion."""

    u_minus: np.ndarray
    p_minus: np.ndarray
    grad_minus: np.ndarray
    u_plus: np.ndarray
    p_plus: np.ndarray
    grad_plus: np.ndarray
    u_prime: np.ndarray
    grad_prime: np.ndarray
    logp_prime: float
    n_prime: int
    keep_going: bool
    alpha: float
    n_alpha: int
    gradient_evals: int
    diverged: bool


def _build_tree(
    u: np.ndarray,
    p: np.ndarray,
    grad_u: np.ndarray,
    log_u: float,
    v: int,
    depth: int,
    step_size: float,
    mass_cov: np.ndarray,
    grad_fn: Callable[[np.ndarray], np.ndarray],
    logp_fn: Callable[[np.ndarray], float],
    joint_logp0: float,
    rng: np.random.Generator,
) -> _BuildTreeResult:
    """Recursive binary-tree trajectory builder (Hoffman & Gelman 2014, Algorithm 3 extended by
    Algorithm 6's ``alpha``/``n_alpha`` accept-statistic bookkeeping).

    ``(u, p, grad_u)`` is the current boundary being extended one further leapfrog step (``depth==0``)
    or one further doubling (``depth>0``) in direction ``v`` (``+1``/``-1``). ``log_u`` is the
    log-space slice threshold sampled once per top-level NUTS transition. Returns the new subtree
    spanning ``2**depth`` leapfrog steps.
    """
    if depth == 0:
        u2, p2, g2 = _leapfrog_step(u, p, grad_u, v * step_size, mass_cov, grad_fn)
        valid_state = bool(np.all(np.isfinite(u2)) and np.all(np.isfinite(p2)))
        if valid_state:
            logp2 = logp_fn(u2)
            valid_state = bool(np.isfinite(logp2))
        if valid_state:
            kinetic2 = 0.5 * float(p2 @ (mass_cov @ p2))
            joint2 = logp2 - kinetic2
        else:
            logp2 = -np.inf
            joint2 = -np.inf
        n_prime = 1 if np.isfinite(joint2) and log_u <= joint2 else 0
        diverged = not (np.isfinite(joint2) and log_u < _NUTS_DELTA_MAX + joint2)
        alpha = float(min(1.0, np.exp(joint2 - joint_logp0))) if np.isfinite(joint2) else 0.0
        return _BuildTreeResult(
            u_minus=u2,
            p_minus=p2,
            grad_minus=g2,
            u_plus=u2,
            p_plus=p2,
            grad_plus=g2,
            u_prime=u2,
            grad_prime=g2,
            logp_prime=logp2,
            n_prime=n_prime,
            keep_going=not diverged,
            alpha=alpha,
            n_alpha=1,
            gradient_evals=1,
            diverged=diverged,
        )

    # Build the first half (extending the current boundary), then -- only if it is still valid -- the
    # second half (extending the *new* boundary the first half produced); a first half that already
    # diverged or U-turned is never extended further, which is what keeps NUTS's average cost per
    # transition well below the worst-case 2**max_tree_depth leapfrog steps.
    first = _build_tree(u, p, grad_u, log_u, v, depth - 1, step_size, mass_cov, grad_fn, logp_fn, joint_logp0, rng)
    u_minus, p_minus, grad_minus = first.u_minus, first.p_minus, first.grad_minus
    u_plus, p_plus, grad_plus = first.u_plus, first.p_plus, first.grad_plus
    u_prime, grad_prime, logp_prime = first.u_prime, first.grad_prime, first.logp_prime
    n_prime, alpha, n_alpha, gradient_evals = first.n_prime, first.alpha, first.n_alpha, first.gradient_evals
    keep_going = first.keep_going
    diverged = first.diverged

    if keep_going:
        if v == -1:
            second = _build_tree(
                u_minus,
                p_minus,
                grad_minus,
                log_u,
                v,
                depth - 1,
                step_size,
                mass_cov,
                grad_fn,
                logp_fn,
                joint_logp0,
                rng,
            )
            u_minus, p_minus, grad_minus = second.u_minus, second.p_minus, second.grad_minus
        else:
            second = _build_tree(
                u_plus, p_plus, grad_plus, log_u, v, depth - 1, step_size, mass_cov, grad_fn, logp_fn, joint_logp0, rng
            )
            u_plus, p_plus, grad_plus = second.u_plus, second.p_plus, second.grad_plus

        total_n = n_prime + second.n_prime
        if total_n > 0 and rng.random() < (second.n_prime / total_n):
            u_prime, grad_prime, logp_prime = second.u_prime, second.grad_prime, second.logp_prime
        alpha += second.alpha
        n_alpha += second.n_alpha
        gradient_evals += second.gradient_evals
        diverged = diverged or second.diverged
        keep_going = second.keep_going and _no_u_turn(u_plus, u_minus, p_plus, p_minus, mass_cov)
        n_prime = total_n

    return _BuildTreeResult(
        u_minus=u_minus,
        p_minus=p_minus,
        grad_minus=grad_minus,
        u_plus=u_plus,
        p_plus=p_plus,
        grad_plus=grad_plus,
        u_prime=u_prime,
        grad_prime=grad_prime,
        logp_prime=logp_prime,
        n_prime=n_prime,
        keep_going=keep_going,
        alpha=alpha,
        n_alpha=n_alpha,
        gradient_evals=gradient_evals,
        diverged=diverged,
    )


def _find_reasonable_step_size(
    u0: np.ndarray,
    grad_u0: np.ndarray,
    logp0: float,
    mass_chol: np.ndarray,
    mass_cov: np.ndarray,
    grad_fn: Callable[[np.ndarray], np.ndarray],
    logp_fn: Callable[[np.ndarray], float],
    rng: np.random.Generator,
) -> tuple[float, int]:
    """Heuristic initial step size (Hoffman & Gelman 2014, Algorithm 4): double or halve ``epsilon``
    starting from 1 until one leapfrog step's acceptance probability crosses one half.

    This only seeds dual averaging (:func:`nuts_field_invert`'s warmup loop corrects it from there);
    it does not need to be exact, only the right order of magnitude, so a bounded iteration count
    (rather than an unbounded ``while``) is a safe, deliberate deviation from the paper's literal
    pseudocode for a pathological posterior where the ratio never crosses one half.
    """
    n = u0.shape[0]
    step_size = 1.0
    r0 = mass_chol @ rng.standard_normal(n)
    kinetic0 = 0.5 * float(r0 @ (mass_cov @ r0))
    joint0 = logp0 - kinetic0
    gradient_evals = 0

    def joint_after_step(eps: float) -> float:
        nonlocal gradient_evals
        u1, p1, _ = _leapfrog_step(u0, r0, grad_u0, eps, mass_cov, grad_fn)
        gradient_evals += 1
        if not (np.all(np.isfinite(u1)) and np.all(np.isfinite(p1))):
            return -np.inf
        logp1 = logp_fn(u1)
        if not np.isfinite(logp1):
            return -np.inf
        kinetic1 = 0.5 * float(p1 @ (mass_cov @ p1))
        return logp1 - kinetic1

    log_accept = joint_after_step(step_size) - joint0
    direction = 1.0 if log_accept > np.log(0.5) else -1.0
    for _ in range(100):
        if direction * log_accept <= -direction * np.log(2.0):
            break
        step_size *= 2.0**direction
        new_log_accept = joint_after_step(step_size) - joint0
        if not np.isfinite(new_log_accept):
            step_size *= 2.0 ** (-direction)  # back off: the doubled/halved step landed somewhere invalid
            break
        log_accept = new_log_accept
    return step_size, gradient_evals


def nuts_field_invert(
    grid: Field3D,
    observations: list[Observation],
    registry: ForwardOperatorRegistry,
    prior: FieldGaussianPrior,
    *,
    n_samples: int,
    warmup: int = 1000,
    thin: int = 1,
    target_accept: float = 0.8,
    max_tree_depth: int = 10,
    initial_step_size: float | None = None,
    initial_unconstrained: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[PosteriorFieldSamples3D, NUTSReport]:
    """Sample a field posterior with the No-U-Turn Sampler, mass-preconditioned by the prior precision.

    Reuses exactly the machinery :func:`hmc_field_invert` already uses -- the same
    :func:`field_log_posterior_grad_kernel` gradient, the same ``N(0, prior.precision(grid))`` momentum,
    the same shared :func:`_leapfrog_step` integrator -- so every observation used here must expose a
    Jacobian, same as MALA/HMC. What replaces HMC's fixed ``(step_size, n_leapfrog)`` pair:

    * **Trajectory length**: each iteration grows a binary tree of leapfrog steps by repeated doubling
      (:func:`_build_tree`) in a randomly chosen direction, and stops the moment the tree's two
      endpoints' velocities imply continuing would double back on the path already taken (the
      no-U-turn criterion, :func:`_no_u_turn`) -- capped at ``2**max_tree_depth`` steps as a safety
      bound, not the typical stopping mechanism. The next sample is drawn from the visited states with
      probability proportional to how many of them were still inside the slice-sampling threshold
      (Neal's slice trick), which is what makes this an exact MCMC transition rather than a heuristic.
    * **Step size**: during the first ``warmup`` iterations, dual averaging (Algorithm 6) adjusts
      ``step_size`` after every iteration to drive the trajectory's average acceptance statistic toward
      ``target_accept``; the value is then frozen for the remaining sampling iterations. Pass
      ``initial_step_size`` to skip the Algorithm-4 bootstrap heuristic and seed dual averaging directly.

    Raises :class:`ValueError` for the same argument problems as the other samplers in this module,
    plus a non-finite log posterior at ``initial_unconstrained`` and an out-of-range ``target_accept``.
    """
    if not observations:
        raise ValueError("need at least one observation to invert.")
    n_samples = int(n_samples)
    warmup = int(warmup)
    thin = int(thin)
    max_tree_depth = int(max_tree_depth)
    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")
    if warmup < 0:
        raise ValueError("warmup must be non-negative.")
    if thin <= 0:
        raise ValueError("thin must be positive.")
    if not 0.0 < target_accept < 1.0:
        raise ValueError("target_accept must be in (0, 1).")
    if max_tree_depth <= 0:
        raise ValueError("max_tree_depth must be positive.")
    if initial_step_size is not None and initial_step_size <= 0.0:
        raise ValueError("initial_step_size must be positive.")

    rng = np.random.default_rng() if rng is None else rng
    n = grid.n
    prior_mean, precision = _prior_precision_and_mean(grid, prior)
    mass_chol = np.linalg.cholesky(precision)  # mass matrix M = prior precision
    mass_cov = np.linalg.inv(precision)  # M^-1, needed for the position update, kinetic energy, and velocity

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
    gradient_evals_total = 1  # the initial gradient evaluation just above, needed to seed the first leapfrog step

    if initial_step_size is None:
        step_size, find_eps_evals = _find_reasonable_step_size(
            current, current_grad, current_logp, mass_chol, mass_cov, grad, logp, rng
        )
        gradient_evals_total += find_eps_evals
    else:
        step_size = float(initial_step_size)

    mu = float(np.log(10.0 * step_size))
    h_bar = 0.0
    log_eps_bar = 0.0

    total_iterations = warmup + n_samples * thin
    best = current.copy()
    best_logp = float(current_logp)
    draws: list[np.ndarray] = []
    draw_logp: list[float] = []
    tree_depth_sum = 0
    max_depth_hits = 0
    n_divergences = 0
    accept_stat_sum = 0.0
    sampling_iterations = 0

    for m in range(1, total_iterations + 1):
        r0 = mass_chol @ rng.standard_normal(n)
        kinetic0 = 0.5 * float(r0 @ (mass_cov @ r0))
        joint0 = current_logp - kinetic0
        log_u = joint0 - rng.exponential()

        u_minus = u_plus = current
        p_minus = p_plus = r0
        grad_minus = grad_plus = current_grad
        u_sample, grad_sample, logp_sample = current, current_grad, current_logp
        n_valid = 1
        keep_going = True
        depth = 0
        alpha_sum = 0.0
        alpha_count = 0
        any_divergence = False

        while keep_going and depth < max_tree_depth:
            direction = -1 if rng.random() < 0.5 else 1
            if direction == -1:
                result = _build_tree(
                    u_minus, p_minus, grad_minus, log_u, direction, depth, step_size, mass_cov, grad, logp, joint0, rng
                )
                u_minus, p_minus, grad_minus = result.u_minus, result.p_minus, result.grad_minus
            else:
                result = _build_tree(
                    u_plus, p_plus, grad_plus, log_u, direction, depth, step_size, mass_cov, grad, logp, joint0, rng
                )
                u_plus, p_plus, grad_plus = result.u_plus, result.p_plus, result.grad_plus

            alpha_sum += result.alpha
            alpha_count += result.n_alpha
            gradient_evals_total += result.gradient_evals
            any_divergence = any_divergence or result.diverged

            if result.keep_going and result.n_prime > 0:
                accept_prob = min(1.0, result.n_prime / n_valid)
                if rng.random() < accept_prob:
                    u_sample, grad_sample, logp_sample = result.u_prime, result.grad_prime, result.logp_prime

            n_valid += result.n_prime
            keep_going = result.keep_going and _no_u_turn(u_plus, u_minus, p_plus, p_minus, mass_cov)
            depth += 1

        hit_max_depth = keep_going  # loop only exited on the depth cap, not a natural stop
        mean_accept_stat = alpha_sum / alpha_count if alpha_count > 0 else 0.0

        current, current_grad, current_logp = u_sample, grad_sample, logp_sample
        if current_logp > best_logp:
            best = current.copy()
            best_logp = current_logp

        if m <= warmup:
            eta_h = 1.0 / (m + _NUTS_T0)
            h_bar = (1.0 - eta_h) * h_bar + eta_h * (target_accept - mean_accept_stat)
            log_eps = mu - (np.sqrt(m) / _NUTS_GAMMA) * h_bar
            eta = m ** (-_NUTS_KAPPA)
            log_eps_bar = eta * log_eps + (1.0 - eta) * log_eps_bar
            step_size = float(np.exp(log_eps))
            if m == warmup:
                step_size = float(np.exp(log_eps_bar))  # freeze at the smoothed value from here on

        if m > warmup:
            tree_depth_sum += depth
            max_depth_hits += int(hit_max_depth)
            n_divergences += int(any_divergence)
            accept_stat_sum += mean_accept_stat
            sampling_iterations += 1
            if (m - warmup) % thin == 0:
                draws.append(current.copy())
                draw_logp.append(float(current_logp))

    posterior = PosteriorFieldSamples3D(
        grid=grid,
        samples=np.asarray(draws, dtype=float),
        log_posterior=np.asarray(draw_logp, dtype=float),
        map=best,
        provenance={
            "method": "nuts",
            "small_reference": True,
            "warmup": warmup,
            "thin": thin,
            "step_size": float(step_size),
            "target_accept": float(target_accept),
            "max_tree_depth": max_tree_depth,
        },
    )
    report = NUTSReport(
        iterations=total_iterations,
        warmup=warmup,
        thin=thin,
        stored_samples=len(draws),
        final_log_posterior=float(current_logp),
        best_log_posterior=best_logp,
        step_size=step_size,
        target_accept=target_accept,
        mean_accept_stat=(accept_stat_sum / sampling_iterations if sampling_iterations else 0.0),
        mean_tree_depth=(tree_depth_sum / sampling_iterations if sampling_iterations else 0.0),
        max_tree_depth=max_tree_depth,
        max_tree_depth_hits=max_depth_hits,
        n_divergences=n_divergences,
        gradient_evaluations=gradient_evals_total,
    )
    return posterior, report
