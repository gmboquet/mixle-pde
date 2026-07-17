"""Mean-field variational inference for latent 3D fields (MP-I5).

``mixle_pde``'s Bayesian field-inference ladder (:mod:`mixle_pde.field_inversion`'s exact
linear-Gaussian solve, :mod:`mixle_pde.field_gauss_newton`'s MAP/Laplace, :mod:`mixle_pde.field_mcmc`'s
Random-Walk Metropolis/pCN/MALA/HMC) had no variational route despite one being mentioned as a future
"production ladder" rung in :mod:`mixle_pde.field_mcmc`'s own module docstring -- an aspiration with no
code behind it. This module is that code: it optimizes a **mean-field (diagonal-covariance) Gaussian**
approximation ``q(u) = N(mu, diag(sigma**2))`` against exactly the same unnormalized log-posterior
:func:`mixle_pde.field_mcmc.field_log_posterior_kernel` and gradient
:func:`mixle_pde.field_mcmc.field_log_posterior_grad_kernel` the gradient-based MCMC samplers
(``mala_field_invert``, ``hmc_field_invert``) already consume -- the same prior/likelihood interface,
not a separate one, so any registered observation family (linear or nonlinear, as long as it declares a
Jacobian) that already works with those samplers works here unchanged.

Method
------
The evidence lower bound ``ELBO(mu, log_sigma) = E_q[log p(u, D)] + H[q]`` is estimated by Monte Carlo
using the reparameterization trick: draw ``eps ~ N(0, I)``, set ``u = mu + sigma * eps`` (``sigma =
exp(log_sigma)``, keeping sigma positive without a constrained optimizer), and average
``field_log_posterior_kernel(u)`` over a small batch of draws. Because ``u`` is a differentiable
function of ``(mu, log_sigma)``, the pathwise gradient of that Monte Carlo estimate is obtained by
chaining :func:`~mixle_pde.field_mcmc.field_log_posterior_grad_kernel`'s exact analytic gradient
through the reparameterization (chain rule), exactly mirroring how ``mala_field_invert``/
``hmc_field_invert`` reuse that same gradient kernel rather than introducing a new autodiff engine --
there is still no reverse-mode autodiff engine in ``mixle_pde`` (see that kernel's own docstring), and
none is added here. The mean-field entropy ``H[q] = sum(log_sigma) + (n/2)(1 + log(2*pi))`` contributes
the constant ``+1`` gradient term per dimension w.r.t. ``log_sigma``. Both parameters are updated by
Adam (Kingma & Ba, 2015) stochastic gradient **ascent**; the returned fit is the tail-average of the
last ``tail_fraction`` of iterates (Polyak-Ruppert averaging), which damps the residual Monte Carlo
noise a single final iterate would otherwise carry.

Honest characterization: mean-field is a biased approximation, not a small-sample artifact
------------------------------------------------------------------------------------------
Restricting ``q`` to a diagonal covariance is exact for the posterior **mean** of a Gaussian target
(the mean update does not depend on the assumed covariance structure) but is a real, well-known
approximation for the **covariance**: whenever the true posterior has correlation between cells (any
nonzero :attr:`~mixle_pde.field_inversion.FieldGaussianPrior.smoothness_precision`, or overlapping
observations, couples neighbours), the KL-optimal diagonal covariance has marginal variances equal to
the *conditional* variances ``1 / diag(precision)``, which are always ``<=`` the *true* marginal
variances ``diag(precision^-1)`` -- with equality iff the posterior is already uncorrelated. Mean-field
VI therefore systematically **underestimates posterior uncertainty** in correlated directions; it never
overestimates it. ``tests/field_vi_test.py`` verifies this quantitatively against a known closed-form
posterior rather than asserting only an easy, uncorrelated case that would hide it -- see
``MeanFieldVICorrelatedBiasTest`` there.

This is therefore an explicitly **approximate** engine, in the same "small/medium-scale reference path"
spirit as :mod:`mixle_pde.field_mcmc`'s samplers, not a replacement for the exact linear-Gaussian or
Laplace routes when those apply -- its job is fast, gradient-based posterior approximation when a full
MCMC run is too slow and independent marginals are an acceptable price, with the bias above disclosed
rather than hidden.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mixle_pde.field_inversion import FieldGaussianPrior
from mixle_pde.field_mcmc import (
    _prior_precision_and_mean,
    field_log_posterior_grad_kernel,
    field_log_posterior_kernel,
)
from mixle_pde.latent import Field3D, PosteriorField3D
from mixle_pde.observations import ForwardOperatorRegistry, Observation

#: log_sigma is clipped to this range after every Adam step -- numerical hygiene only (guards against a
#: runaway step blowing sigma up/down to overflow/underflow), not part of the statistical model. It
#: corresponds to sigma in roughly [3.4e-4, 2981], far outside any range a well-scaled problem needs.
_LOG_SIGMA_CLIP = 8.0


@dataclass(frozen=True)
class VIReport:
    """Optimization diagnostics for a mean-field variational fit.

    ``elbo_history`` is the per-iteration Monte Carlo ELBO estimate (noisy by construction -- it is a
    stochastic estimate, not the exact ELBO at every step); ``final_elbo``/``best_elbo`` are its last
    and maximum entries. ``converged`` compares the mean ELBO of the last tenth of iterations against
    the tenth before it, using their own combined standard error as the threshold (a two-sample
    z-style check, self-scaling with however noisy ``n_mc_samples`` makes a single iteration, rather
    than a fixed relative tolerance that would be too strict at small ``n_mc_samples`` and too loose at
    large ``n_mc_samples``) -- a coarse, honest signal that the objective has stopped drifting beyond
    its own noise floor, not a proof of having reached the global optimum of a nonconvex ELBO.
    """

    iterations: int
    n_mc_samples: int
    learning_rate: float
    final_elbo: float
    best_elbo: float
    elbo_history: tuple[float, ...]
    converged: bool


def _prior_log_normalizer(prior_precision: np.ndarray, n: int) -> float:
    """The additive Gaussian-prior normalizing constant :func:`~mixle_pde.field_mcmc.field_log_posterior_kernel`
    omits (it scores only ``-0.5 * delta @ precision @ delta``, which is all a Metropolis ratio or a
    gradient ever needs -- any additive, ``u``-independent constant cancels or vanishes there). The
    ELBO is not shift-invariant the same way: ``E_q[log p(u, D)] + H[q]`` is only a genuine lower bound
    on ``log p(D)`` when ``log p(u, D)`` is the fully normalized joint density. This restores the
    missing ``+0.5 * log|precision| - 0.5 * n * log(2*pi)`` term once, from the fixed prior precision,
    so this module's reported ELBO stays a mathematically meaningful bound without modifying the shared
    kernel every other sampler in :mod:`mixle_pde.field_mcmc` already relies on being unnormalized-cheap.
    """
    _, logdet = np.linalg.slogdet(prior_precision)
    return float(0.5 * logdet - 0.5 * n * np.log(2.0 * np.pi))


def _initial_log_sigma_vector(initial_log_sigma: float | np.ndarray, n: int) -> np.ndarray:
    value = np.asarray(initial_log_sigma, dtype=float)
    if value.ndim == 0:
        return np.full(n, float(value))
    if value.shape != (n,):
        raise ValueError(f"initial_log_sigma must be a scalar or have shape ({n},).")
    return value.copy()


def _reparam_elbo_and_grad(
    grid: Field3D,
    observations: list[Observation],
    registry: ForwardOperatorRegistry,
    prior: FieldGaussianPrior,
    mu: np.ndarray,
    log_sigma: np.ndarray,
    eps: np.ndarray,
    *,
    prior_mean: np.ndarray,
    prior_precision: np.ndarray,
    prior_log_normalizer: float,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Monte Carlo ELBO and its gradient w.r.t. ``(mu, log_sigma)`` via the reparameterization trick.

    ``eps`` is an ``(n_mc, n)`` batch of standard-normal draws; each row reparameterizes one posterior
    draw ``u = mu + sigma * eps``. Reuses :func:`~mixle_pde.field_mcmc.field_log_posterior_kernel` and
    :func:`~mixle_pde.field_mcmc.field_log_posterior_grad_kernel` -- the same analytic
    gradient-through-the-registry chain ``mala_field_invert``/``hmc_field_invert`` already use -- per
    draw, so no additional autodiff machinery is introduced. ``prior_log_normalizer`` (see
    :func:`_prior_log_normalizer`) restores the Gaussian-prior normalizing constant the shared kernel
    omits, so the returned ``elbo`` is the true, fully-normalized evidence lower bound (it never affects
    the returned gradients -- a ``u``-independent constant has zero gradient).
    """
    sigma = np.exp(log_sigma)
    n_mc, n = eps.shape
    grad_mu = np.zeros(n, dtype=float)
    grad_log_sigma = np.zeros(n, dtype=float)
    log_joint_sum = 0.0
    for s in range(n_mc):
        u = mu + sigma * eps[s]
        log_joint_sum += field_log_posterior_kernel(
            grid, observations, registry, prior, u, prior_mean=prior_mean, prior_precision=prior_precision
        )
        g = field_log_posterior_grad_kernel(
            grid, observations, registry, prior, u, prior_mean=prior_mean, prior_precision=prior_precision
        )
        grad_mu += g
        grad_log_sigma += g * eps[s] * sigma
    grad_mu /= n_mc
    grad_log_sigma /= n_mc
    grad_log_sigma += 1.0  # d/d(log_sigma) of the mean-field entropy sum(log_sigma) + const
    entropy = float(np.sum(log_sigma) + 0.5 * n * (1.0 + np.log(2.0 * np.pi)))
    elbo = log_joint_sum / n_mc + entropy + prior_log_normalizer
    return elbo, grad_mu, grad_log_sigma


def mean_field_vi_invert(
    grid: Field3D,
    observations: list[Observation],
    registry: ForwardOperatorRegistry,
    prior: FieldGaussianPrior,
    *,
    n_iterations: int,
    n_mc_samples: int = 8,
    learning_rate: float = 0.1,
    tail_fraction: float = 0.2,
    initial_unconstrained: np.ndarray | None = None,
    initial_log_sigma: float | np.ndarray = 0.0,
    convergence_sigma: float = 2.0,
    rng: np.random.Generator | None = None,
) -> tuple[PosteriorField3D, VIReport]:
    """Fit a mean-field Gaussian posterior approximation by stochastic ELBO ascent.

    Every observation's operator must expose a Jacobian (``op.has_adjoint()``), exactly like
    ``mala_field_invert``/``hmc_field_invert`` -- the reparameterization gradient chains through
    :func:`~mixle_pde.field_mcmc.field_log_posterior_grad_kernel`, which raises a clear ``ValueError``
    naming the offending observation kind otherwise. Returns a :class:`~mixle_pde.latent.PosteriorField3D`
    in ``diag_var`` (independent-marginals) mode -- the same posterior artifact class the exact and
    Laplace routes return, but carrying only marginal variances, never correlations, which is precisely
    the mean-field restriction (see the module docstring's honesty note).

    ``tail_fraction`` (default 0.2) Polyak-Ruppert-averages the last fraction of ``(mu, log_sigma)``
    iterates for the returned point estimate, damping residual Monte Carlo noise from the stochastic
    gradient rather than reporting a single, noisier final iterate. ``convergence_sigma`` (default 2.0)
    is the z-style threshold :class:`VIReport`'s ``converged`` flag uses when comparing the last tenth
    of iterations' mean ELBO against the tenth before it.
    """
    if not observations:
        raise ValueError("need at least one observation to invert.")
    n_iterations = int(n_iterations)
    n_mc_samples = int(n_mc_samples)
    if n_iterations <= 0:
        raise ValueError("n_iterations must be positive.")
    if n_mc_samples <= 0:
        raise ValueError("n_mc_samples must be positive.")
    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive.")
    if not 0.0 < tail_fraction <= 1.0:
        raise ValueError("tail_fraction must be in (0, 1].")
    if convergence_sigma <= 0.0:
        raise ValueError("convergence_sigma must be positive.")

    rng = np.random.default_rng() if rng is None else rng
    n = grid.n
    prior_mean, precision = _prior_precision_and_mean(grid, prior)
    prior_log_normalizer = _prior_log_normalizer(precision, n)

    if initial_unconstrained is None:
        mu = prior_mean.copy()
    else:
        mu = np.asarray(initial_unconstrained, dtype=float)
        if mu.shape != (n,):
            raise ValueError(f"initial_unconstrained must have shape ({n},).")
        mu = mu.copy()
    log_sigma = _initial_log_sigma_vector(initial_log_sigma, n)

    initial_logp = field_log_posterior_kernel(
        grid, observations, registry, prior, mu, prior_mean=prior_mean, prior_precision=precision
    )
    if not np.isfinite(initial_logp):
        raise ValueError("initial_unconstrained has non-finite log posterior.")

    beta1, beta2, adam_eps = 0.9, 0.999, 1.0e-8
    m_mu, v_mu = np.zeros(n), np.zeros(n)
    m_ls, v_ls = np.zeros(n), np.zeros(n)

    tail_window = max(1, int(round(n_iterations * tail_fraction)))
    tail_start = n_iterations - tail_window
    mu_tail_sum = np.zeros(n)
    log_sigma_tail_sum = np.zeros(n)
    tail_count = 0
    elbo_history: list[float] = []

    for t in range(1, n_iterations + 1):
        eps = rng.standard_normal((n_mc_samples, n))
        elbo, grad_mu, grad_log_sigma = _reparam_elbo_and_grad(
            grid,
            observations,
            registry,
            prior,
            mu,
            log_sigma,
            eps,
            prior_mean=prior_mean,
            prior_precision=precision,
            prior_log_normalizer=prior_log_normalizer,
        )
        elbo_history.append(elbo)

        m_mu = beta1 * m_mu + (1.0 - beta1) * grad_mu
        v_mu = beta2 * v_mu + (1.0 - beta2) * grad_mu**2
        mu_hat = m_mu / (1.0 - beta1**t)
        v_mu_hat = v_mu / (1.0 - beta2**t)
        mu = mu + learning_rate * mu_hat / (np.sqrt(v_mu_hat) + adam_eps)

        m_ls = beta1 * m_ls + (1.0 - beta1) * grad_log_sigma
        v_ls = beta2 * v_ls + (1.0 - beta2) * grad_log_sigma**2
        ls_hat = m_ls / (1.0 - beta1**t)
        v_ls_hat = v_ls / (1.0 - beta2**t)
        log_sigma = log_sigma + learning_rate * ls_hat / (np.sqrt(v_ls_hat) + adam_eps)
        log_sigma = np.clip(log_sigma, -_LOG_SIGMA_CLIP, _LOG_SIGMA_CLIP)

        if t > tail_start:
            mu_tail_sum += mu
            log_sigma_tail_sum += log_sigma
            tail_count += 1

    mu_final = mu_tail_sum / tail_count
    sigma_final = np.exp(log_sigma_tail_sum / tail_count)

    window = max(1, n_iterations // 10)
    converged = False
    if n_iterations >= 2 * window and window > 1:
        prev_block = np.asarray(elbo_history[-2 * window : -window])
        last_block = np.asarray(elbo_history[-window:])
        combined_se = np.sqrt(prev_block.var(ddof=1) / window + last_block.var(ddof=1) / window)
        if combined_se > 0.0:
            converged = bool(abs(last_block.mean() - prev_block.mean()) < convergence_sigma * combined_se)

    report = VIReport(
        iterations=n_iterations,
        n_mc_samples=n_mc_samples,
        learning_rate=learning_rate,
        final_elbo=float(elbo_history[-1]),
        best_elbo=float(max(elbo_history)),
        elbo_history=tuple(elbo_history),
        converged=converged,
    )
    posterior = PosteriorField3D(grid=grid, mean=mu_final, map=mu_final.copy(), diag_var=sigma_final**2)
    return posterior, report


def elbo_estimate(
    grid: Field3D,
    observations: list[Observation],
    registry: ForwardOperatorRegistry,
    prior: FieldGaussianPrior,
    posterior: PosteriorField3D,
    *,
    n_mc_samples: int = 1000,
    rng: np.random.Generator | None = None,
) -> float:
    """A fresh, high-sample-count Monte Carlo ELBO estimate at a fitted mean-field ``posterior`` (as
    returned by :func:`mean_field_vi_invert`), independent of any single noisy training-loop iteration
    recorded in :class:`VIReport`.elbo_history. Since the (fully-normalized) ELBO is a true lower bound
    on the log model evidence ``log p(D)``, this is the number to compare against a closed-form evidence
    when one is available -- see ``tests/field_vi_test.py``'s evidence-bound check.
    """
    if posterior.diag_var is None:
        raise ValueError(
            "elbo_estimate expects a mean-field posterior in diag_var mode, as mean_field_vi_invert returns."
        )
    rng = np.random.default_rng() if rng is None else rng
    n = grid.n
    prior_mean, precision = _prior_precision_and_mean(grid, prior)
    prior_log_normalizer = _prior_log_normalizer(precision, n)
    log_sigma = 0.5 * np.log(posterior.diag_var)
    eps = rng.standard_normal((int(n_mc_samples), n))
    elbo, _, _ = _reparam_elbo_and_grad(
        grid,
        observations,
        registry,
        prior,
        posterior.mean,
        log_sigma,
        eps,
        prior_mean=prior_mean,
        prior_precision=precision,
        prior_log_normalizer=prior_log_normalizer,
    )
    return elbo
