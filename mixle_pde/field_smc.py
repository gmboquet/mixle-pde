"""Sequential Monte Carlo with tempering (SMC-tempering) for latent 3D field posteriors (MP-I5).

``docs/reconciliation/mp-task-ledger.md``'s MP-I5 entry ("Inference engines and capability
negotiation") names :mod:`mixle_pde.field_mcmc` (Metropolis, pCN, MALA, HMC -- all single-chain) and
:func:`mixle_pde.field_assimilation.particle_assimilate_4d` (a particle filter over the TIME axis) as
the existing inference engines, and records that "no HMC/NUTS, SMC-tempering, or VI engine [was] found
anywhere". This module closes the SMC-tempering slice of that gap.

Why a population method earns its own module when :mod:`mixle_pde.field_mcmc` already has four
single-chain samplers: pCN and HMC both look for a second, well-separated posterior mode by hoping one
proposal (a fresh prior-scaled draw, or a long deterministic leapfrog trajectory) is enough to cross a
low-likelihood gap in one jump from a SINGLE chain. That is a gamble that gets worse as the gap between
modes widens. SMC-tempering does not gamble: it walks a whole POPULATION of particles across a sequence
of intermediate distributions

    pi_temperature(u) propto prior(u) * likelihood(u) ** temperature,      temperature: 0 -> 1

that geometrically bridges the (easy to sample exactly) prior at ``temperature=0`` to the true posterior
at ``temperature=1``. At small temperature, the likelihood barrier between modes is raised to a small
power and is shallow -- easy to cross; the population only has to track the barrier's SHAPE as it
sharpens on the way to ``temperature=1``, never having to leap it in one move the way a single chain
does. :func:`smc_tempering_field_invert` proves this experimentally against a genuinely multimodal
posterior in ``tests/field_smc_test.py``: it recovers both modes with occupancy fractions matching the
true (quadrature-computed) relative mode weights, while :func:`~mixle_pde.field_mcmc.metropolis_field_invert`
started from the same problem collapses onto one side, exactly the failure mode
``mixle_pde/c5_sampler_test.py``'s ``BimodalPosteriorSamplerTest`` already documents for plain
Metropolis. SMC also reports a log-evidence estimate as a free byproduct of the reweighting recursion
(see :attr:`SMCReport.log_evidence`) -- something none of the single-chain samplers can produce at all.

Reused conventions (per this module's own design brief: reuse this repo's established particle/ensemble
and checkpoint patterns rather than inventing parallel ones)
--------------------------------------------------------------------------------------------------------
* Particle population, importance weights, effective sample size, and systematic resampling all reuse
  :mod:`mixle_pde.field_assimilation`'s own machinery byte-for-byte: :func:`~mixle_pde.field_assimilation.
  _logsumexp` and :func:`~mixle_pde.field_assimilation._systematic_resample` are imported directly (the
  same cross-module reuse :mod:`mixle_pde.sample_update` already does for the same two helpers), and the
  effective-sample-size formula ``1 / sum(weight**2)`` on self-normalized weights is exactly
  :func:`~mixle_pde.field_assimilation.particle_assimilate_4d`'s.
* The rejuvenation (post-resample MCMC move) kernel is the SAME preconditioned Crank-Nicolson
  construction :func:`~mixle_pde.field_mcmc.pcn_field_invert` and :func:`~mixle_pde.field_assimilation.
  _pcn_rejuvenate_particles` already use -- prior-reversible, so its Metropolis-Hastings ratio needs no
  Jacobian and works for any registered forward operator, linear or not (this repo's own docstring for
  pCN already calls it "the recommended multimodal fallback"). The only generalization is a
  ``temperature`` exponent multiplying the log-likelihood delta in the acceptance test (see
  :func:`_tempered_pcn_move`'s docstring for the one-line derivation of why that exact exponent, and
  nothing else, is what turns pCN's prior-invariance into invariance w.r.t. the CURRENT tempered target).
* Diagnostics reuse the field's unconstrained-space prior log-density kernel
  (:func:`mixle_pde.field_mcmc._prior_log_density_kernel`) and precision/mean helper
  (:func:`mixle_pde.field_mcmc._prior_precision_and_mean`) so the returned ``log_posterior`` is the exact
  same unnormalized quantity :func:`~mixle_pde.field_mcmc.metropolis_field_invert` et al. report.
* The public return shape is the established ``(PosteriorFieldSamples3D, <Report>)`` pair every sampler
  in :mod:`mixle_pde.field_mcmc` returns; the final temperature rung always resamples (mirroring
  :func:`~mixle_pde.field_assimilation.particle_assimilate_4d`'s forced resample at its last time index)
  so the returned samples are equally weighted and compatible with :class:`~mixle_pde.latent.
  PosteriorFieldSamples3D`'s existing (unweighted) query surface -- there is no separate "weighted
  posterior" artifact to maintain.

Temperature schedule
---------------------
Three modes, chosen by ``adaptive``/``temperature_schedule``:

* ``temperature_schedule`` supplied explicitly -- used verbatim (must start at ``0.0``, end at ``1.0``,
  strictly increasing). Full caller control, the same escape hatch :func:`mixle_pde.reliability.form`
  gives a caller who wants to supply an analytic ``gradient`` instead of a finite difference.
* ``adaptive=True`` (the default) -- at every rung, the next temperature is chosen by bisection so the
  post-reweight effective sample size lands at ``resample_threshold * n_particles`` (or the schedule
  jumps straight to ``temperature=1`` when even that full jump keeps ESS above the target). This is the
  standard adaptive-ESS construction (Del Moral, Doucet & Jasra 2006; Zhou, Johansen & Aston 2016): it
  never wastes rungs on an overly-cautious fixed grid, and it never takes a step so large the population
  degenerates before the next resample. ``n_temperatures`` becomes a safety cap on the number of rungs;
  exceeding it raises ``RuntimeError`` rather than silently truncating the schedule short of
  ``temperature=1``.
* ``adaptive=False`` and no explicit schedule -- a fixed ``n_temperatures``-step linear grid
  (``linspace(0, 1, n_temperatures + 1)``), resampling only when ESS actually drops below
  ``resample_threshold * n_particles`` (or at the final rung, forced) -- the same conditional-resample
  structure :func:`~mixle_pde.field_assimilation.particle_assimilate_4d` already uses over the time axis.

Limitations
-----------
* Small/medium-scale reference path, like every other sampler in :mod:`mixle_pde.field_mcmc` -- not a
  distributed or GPU-batched production SMC implementation.
* The rejuvenation kernel is pCN only; it inherits pCN's own scope note (a Gaussian prior, evaluated in
  the field's unconstrained coordinates).
* ``log_evidence`` is the standard SMC marginal-likelihood estimator (unbiased in expectation for a
  fixed schedule; the adaptive schedule's own selection introduces a small extra variance term the
  literature usually neglects) -- treat it as a diagnostic, not a certified Bayes-factor input.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mixle_pde.field_assimilation import _logsumexp, _systematic_resample
from mixle_pde.field_inversion import FieldGaussianPrior
from mixle_pde.field_mcmc import _prior_log_density_kernel, _prior_precision_and_mean
from mixle_pde.latent import Field3D, PosteriorFieldSamples3D
from mixle_pde.observations import ForwardOperatorRegistry, Observation

__all__ = ["SMCReport", "smc_tempering_field_invert"]


@dataclass(frozen=True)
class SMCReport:
    """Diagnostics for an :func:`smc_tempering_field_invert` run.

    ``temperature_schedule`` is the actual sequence of temperatures used (``[0.0, ..., 1.0]``, length
    ``n_steps + 1``); ``effective_sample_size[k]`` is the population's effective sample size
    immediately after reweighting to ``temperature_schedule[k]`` (index 0 is always ``n_particles``,
    the starting population's exact ESS before any reweighting). ``log_evidence`` sums each rung's
    incremental log-normalizer, the standard SMC estimate of ``log p(observations)``. A resample is
    always followed by ``rejuvenate_steps`` pCN moves (0 by default disables rejuvenation, matching
    every other rejuvenation knob in this repo); ``mean_rejuvenation_acceptance_rate`` averages the
    per-move acceptance fraction across every rejuvenation move actually run, and is ``nan`` (not an
    error) when ``rejuvenate_steps == 0`` or no resample ever occurred.
    """

    n_particles: int
    n_steps: int
    temperature_schedule: np.ndarray
    resample_count: int
    effective_sample_size: list[float]
    log_evidence: float
    mean_rejuvenation_acceptance_rate: float

    def __post_init__(self) -> None:
        schedule = np.asarray(self.temperature_schedule, dtype=float)
        if schedule.ndim != 1 or schedule.shape[0] != self.n_steps + 1:
            raise ValueError("temperature_schedule must have shape (n_steps + 1,).")
        if schedule[0] != 0.0 or schedule[-1] != 1.0:
            raise ValueError("temperature_schedule must start at 0.0 and end at 1.0.")
        if np.any(np.diff(schedule) <= 0.0):
            raise ValueError("temperature_schedule must be strictly increasing.")
        object.__setattr__(self, "temperature_schedule", schedule)
        if len(self.effective_sample_size) != self.n_steps + 1:
            raise ValueError("effective_sample_size must have one entry per temperature_schedule entry.")
        if self.n_particles < 2:
            raise ValueError("n_particles must be at least 2.")
        if self.resample_count < 0:
            raise ValueError("resample_count must be non-negative.")
        if not np.isfinite(self.log_evidence):
            raise ValueError("log_evidence must be finite.")


def _reweight(delta: float, log_weights: np.ndarray, loglik: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    """One tempering increment's self-normalized importance reweight.

    ``loglik`` is each particle's CURRENT log-likelihood (unchanged by reweighting, only by a move
    step); reweighting by ``exp(delta * loglik)`` is exactly the incremental importance weight from
    ``pi_temperature`` to ``pi_{temperature + delta}`` for a population already distributed
    (approximately) as ``pi_temperature``. Returns ``(new_log_weights, weights, ess, log_norm)``.
    """
    incremental = log_weights + delta * loglik
    log_norm = _logsumexp(incremental)
    new_log_weights = incremental - log_norm
    weights = np.exp(new_log_weights)
    ess = float(1.0 / np.sum(weights**2))
    return new_log_weights, weights, ess, float(log_norm)


def _next_adaptive_temperature(
    temperature: float,
    log_weights: np.ndarray,
    loglik: np.ndarray,
    *,
    target_ess: float,
    max_bisection_iter: int = 60,
) -> tuple[float, np.ndarray, np.ndarray, float, float, bool]:
    """Choose the next temperature via bisection on effective sample size.

    Effective sample size is monotonically non-increasing in ``delta`` (a bigger tempering jump always
    concentrates weight on the currently-best-fit particles at least as much), so bisecting for the
    ``delta`` where ESS first reaches ``target_ess`` is well posed. If even the full jump to
    ``temperature=1`` keeps ESS at or above ``target_ess``, that full jump is taken directly (no need to
    stop early just because we could). Returns ``(next_temperature, new_log_weights, weights, ess,
    log_norm, hit_target)``; ``hit_target`` tells the caller this step was deliberately sized to land
    ESS at the resampling floor, so it must resample now (a floating-point-exact ``ess <=
    target_ess`` re-check would be fragile against the bisection's own tolerance).
    """
    remaining = 1.0 - temperature
    full_log_weights, full_weights, full_ess, full_log_norm = _reweight(remaining, log_weights, loglik)
    if full_ess >= target_ess:
        return 1.0, full_log_weights, full_weights, full_ess, full_log_norm, False

    lo, hi = 0.0, remaining
    for _ in range(max_bisection_iter):
        mid = 0.5 * (lo + hi)
        _, _, ess_mid, _ = _reweight(mid, log_weights, loglik)
        if ess_mid >= target_ess:
            lo = mid
        else:
            hi = mid
    new_log_weights, weights, ess, log_norm = _reweight(lo, log_weights, loglik)
    return temperature + lo, new_log_weights, weights, ess, log_norm, True


def _batch_log_likelihood(
    grid: Field3D,
    registry: ForwardOperatorRegistry,
    observations: list[Observation],
    particles: np.ndarray,
) -> np.ndarray:
    """Each particle's total log-likelihood, ``-inf`` for a particle whose physical field is non-finite.

    Loops one particle at a time through :meth:`~mixle_pde.observations.ForwardOperatorRegistry.
    total_log_likelihood`, the same per-particle loop :func:`~mixle_pde.field_assimilation.
    particle_assimilate_4d` already uses (that method takes one field vector at a time; there is no
    batched registry entry point to vectorize against).
    """
    physical = grid.from_unconstrained(particles)
    out = np.empty(particles.shape[0], dtype=float)
    for i in range(particles.shape[0]):
        if np.all(np.isfinite(physical[i])):
            out[i] = registry.total_log_likelihood(grid, physical[i], observations)
        else:
            out[i] = -np.inf
    return out


def _tempered_pcn_move(
    particles: np.ndarray,
    loglik: np.ndarray,
    grid: Field3D,
    registry: ForwardOperatorRegistry,
    observations: list[Observation],
    prior_mean: np.ndarray,
    chol_precision: np.ndarray,
    *,
    temperature: float,
    rejuvenate_beta: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, float]:
    """One vectorized tempered-pCN rejuvenation move over every particle (resample-move, Gilks & Berzuini 2001).

    The proposal is exactly :func:`mixle_pde.field_assimilation._pcn_rejuvenate_particles`'s: it is
    reversible with respect to the Gaussian prior itself, so a target of the prior ALONE would accept
    with probability 1 unconditionally. For the tempered target ``pi_temperature(u) propto prior(u) *
    likelihood(u) ** temperature``, prior-reversibility means the prior factor cancels exactly out of
    the Metropolis-Hastings ratio, leaving ``alpha = min(1, (likelihood(proposal) /
    likelihood(current)) ** temperature)`` -- i.e. ``log_alpha = temperature * (proposal_loglik -
    current_loglik)``, :func:`mixle_pde.field_assimilation._pcn_rejuvenate_particles`'s own acceptance
    test with one extra ``temperature`` factor. That factor is exactly what makes this kernel invariant
    to THIS rung's tempered target instead of the untempered posterior.
    """
    n_particles, n_state = particles.shape
    shrink = float(np.sqrt(1.0 - rejuvenate_beta**2))
    z = rng.standard_normal((n_particles, n_state))
    xi = np.linalg.solve(chol_precision.T, z.T).T  # each row ~ N(0, prior_cov)
    proposal = prior_mean[None, :] + shrink * (particles - prior_mean[None, :]) + rejuvenate_beta * xi
    proposal_loglik = _batch_log_likelihood(grid, registry, observations, proposal)
    log_alpha = temperature * (proposal_loglik - loglik)
    accept = np.log(rng.random(n_particles)) < log_alpha
    out_particles = particles.copy()
    out_particles[accept] = proposal[accept]
    out_loglik = loglik.copy()
    out_loglik[accept] = proposal_loglik[accept]
    return out_particles, out_loglik, float(np.mean(accept))


def smc_tempering_field_invert(
    grid: Field3D,
    observations: list[Observation],
    registry: ForwardOperatorRegistry,
    prior: FieldGaussianPrior,
    *,
    n_particles: int = 500,
    n_temperatures: int = 50,
    temperature_schedule: np.ndarray | None = None,
    adaptive: bool = True,
    resample_threshold: float = 0.5,
    rejuvenate_steps: int = 5,
    rejuvenate_beta: float = 0.3,
    jitter: float = 1.0e-9,
    rng: np.random.Generator | None = None,
) -> tuple[PosteriorFieldSamples3D, SMCReport]:
    """Sample a field posterior with Sequential Monte Carlo tempering.

    The population starts as ``n_particles`` exact i.i.d. draws from the prior (``temperature=0``,
    trivial to sample exactly) and is walked to the true posterior (``temperature=1``) through a
    sequence of tempered targets ``prior(u) * likelihood(u) ** temperature`` (see the module docstring
    for the temperature-schedule modes and for why this is the recommended fallback when a posterior may
    have well-separated modes a single chain could miss). At each rung: reweight by the incremental
    likelihood power, resample via :func:`~mixle_pde.field_assimilation._systematic_resample` when
    effective sample size drops to (adaptive) or below (fixed schedule) ``resample_threshold *
    n_particles`` -- always at the final rung, regardless -- and, after every resample, rejuvenate with
    ``rejuvenate_steps`` tempered-pCN moves (see :func:`_tempered_pcn_move`) to restore the diversity
    resampling collapsed. Because the final rung always resamples, the returned population is exactly
    equally weighted.

    Raises ``ValueError`` for the usual invalid-argument reasons (see the parameter list) and
    ``RuntimeError`` if an adaptive schedule fails to reach ``temperature=1`` within ``n_temperatures``
    rungs (pass a larger ``n_temperatures`` or an explicit ``temperature_schedule``).
    """
    if not observations:
        raise ValueError("need at least one observation to invert.")
    n_particles = int(n_particles)
    if n_particles < 2:
        raise ValueError("n_particles must be at least 2.")
    if not 0.0 < resample_threshold <= 1.0:
        raise ValueError("resample_threshold must be in (0, 1].")
    rejuvenate_steps = int(rejuvenate_steps)
    if rejuvenate_steps < 0:
        raise ValueError("rejuvenate_steps must be non-negative.")
    if not 0.0 < rejuvenate_beta <= 1.0:
        raise ValueError("rejuvenate_beta must be in (0, 1].")
    if jitter < 0.0:
        raise ValueError("jitter must be non-negative.")
    n_temperatures = int(n_temperatures)
    if n_temperatures < 1:
        raise ValueError("n_temperatures must be positive.")

    fixed_schedule: np.ndarray | None
    if temperature_schedule is not None:
        fixed_schedule = np.asarray(temperature_schedule, dtype=float)
        if fixed_schedule.ndim != 1 or fixed_schedule.shape[0] < 2:
            raise ValueError("temperature_schedule must be a 1-D array of at least two values.")
        if fixed_schedule[0] != 0.0 or fixed_schedule[-1] != 1.0:
            raise ValueError("temperature_schedule must start at 0.0 and end at 1.0.")
        if np.any(np.diff(fixed_schedule) <= 0.0):
            raise ValueError("temperature_schedule must be strictly increasing.")
    elif not adaptive:
        fixed_schedule = np.linspace(0.0, 1.0, n_temperatures + 1)
    else:
        fixed_schedule = None

    rng = np.random.default_rng() if rng is None else rng
    n = grid.n
    prior_mean, precision = _prior_precision_and_mean(grid, prior)
    prior_cov = np.linalg.inv(precision + jitter * np.eye(n))
    chol_precision = np.linalg.cholesky(precision)

    particles = rng.multivariate_normal(prior_mean, prior_cov, size=n_particles)
    loglik = _batch_log_likelihood(grid, registry, observations, particles)
    if not np.any(np.isfinite(loglik)):
        raise ValueError("every particle drawn from the prior has a non-finite likelihood; cannot start tempering.")

    log_weights = np.full(n_particles, -np.log(n_particles), dtype=float)
    temperature = 0.0
    temperatures: list[float] = [0.0]
    ess_history: list[float] = [float(n_particles)]
    log_evidence = 0.0
    resample_count = 0
    acceptance_rates: list[float] = []

    step = 0
    while temperature < 1.0:
        step += 1
        if fixed_schedule is not None:
            next_temperature = float(fixed_schedule[step])
            delta = next_temperature - temperature
            new_log_weights, weights, ess, log_norm = _reweight(delta, log_weights, loglik)
            hit_target = False
        else:
            if step > n_temperatures:
                raise RuntimeError(
                    f"adaptive SMC tempering did not reach temperature=1 within n_temperatures={n_temperatures} "
                    f"rungs (stalled at temperature={temperature!r}); pass a larger n_temperatures or an "
                    "explicit temperature_schedule."
                )
            target_ess = resample_threshold * n_particles
            next_temperature, new_log_weights, weights, ess, log_norm, hit_target = _next_adaptive_temperature(
                temperature, log_weights, loglik, target_ess=target_ess
            )

        if not np.isfinite(log_norm):
            raise ValueError(
                f"all particle weights vanished while tempering from {temperature!r} to {next_temperature!r}; "
                "the likelihood is degenerate (non-finite everywhere the current population has support)."
            )

        temperature = next_temperature
        temperatures.append(temperature)
        ess_history.append(ess)
        log_evidence += log_norm
        log_weights = new_log_weights

        should_resample = hit_target or temperature >= 1.0 or ess <= resample_threshold * n_particles
        if should_resample:
            idx = _systematic_resample(weights, rng)
            particles = particles[idx]
            loglik = loglik[idx]
            log_weights = np.full(n_particles, -np.log(n_particles), dtype=float)
            resample_count += 1
            for _ in range(rejuvenate_steps):
                particles, loglik, acceptance_rate = _tempered_pcn_move(
                    particles,
                    loglik,
                    grid,
                    registry,
                    observations,
                    prior_mean,
                    chol_precision,
                    temperature=temperature,
                    rejuvenate_beta=rejuvenate_beta,
                    rng=rng,
                )
                acceptance_rates.append(acceptance_rate)

    prior_logdensity = np.array([_prior_log_density_kernel(u, prior_mean, precision) for u in particles])
    log_posterior = prior_logdensity + loglik
    best = particles[int(np.argmax(log_posterior))].copy()

    posterior = PosteriorFieldSamples3D(
        grid=grid,
        samples=particles,
        log_posterior=log_posterior,
        map=best,
        provenance={
            "method": "smc_tempering",
            "small_reference": True,
            "n_particles": n_particles,
            "n_steps": len(temperatures) - 1,
            "adaptive": fixed_schedule is None,
            "resample_threshold": float(resample_threshold),
            "temperature_schedule": [float(t) for t in temperatures],
        },
    )
    report = SMCReport(
        n_particles=n_particles,
        n_steps=len(temperatures) - 1,
        temperature_schedule=np.asarray(temperatures, dtype=float),
        resample_count=resample_count,
        effective_sample_size=ess_history,
        log_evidence=float(log_evidence),
        mean_rejuvenation_acceptance_rate=float(np.mean(acceptance_rates)) if acceptance_rates else float("nan"),
    )
    return posterior, report
