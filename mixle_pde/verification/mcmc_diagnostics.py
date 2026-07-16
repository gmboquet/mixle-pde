"""Multi-chain MCMC convergence diagnostics: split R-hat and effective sample size (MP-I8).

Source: notes/mixle-pde-ai-native-multiphysics-work-plan.md workstream I, MP-I8 ("Posterior
validation, calibration, identifiability, and model comparison"), which names "ESS/R-hat/
divergences/autocorrelation" as required sampler diagnostics, and MP-I9 ("Scalable, restartable,
and reproducible Bayesian execution"), whose accept criterion requires that "serial/parallel
analytic posterior summaries agree within Monte Carlo error." Neither of those is claimed in full
here -- this module covers the mixing/stationarity slice only (see Limitations below).

:mod:`mixle_pde.field_mcmc` already ships four single-chain samplers (Random-Walk Metropolis, pCN,
MALA, HMC) but nothing that asks the question every one of them needs answered before a posterior
is trustworthy: *did independently started chains actually converge to the same distribution, and
how many independent-equivalent draws did they produce?* This module answers that from raw chain
arrays -- it never runs a sampler itself and never imports :mod:`mixle_pde.field_mcmc` -- so it
applies equally to any sampler's stored draws, including :func:`mixle_pde.sample_update`'s
importance-weighted updates once resampled to equal weight, or a future sampler this repo has not
written yet.

Method
------
``r_hat`` is the classic **split** potential scale reduction factor (Gelman & Rubin, 1992; the split
variant is BDA3 section 11.4): each of the ``m`` input chains is cut into a first and second half
before the between/within variance decomposition, so a chain that drifts over time (non-stationary,
not just poorly mixed between chains) also inflates R-hat instead of hiding behind a healthy
between-chain agreement. ``var_plus = (n-1)/n * W + B/n`` is the usual pooled-variance estimate of
the target's marginal variance; ``r_hat = sqrt(var_plus / W)``. A well-mixed, stationary set of
chains gives ``r_hat`` close to ``1.0``; chains that disagree or drift give ``r_hat`` visibly above
it.

``ess`` (effective sample size) applies Geyer's (1992) initial-positive-sequence estimator of the
integrated autocorrelation time to each of the ``2m`` split chains independently (autocovariance via
a zero-padded FFT, consecutive lags paired and summed, truncated at the first non-positive pair, and
the pair-sum sequence forced non-increasing -- Geyer's "initial monotone sequence", which keeps noisy
tail lags from being summed indefinitely), then **sums** the ``n / tau_hat`` contribution of each
split chain. This is a simpler, per-chain-normalized relative of Stan/ArviZ's pooled bulk-ESS (which
pools the autocovariance across chains before the tau estimate rather than after); it is not a
byte-for-byte reproduction of that statistic, but it has the same asymptotics: independent draws give
``ess`` close to the total draw count, and it degrades correctly under within-chain autocorrelation.

Limitations
-----------
* Mixing/stationarity evidence only. No prior/posterior predictive check, simulation-based
  calibration, coverage/rank diagnostic, multimodality test, or divergence-transition count is
  computed here -- those remain open per MP-I8.
* This repo's registered samplers (Random-Walk Metropolis, pCN, MALA, HMC) are not NUTS, so there is
  no divergence-transition count to report even in principle; that diagnostic does not apply.
* ``ess`` is the per-split-chain-summed Geyer estimator described above, not Stan's pooled bulk-ESS
  or tail-ESS. Do not compare the numbers this module reports directly against ArviZ/Stan output and
  expect an exact match; the qualitative conclusion (well-mixed vs. not) will agree.
* No resumability, checkpointing, counter-based random streams, or distributed/multi-process
  execution is implemented (the broader MP-I9 scope). This module only evaluates chains a caller
  already produced by whatever means.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from mixle_pde.latent import PosteriorFieldSamples3D

__all__ = [
    "ChainDiagnostics",
    "split_rhat",
    "multichain_ess",
    "evaluate_chain_convergence",
    "chains_from_posterior_samples",
]

# Widely used rule-of-thumb thresholds (e.g. Stan/ArviZ guidance): R-hat comfortably below 1.1, and
# a total effective sample size of at least a few hundred before trusting posterior summaries.
# Callers with stricter or looser requirements should pass their own values.
_DEFAULT_R_HAT_THRESHOLD = 1.05
_DEFAULT_MIN_ESS = 400.0


def _prepare_chains(chains: Any) -> np.ndarray:
    """Validate and reshape input into ``(n_chains, n_draws, n_parameters)``."""
    arr = np.asarray(chains, dtype=float)
    if arr.ndim == 2:
        arr = arr[:, :, None]
    if arr.ndim != 3:
        raise ValueError("chains must have shape (n_chains, n_draws) or (n_chains, n_draws, n_parameters).")
    m, n, _k = arr.shape
    if m < 1:
        raise ValueError("need at least one chain.")
    if n < 4:
        raise ValueError("need at least four draws per chain (chains are split in half for diagnostics).")
    if not np.all(np.isfinite(arr)):
        raise ValueError("chains must be finite; found NaN or inf.")
    return arr


def _split_chains(chains: np.ndarray) -> np.ndarray:
    """``(m, n, k) -> (2m, n // 2, k)``: cut every chain into a first half and a last half.

    When ``n`` is odd, the single middle draw is dropped from both halves (kept out of both, not
    duplicated into either) so the two halves stay equal length.
    """
    m, n, k = chains.shape
    half = n // 2
    first = chains[:, :half, :]
    second = chains[:, n - half :, :]
    return np.concatenate([first, second], axis=0)


def split_rhat(chains: Any) -> np.ndarray:
    """Split Gelman-Rubin potential scale reduction factor, per parameter.

    ``chains`` is ``(n_chains, n_draws)`` or ``(n_chains, n_draws, n_parameters)``. Returns an
    ``(n_parameters,)`` array (or ``(1,)`` for 2-D input). A chain with zero within-chain variance
    (perfectly stuck) yields a non-finite entry for that parameter rather than a silently wrong
    number; treat non-finite R-hat as "not converged."
    """
    prepared = _prepare_chains(chains)
    split = _split_chains(prepared)
    m2, half, _k = split.shape
    chain_means = split.mean(axis=1)  # (2m, k)
    chain_vars = split.var(axis=1, ddof=1)  # (2m, k)
    within = chain_vars.mean(axis=0)  # (k,)  == W
    grand_mean = chain_means.mean(axis=0)  # (k,)
    between_over_n = ((chain_means - grand_mean) ** 2).sum(axis=0) / (m2 - 1)  # (k,) == B / n
    var_plus = ((half - 1) / half) * within + between_over_n
    with np.errstate(invalid="ignore", divide="ignore"):
        rhat = np.sqrt(var_plus / within)
    return rhat


def _chain_autocorrelation(x: np.ndarray, max_lag: int) -> np.ndarray:
    """Lag ``0..max_lag`` autocorrelation of one 1-D chain via a zero-padded FFT autocovariance.

    Zero-padding to at least ``2 * n`` before the FFT avoids circular wraparound contamination, so
    the result matches the direct (textbook) lag-``t`` autocovariance ``sum_i (x_i - mean)(x_{i+t} -
    mean) / (n - t)`` for every ``t <= max_lag < n``. A chain with zero variance (perfectly constant)
    returns an all-zero array -- callers must not read that as "zero autocorrelation is good mixing";
    :func:`_chain_effective_sample_size` treats it as a degenerate, single-effective-draw chain.
    """
    n = x.shape[0]
    if max_lag >= n:
        raise ValueError("max_lag must be smaller than the chain length.")
    centered = x - x.mean()
    fft_size = 2
    while fft_size < 2 * n:
        fft_size *= 2
    freq = np.fft.rfft(centered, n=fft_size)
    autocov_full = np.fft.irfft(freq * np.conjugate(freq), n=fft_size).real
    lag_counts = np.arange(n, n - max_lag - 1, -1, dtype=float)
    autocov = autocov_full[: max_lag + 1] / lag_counts
    variance = autocov[0]
    if variance <= 0.0:
        return np.zeros(max_lag + 1)
    return autocov / variance


def _geyer_initial_positive_tau(rho: np.ndarray) -> float:
    """Geyer's (1992) initial positive + initial monotone sequence estimate of ``tau = 1 + 2*sum(rho_t)``.

    ``rho`` holds lag-``0..T`` autocorrelations of one chain (``rho[0] == 1``). Consecutive lags
    ``(rho[1]+rho[2]), (rho[3]+rho[4]), ...`` are summed in pairs; summation stops at the first
    non-positive pair, and the accepted pair sums are then forced to be non-increasing. Both rules
    keep noisy, small tail autocorrelations from being summed indefinitely into a spuriously large or
    negative ``tau``.
    """
    n_lags = rho.shape[0]
    pair_sums: list[float] = []
    t = 1
    while t + 1 < n_lags:
        pair = float(rho[t] + rho[t + 1])
        if pair < 0.0:
            break
        pair_sums.append(pair)
        t += 2
    for i in range(1, len(pair_sums)):
        if pair_sums[i] > pair_sums[i - 1]:
            pair_sums[i] = pair_sums[i - 1]
    tau = 1.0 + 2.0 * sum(pair_sums)
    return max(tau, 1.0)


def _chain_effective_sample_size(x: np.ndarray) -> float:
    """Geyer initial-positive-sequence effective sample size of one 1-D chain."""
    n = x.shape[0]
    if n < 4:
        return float(n)
    rho = _chain_autocorrelation(x, max_lag=n - 1)
    if not np.any(rho[1:]):
        # Degenerate (zero-variance / perfectly stuck) chain: one effective draw, not `n`.
        return 1.0
    tau = _geyer_initial_positive_tau(rho)
    return float(n / tau)


def multichain_ess(chains: Any) -> np.ndarray:
    """Multi-chain effective sample size, per parameter (see module docstring for the exact method).

    ``chains`` is ``(n_chains, n_draws)`` or ``(n_chains, n_draws, n_parameters)``. Returns an
    ``(n_parameters,)`` array. Each of the ``2 * n_chains`` split chains contributes its own Geyer
    initial-positive-sequence estimate; the total is their sum, so a set of genuinely independent
    draws reports an ``ess`` close to the total draw count (``n_chains * n_draws``), and stronger
    within-chain autocorrelation reports proportionally less.
    """
    prepared = _prepare_chains(chains)
    split = _split_chains(prepared)
    n_split, _half, k = split.shape
    ess = np.zeros(k, dtype=float)
    for p in range(k):
        ess[p] = sum(_chain_effective_sample_size(split[c, :, p]) for c in range(n_split))
    return ess


@dataclass(frozen=True)
class ChainDiagnostics:
    """Multi-chain convergence verdict for a set of posterior sampler chains.

    ``r_hat`` and ``ess`` are per-parameter, in the same order as ``parameter_names``. ``converged``
    is true only when every parameter's R-hat is finite and at or below ``r_hat_threshold`` *and*
    every parameter's ess is at or above ``min_ess`` -- a single failing parameter fails the whole
    verdict, never averaged away. See the module docstring for exactly what this does and does not
    check.
    """

    n_chains: int
    n_draws: int
    n_parameters: int
    parameter_names: tuple[str, ...]
    r_hat: tuple[float, ...]
    ess: tuple[float, ...]
    r_hat_threshold: float
    min_ess: float
    converged: bool
    detail: str
    limitations: tuple[str, ...] = (
        "mixing/stationarity evidence only: no prior/posterior predictive check, simulation-based "
        "calibration, or divergence-transition diagnostic",
        "ess is a per-split-chain Geyer initial-positive-sequence estimate summed across chains, not "
        "a reproduction of Stan/ArviZ's pooled bulk-ESS",
    )


def evaluate_chain_convergence(
    chains: Any,
    *,
    parameter_names: Sequence[str] | None = None,
    r_hat_threshold: float = _DEFAULT_R_HAT_THRESHOLD,
    min_ess: float = _DEFAULT_MIN_ESS,
) -> ChainDiagnostics:
    """Run :func:`split_rhat` and :func:`multichain_ess` and fold the result into a verdict.

    ``chains`` is ``(n_chains, n_draws)`` or ``(n_chains, n_draws, n_parameters)`` -- independent
    chains from any sampler, not necessarily one of this repo's own (see
    :func:`chains_from_posterior_samples` for a convenience bridge from
    :mod:`mixle_pde.field_mcmc` output). Raises :class:`ValueError` for fewer than one chain, fewer
    than four draws per chain, non-finite draws, or a ``parameter_names`` length mismatch.
    """
    prepared = _prepare_chains(chains)
    m, n, k = prepared.shape
    if parameter_names is None:
        names = tuple(f"param_{i}" for i in range(k))
    else:
        names = tuple(parameter_names)
        if len(names) != k:
            raise ValueError(f"parameter_names must have length {k}, got {len(names)}.")

    r_hat = split_rhat(prepared)
    ess = multichain_ess(prepared)
    finite = np.isfinite(r_hat)
    converged = bool(np.all(finite) and np.all(r_hat[finite] <= r_hat_threshold) and np.all(ess >= min_ess))
    worst_r_hat = float(np.max(r_hat)) if np.all(finite) else float("inf")
    worst_ess = float(np.min(ess))
    detail = (
        f"{m} chains x {n} draws x {k} parameter(s): max r_hat={worst_r_hat:.4f} "
        f"(threshold {r_hat_threshold:.4f}), min ess={worst_ess:.1f} (threshold {min_ess:.1f}) -> "
        f"{'CONVERGED' if converged else 'NOT CONVERGED'}"
    )
    return ChainDiagnostics(
        n_chains=m,
        n_draws=n,
        n_parameters=k,
        parameter_names=names,
        r_hat=tuple(float(v) for v in r_hat),
        ess=tuple(float(v) for v in ess),
        r_hat_threshold=r_hat_threshold,
        min_ess=min_ess,
        converged=converged,
        detail=detail,
    )


def chains_from_posterior_samples(posteriors: Sequence[PosteriorFieldSamples3D]) -> np.ndarray:
    """Stack independent sampler runs into the ``(n_chains, n_draws, n_parameters)`` array this module expects.

    Each element of ``posteriors`` is one independent chain's stored output -- e.g. one
    :func:`mixle_pde.field_mcmc.metropolis_field_invert` or
    :func:`mixle_pde.field_mcmc.pcn_field_invert` call with its own seed and/or start point. This
    module never calls a sampler itself; the caller is responsible for making the runs genuinely
    independent (different ``rng``) and comparable (same ``n_samples``/``thin``/``burn_in``, same
    field/grid). Raises :class:`ValueError` if the stored ``.samples`` shapes disagree.
    """
    if len(posteriors) < 1:
        raise ValueError("need at least one posterior chain.")
    arrays = [np.asarray(p.samples, dtype=float) for p in posteriors]
    shapes = {a.shape for a in arrays}
    if len(shapes) != 1:
        raise ValueError(f"all posterior chains must share one (n_draws, n_parameters) shape; got {sorted(shapes)}")
    return np.stack(arrays, axis=0)
