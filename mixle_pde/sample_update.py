"""Importance updates for sampled 3D/4D field posteriors."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from mixle_pde.field_assimilation import PosteriorFieldSamples4D, _logsumexp, _systematic_resample
from mixle_pde.latent import PosteriorFieldSamples3D


LogLikelihood3D = Callable[[np.ndarray], float]
LogLikelihood4D = Callable[[np.ndarray, float], float]


@dataclass(frozen=True)
class SampleUpdateReport:
    """Diagnostics for a sampled-posterior Bayesian importance update."""

    n_input_samples: int
    n_output_samples: int
    effective_sample_size: float
    log_evidence_increment: float
    max_log_likelihood: float
    resampled: bool
    likelihood_count: int


def update_sampled_field_posterior(
    posterior: PosteriorFieldSamples3D,
    log_likelihoods: Sequence[LogLikelihood3D],
    *,
    n_samples: int | None = None,
    rng: np.random.Generator | None = None,
    resample: bool = True,
) -> tuple[PosteriorFieldSamples3D, SampleUpdateReport]:
    """Bayesian-update a sampled 3D posterior with arbitrary field log-likelihood callbacks."""
    if not log_likelihoods:
        raise ValueError("log_likelihoods must contain at least one callback.")
    rng = np.random.default_rng() if rng is None else rng
    physical = posterior.physical_samples
    log_like = np.array(
        [sum(float(fn(draw)) for fn in log_likelihoods) for draw in physical],
        dtype=float,
    )
    updated, report = _update_samples(
        posterior.samples,
        log_like,
        old_log_posterior=posterior.log_posterior,
        n_samples=n_samples,
        rng=rng,
        resample=resample,
        likelihood_count=len(log_likelihoods),
    )
    out = PosteriorFieldSamples3D(
        grid=posterior.grid,
        samples=updated["samples"],
        log_posterior=updated["log_posterior"],
        map=updated["map"],
        provenance=posterior.provenance
        | {
            "method": "importance_update",
            "effective_sample_size": report.effective_sample_size,
            "log_evidence_increment": report.log_evidence_increment,
        },
    )
    return out, report


def update_sampled_field_posterior_4d(
    posterior: PosteriorFieldSamples4D,
    log_likelihoods_by_time: Sequence[Sequence[LogLikelihood4D]],
    *,
    n_samples: int | None = None,
    rng: np.random.Generator | None = None,
    resample: bool = True,
) -> tuple[PosteriorFieldSamples4D, SampleUpdateReport]:
    """Bayesian-update sampled 4D posterior trajectories with per-time log-likelihood callbacks."""
    if len(log_likelihoods_by_time) != posterior.times.size:
        raise ValueError("log_likelihoods_by_time must have one entry per posterior time.")
    if not any(log_likelihoods_by_time):
        raise ValueError("at least one time must contain a likelihood callback.")
    rng = np.random.default_rng() if rng is None else rng
    physical = posterior.physical_samples
    log_like = np.zeros(posterior.n_samples, dtype=float)
    count = 0
    for ti, callbacks in enumerate(log_likelihoods_by_time):
        time = float(posterior.times[ti])
        for fn in callbacks:
            count += 1
            for sample_id in range(posterior.n_samples):
                log_like[sample_id] += float(fn(physical[sample_id, ti], time))
    updated, report = _update_samples(
        posterior.samples,
        log_like,
        old_log_posterior=None,
        n_samples=n_samples,
        rng=rng,
        resample=resample,
        likelihood_count=count,
    )
    out = PosteriorFieldSamples4D(
        grid=posterior.grid,
        times=posterior.times,
        samples=updated["samples"],
        provenance=posterior.provenance
        | {
            "method": "importance_update_4d",
            "effective_sample_size": report.effective_sample_size,
            "log_evidence_increment": report.log_evidence_increment,
        },
    )
    return out, report


def _update_samples(
    samples: np.ndarray,
    log_likelihood: np.ndarray,
    *,
    old_log_posterior: np.ndarray | None,
    n_samples: int | None,
    rng: np.random.Generator,
    resample: bool,
    likelihood_count: int,
) -> tuple[dict[str, Any], SampleUpdateReport]:
    if not np.all(np.isfinite(log_likelihood)):
        raise ValueError("all likelihood callbacks must return finite log-likelihoods.")
    n_input = int(samples.shape[0])
    n_output = n_input if n_samples is None else int(n_samples)
    if n_output <= 0:
        raise ValueError("n_samples must be positive.")
    log_norm = _logsumexp(log_likelihood)
    weights = np.exp(log_likelihood - log_norm)
    ess = float(1.0 / np.sum(weights**2))
    best = int(np.argmax(log_likelihood))
    if resample:
        idx = _systematic_resample(weights, rng)
        if n_output != n_input:
            idx = rng.choice(n_input, size=n_output, replace=True, p=weights)
        updated_samples = samples[idx].copy()
        if old_log_posterior is None:
            updated_logp = log_likelihood[idx].copy()
        else:
            updated_logp = old_log_posterior[idx] + log_likelihood[idx]
    else:
        updated_samples = samples.copy()
        if old_log_posterior is None:
            updated_logp = log_likelihood.copy()
        else:
            updated_logp = old_log_posterior + log_likelihood
    report = SampleUpdateReport(
        n_input_samples=n_input,
        n_output_samples=int(updated_samples.shape[0]),
        effective_sample_size=ess,
        log_evidence_increment=float(log_norm - np.log(n_input)),
        max_log_likelihood=float(log_likelihood[best]),
        resampled=bool(resample),
        likelihood_count=int(likelihood_count),
    )
    return {"samples": updated_samples, "log_posterior": updated_logp, "map": samples[best].copy()}, report
