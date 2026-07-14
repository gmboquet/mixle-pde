"""Model-error (theory-error) discrepancy covariance for likelihoods that assume a perfect forward model.

Every likelihood in :mod:`mixle_pde.observations` and every ``Differential`` fit in
:mod:`mixle_pde.inverse` scores data against a forward operator's prediction under a single noise
covariance -- the INSTRUMENT noise. That is correct only when the forward operator itself is exact.
Real forward operators (a truncated physics, a coarse discretization, an unmodeled process) carry
their own systematic error, and treating the data covariance as if it were the whole story makes the
posterior overconfident: the credible intervals shrink around a systematically wrong estimate rather
than widening to admit the theory is imperfect (Kennedy & O'Hagan's "model discrepancy").

This module supplies the discrepancy covariance and the one way to fold it into a total covariance --
inflating the noise model so a mis-specified operator no longer buys unwarranted certainty. It is
deliberately minimal: a diagonal discrepancy sized by a single ``sigma_model`` (the common case, when
the analyst has an estimate of the RMS size of the theory error but no belief about its correlation
structure), or a full discrepancy covariance supplied directly when one is known. Learned or
correlated discrepancy kernels are out of scope here.
"""

from __future__ import annotations

import numpy as np

__all__ = ["diagonal_discrepancy", "inflated_noise_cov"]


def diagonal_discrepancy(n: int, sigma_model: float) -> np.ndarray:
    """Diagonal model-discrepancy covariance: ``n`` independent variances of size ``sigma_model**2``.

    The common case -- an analyst has an estimate of the RMS size of the theory error (e.g. from a
    held-out comparison against a higher-fidelity forward model) but no belief about how it correlates
    across observations, so each datum gets the same independent variance inflation.
    """
    if n <= 0:
        raise ValueError("n must be a positive integer.")
    if sigma_model < 0.0:
        raise ValueError("sigma_model must be non-negative.")
    return np.full(int(n), float(sigma_model) ** 2)


def inflated_noise_cov(noise_cov: np.ndarray, model_error_cov: np.ndarray) -> np.ndarray:
    """Total covariance ``R + R_model``: instrument noise inflated by the model-discrepancy covariance.

    Both arguments are either ``(n,)`` (diagonal variances) or ``(n, n)`` (full covariance), matching
    the shape convention of :class:`mixle_pde.observations.Observation.noise_cov`. When both are
    diagonal the sum stays diagonal (an ``(n,)`` array); if either is a full covariance the other is
    promoted to a diagonal matrix and the sum is returned as an ``(n, n)`` array.
    """
    noise_cov = np.asarray(noise_cov, dtype=float)
    model_error_cov = np.asarray(model_error_cov, dtype=float)
    if noise_cov.ndim not in (1, 2) or model_error_cov.ndim not in (1, 2):
        raise ValueError("noise_cov and model_error_cov must each be a (n,) or (n, n) array.")

    if noise_cov.ndim == 1 and model_error_cov.ndim == 1:
        if noise_cov.shape != model_error_cov.shape:
            raise ValueError(
                f"diagonal noise_cov and model_error_cov must have matching shape, "
                f"got {noise_cov.shape} and {model_error_cov.shape}."
            )
        return noise_cov + model_error_cov

    full_noise = np.diag(noise_cov) if noise_cov.ndim == 1 else noise_cov
    full_model = np.diag(model_error_cov) if model_error_cov.ndim == 1 else model_error_cov
    if full_noise.shape != full_model.shape:
        raise ValueError(
            f"noise_cov and model_error_cov describe a different number of observations: "
            f"{full_noise.shape} vs {full_model.shape}."
        )
    return full_noise + full_model
