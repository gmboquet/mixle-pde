"""Single-gather seismic processing and well-to-seismic calibration.

These sit alongside the RTM imaging in :mod:`mixle_pde.migration` as the conventional (non-migration)
processing chain applied to a recorded gather before or instead of imaging: :func:`nmo_correction` flattens
the reflection hyperbola recorded across offsets onto the zero-offset (vertical) travel time using the
standard hyperbolic moveout equation, :func:`stack` collapses a corrected common-midpoint gather to one
trace, and :func:`well_tie` cross-correlates a synthetic seismogram (built from a well log's reflectivity
series) against a recorded trace to find the checkshot time-depth shift -- the standard well-to-seismic
calibration step that ties borehole depth control to the seismic time domain. Real shot/receiver/well
geometry is expected to arrive via B2's SEG-Y ingest and IC-4 ``SurveyGeometry``; these functions operate on
plain arrays so they work equally on synthetic or real gathers/logs.
"""

from __future__ import annotations

import numpy as np

__all__ = ["nmo_correction", "stack", "well_tie"]


def nmo_correction(gather: np.ndarray, offsets: np.ndarray, t0: np.ndarray, vnmo) -> np.ndarray:
    """Hyperbolic normal-moveout correction: flattens reflections onto the zero-offset time.

    ``gather`` is ``(n_samples, n_traces)`` recorded on the ``t0`` time axis (length ``n_samples``);
    ``offsets`` is the ``(n_traces,)`` source-receiver offset of each trace. ``vnmo`` is the NMO velocity --
    a scalar or a ``(n_samples,)`` array giving the velocity at each zero-offset time. For each output
    sample ``t0[i]`` and trace ``j`` the observed hyperbolic traveltime is

        t_obs = sqrt(t0[i]**2 + offsets[j]**2 / vnmo[i]**2)

    and the corrected amplitude is ``gather[:, j]`` linearly interpolated at ``t_obs`` (mapping the energy
    that arrived late, at the far offset, back onto the near-offset zero-offset time). Samples whose
    ``t_obs`` falls beyond the recorded window are muted (left at zero) -- the standard NMO stretch mute at
    far offset / shallow time.
    """
    gather = np.asarray(gather, dtype=float)
    offsets = np.asarray(offsets, dtype=float).reshape(-1)
    t0 = np.asarray(t0, dtype=float).reshape(-1)
    if gather.ndim != 2:
        raise ValueError("gather must be 2-D (n_samples, n_traces)")
    n_samples, n_traces = gather.shape
    if offsets.shape[0] != n_traces:
        raise ValueError("offsets length must match the gather's trace count")
    if t0.shape[0] != n_samples:
        raise ValueError("t0 length must match the gather's sample count")
    vnmo_arr = np.broadcast_to(np.asarray(vnmo, dtype=float), t0.shape).astype(float)

    corrected = np.zeros_like(gather)
    t_max = t0[-1]
    for j in range(n_traces):
        t_obs = np.sqrt(t0**2 + (offsets[j] ** 2) / vnmo_arr**2)
        valid = t_obs <= t_max
        corrected[valid, j] = np.interp(t_obs[valid], t0, gather[:, j])
    return corrected


def stack(gathers: np.ndarray) -> np.ndarray:
    """Stack a (typically NMO-corrected) common-midpoint gather to one trace: the offset-axis mean.

    ``gathers`` is ``(n_samples, n_traces)``; returns the ``(n_samples,)`` stacked trace. A 1-D input
    (already a single trace) passes through unchanged, so ``stack`` composes with itself.
    """
    g = np.asarray(gathers, dtype=float)
    return g.copy() if g.ndim == 1 else g.mean(axis=-1)


def _shift_zero_fill(x: np.ndarray, lag: int) -> np.ndarray:
    """Delay ``x`` by ``lag`` samples with zero fill (a negative ``lag`` advances it instead).

    Zero-fill rather than a circular ``np.roll`` -- a checkshot mis-tie shift should not wrap the trace's
    tail around onto its head.
    """
    n = x.shape[0]
    out = np.zeros_like(x)
    if lag >= 0:
        if lag < n:
            out[lag:] = x[: n - lag]
    else:
        adv = -lag
        if adv < n:
            out[: n - adv] = x[adv:]
    return out


def _default_wavelet(n: int, *, f0: float = 25.0, dt: float = 0.002) -> np.ndarray:
    """A short Ricker wavelet (dominant frequency ``f0`` Hz, sample rate ``dt`` s), used when the caller
    does not supply one; length is capped to stay local to each reflectivity spike and to the trace."""
    length = max(3, min(int(n), int(round(4.0 / (f0 * dt)))))
    if length % 2 == 0:
        length += 1
    t = (np.arange(length) - length // 2) * dt
    a = (np.pi * f0 * t) ** 2
    return (1.0 - 2.0 * a) * np.exp(-a)


def well_tie(
    reflectivity_log: np.ndarray,
    log_twt: np.ndarray,
    seismic_trace: np.ndarray,
    *,
    wavelet: np.ndarray | None = None,
    max_lag: int | None = None,
) -> dict:
    """Tie a well log to a seismic trace: checkshot time-depth plus the max-correlation shift.

    ``reflectivity_log`` is the reflection-coefficient series derived from the well's sonic/density logs,
    sampled at the two-way times ``log_twt`` -- the checkshot time-depth curve that maps the log's depth
    samples onto seismic time. The log is resampled onto ``seismic_trace``'s own uniform sample grid,
    convolved with ``wavelet`` (a short Ricker by default) to build a synthetic seismogram, and
    cross-correlated (zero-fill shifts, no wraparound) against ``seismic_trace`` over ``+/- max_lag`` samples
    (default: the full trace length) to find the shift that best ties the well to the seismic -- the
    residual checkshot mis-tie left after the initial time-depth conversion.

    Returns a dict with keys ``synthetic`` (the resampled/convolved synthetic trace), ``shift`` (the integer
    sample lag maximizing the normalized cross-correlation -- positive delays the synthetic), ``correlation``
    (that peak normalized correlation, in ``[-1, 1]``), and ``time_depth`` (the input ``log_twt``, echoed
    back as the time-depth curve actually used).
    """
    reflectivity_log = np.asarray(reflectivity_log, dtype=float).reshape(-1)
    log_twt = np.asarray(log_twt, dtype=float).reshape(-1)
    seismic_trace = np.asarray(seismic_trace, dtype=float).reshape(-1)
    if reflectivity_log.shape != log_twt.shape:
        raise ValueError("reflectivity_log and log_twt must have the same length")

    n = seismic_trace.shape[0]
    if reflectivity_log.shape[0] == n:
        refl_on_trace_grid = reflectivity_log
    else:
        src_idx = np.linspace(0, n - 1, num=reflectivity_log.shape[0])
        refl_on_trace_grid = np.interp(np.arange(n), src_idx, reflectivity_log)

    w = np.asarray(wavelet, dtype=float) if wavelet is not None else _default_wavelet(n)
    synthetic = np.convolve(refl_on_trace_grid, w, mode="same")

    if max_lag is None:
        max_lag = n - 1
    lags = np.arange(-int(max_lag), int(max_lag) + 1)
    b = seismic_trace - seismic_trace.mean()
    b_norm = np.linalg.norm(b)
    scores = np.zeros(lags.shape[0], dtype=float)
    for idx, lag in enumerate(lags):
        shifted = _shift_zero_fill(synthetic, int(lag))
        a = shifted - shifted.mean()
        a_norm = np.linalg.norm(a)
        if a_norm > 0.0 and b_norm > 0.0:
            scores[idx] = float(np.dot(a, b) / (a_norm * b_norm))
    best = int(np.argmax(scores))
    shift = int(lags[best])

    return {
        "synthetic": synthetic,
        "shift": shift,
        "correlation": float(scores[best]),
        "time_depth": log_twt,
    }
