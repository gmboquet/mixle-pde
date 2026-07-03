"""Cycle-skip-robust FWI misfit functionals: differentiable objectives that replace the plain L2 residual for
wave-equation inversion (seismic full-waveform inversion and ultrasonic NDE).

The trouble with L2. Fitting a predicted seismogram to an observed one by ``0.5 ||pred - obs||^2`` is the
maximum-likelihood choice under white Gaussian noise, but it is a terrible objective for an oscillatory signal
whose *phase* is what carries the velocity information. When the model is wrong enough that the predicted
wiggle is more than half a period out of step with the data, the L2 misfit prefers to line up the *next*
cycle rather than slide the whole wavelet back to the true one. The objective grows non-convex, riddled with
local minima one period apart, and a gradient descent falls into the nearest wrong cycle. This is cycle
skipping, the central failure mode of FWI.

The cure is to compare the traces through a functional whose landscape stays single-basined as one trace
slides past the other. Three classic families do this, and all three are implemented here, each fully
differentiable so they drop straight into a gradient-based inversion:

* **Envelope** (``envelope_misfit``): compare the instantaneous amplitude ``|analytic signal|`` instead of the
  raw wiggle. The envelope of a shifted wavelet is just the shifted envelope with no internal oscillation, so
  its mismatch is a smooth bowl in the shift. The analytic signal is formed by the FFT Hilbert transform.

* **Cross-correlation traveltime** (``xcorr_traveltime_misfit``): the single number that best explains the
  data is the time shift that maximizes the cross-correlation of pred and obs; penalize its square. The lag of
  the correlation peak is a monotone, sign-correct function of the true shift over a wide range, so this is the
  most cycle-skip-immune of the three. The argmax is made differentiable by a parabolic (soft) interpolation of
  the three samples around the discrete peak, giving a sub-sample, autograd-capable shift.

* **Optimal transport / Wasserstein** (``wasserstein1d_misfit``): treat each (nonneg-normalized) trace as a 1D
  probability distribution and measure the transport cost between them. In 1D the squared-Wasserstein distance
  is the L2 distance between quantile functions, i.e. matching CDFs, which for a shifted mass is exactly the
  shift -- convex in the shift by construction.

Everything is a differentiable torch function; :func:`misfit` dispatches by ``kind`` in
``{"l2", "envelope", "xcorr", "w1"}``. The L2 case is kept so the failure it suffers is measurable against the
three robust ones on the same footing.
"""

from __future__ import annotations

__all__ = [
    "hilbert_envelope",
    "envelope_misfit",
    "xcorr_traveltime_misfit",
    "wasserstein1d_misfit",
    "l2_misfit",
    "misfit",
]


def _torch():
    import torch

    return torch


def _as1d(x, torch):
    t = torch.as_tensor(x, dtype=torch.float64) if not torch.is_tensor(x) else x
    return t.reshape(-1)


def hilbert_envelope(trace):
    """Instantaneous amplitude ``|z(t)|`` of a real ``trace``, where ``z`` is its analytic signal.

    The analytic signal ``z = trace + i H[trace]`` is built with the FFT Hilbert transform: forward FFT, zero
    the negative frequencies and double the positive ones (Nyquist and DC kept once), inverse FFT. Its modulus
    is the envelope, a smooth, non-oscillatory amplitude that follows the wave packet. Differentiable through
    ``torch.fft`` and a soft ``sqrt(|z|^2 + eps)``.
    """
    torch = _torch()
    x = _as1d(trace, torch)
    n = x.shape[0]
    z = torch.fft.fft(x.to(torch.complex128))
    h = torch.zeros(n, dtype=torch.float64)
    if n % 2 == 0:
        h[0] = 1.0
        h[n // 2] = 1.0
        h[1 : n // 2] = 2.0
    else:
        h[0] = 1.0
        h[1 : (n + 1) // 2] = 2.0
    analytic = torch.fft.ifft(z * h.to(torch.complex128))
    # |z| with a floor so the gradient is finite where the envelope touches zero
    return torch.sqrt(analytic.real**2 + analytic.imag**2 + 1e-30)


def envelope_misfit(pred, obs):
    """Squared L2 distance between the Hilbert envelopes of ``pred`` and ``obs``.

    Because the envelope of a time-shifted wavelet is the shifted envelope (no carrier oscillation), the map
    from a rigid time shift to this misfit is a single smooth basin with its minimum at zero shift, so it does
    not cycle-skip.
    """
    torch = _torch()
    ep = hilbert_envelope(pred)
    eo = hilbert_envelope(obs)
    return torch.sum((ep - eo) ** 2)


def _parabolic_peak_offset(y_m1, y_0, y_p1):
    """Sub-sample offset of a parabola through three consecutive samples with the middle the discrete max.

    For samples at lags ``-1, 0, +1`` with values ``y_m1, y_0, y_p1`` the vertex is at
    ``delta = 0.5 (y_m1 - y_p1) / (y_m1 - 2 y_0 + y_p1)`` in units of the lag step. Differentiable and, near a
    correlation peak (concave down), ``|delta| <= 0.5``.
    """
    denom = y_m1 - 2.0 * y_0 + y_p1
    return 0.5 * (y_m1 - y_p1) / (denom - 1e-30)


def xcorr_traveltime_misfit(pred, obs, *, dt: float = 1.0):
    """Squared cross-correlation traveltime shift between ``pred`` and ``obs``.

    Cross-correlate the two traces, locate the correlation maximum, and refine it to sub-sample precision with a
    parabolic fit of the three samples around the discrete peak. The resulting lag ``tau`` (in time units, via
    ``dt``) is the best single time shift aligning pred to obs; the misfit is ``tau^2``, minimized at zero when
    the traces already align. The lag of the correlation peak tracks the true shift monotonically over a wide
    range, which is why this objective is the most robust to cycle skipping.

    The argmax over discrete lags is not differentiable, but its *value* need not flow gradients: the misfit
    ``tau^2`` is differentiable through the parabolic-interpolation formula, whose three input correlation
    samples depend smoothly on ``pred``. That is enough to descend a time-shift parameter.
    """
    torch = _torch()
    p = _as1d(pred, torch)
    o = _as1d(obs, torch)
    n = p.shape[0]
    # full linear cross-correlation r[l] = sum_t p[t] o[t - l] via FFT (zero-padded to avoid wraparound).
    nfft = 1
    while nfft < 2 * n:
        nfft *= 2
    fp = torch.fft.rfft(p, nfft)
    fo = torch.fft.rfft(o, nfft)
    corr_full = torch.fft.irfft(fp * torch.conj(fo), nfft)
    # arrange lags from -(n-1) .. (n-1). With r[l] = sum_t p[t] o[t-l], the circular correlation puts a positive
    # lag l at index l (in corr_full[0..n-1]) and a negative lag -j at index nfft-j (the tail), so lags -(n-1)..-1
    # sit contiguously at corr_full[nfft-n+1 : nfft] already in increasing-lag order (no flip).
    pos = corr_full[:n]
    neg = corr_full[nfft - n + 1 : nfft]
    corr = torch.cat([neg, pos])  # length 2n-1, index k -> lag = k - (n-1)
    lags = torch.arange(-(n - 1), n, dtype=torch.float64)
    kmax = int(torch.argmax(corr.detach()).item())
    kmax = min(max(kmax, 1), corr.shape[0] - 2)
    delta = _parabolic_peak_offset(corr[kmax - 1], corr[kmax], corr[kmax + 1])
    tau = (lags[kmax] + delta) * dt
    return tau**2


def _to_density(x, torch):
    """Turn a real trace into a nonneg probability density on the samples: shift to nonneg, then normalize.

    A signed seismic trace is not a distribution. The standard OT-for-FWI transform makes it one by mapping
    amplitudes through a positive function (here a shift by the minimum plus a small floor) and normalizing to
    unit mass, so CDF matching is well defined. Differentiable.
    """
    x = _as1d(x, torch)
    lo = torch.min(x)
    pos = x - lo + 1e-12
    return pos / torch.sum(pos)


def wasserstein1d_misfit(pred, obs, *, dt: float = 1.0):
    """Squared 1D Wasserstein-2 distance between ``pred`` and ``obs`` as nonneg-normalized densities.

    Each trace is turned into a probability density over its sample times (:func:`_to_density`), and the
    optimal-transport cost between them is computed in closed form: in 1D the squared W2 distance is the
    integrated squared difference of the inverse CDFs (quantile functions). Equivalently, matching CDFs. For a
    density rigidly shifted by ``s`` this equals ``s^2``, so the objective is convex in a time shift and cannot
    cycle-skip. Fully differentiable (cumulative sums and an interpolation of quantiles).
    """
    torch = _torch()
    dp = _to_density(pred, torch)
    do = _to_density(obs, torch)
    n = dp.shape[0]
    t = torch.arange(n, dtype=torch.float64) * dt
    cdf_p = torch.cumsum(dp, dim=0)
    cdf_o = torch.cumsum(do, dim=0)
    # W2^2 = integral_0^1 (Qp(u) - Qo(u))^2 du, with Q the quantile function.  Evaluate on a common grid of u.
    m = max(2 * n, 256)
    u = (torch.arange(m, dtype=torch.float64) + 0.5) / m
    qp = _quantile(t, cdf_p, u, torch)
    qo = _quantile(t, cdf_o, u, torch)
    return torch.mean((qp - qo) ** 2)


def _quantile(t, cdf, u, torch):
    """Quantile function ``Q(u) = inf{ t : CDF(t) >= u }`` evaluated at probabilities ``u``, linearly
    interpolated in the CDF for differentiability. ``t`` are the support points, ``cdf`` the (increasing)
    cumulative masses at those points."""
    # searchsorted on the (detached) CDF gives the bracket; the interpolation weight carries the gradient.
    idx = torch.searchsorted(cdf.detach(), u)
    idx = torch.clamp(idx, 1, cdf.shape[0] - 1)
    c_lo = cdf[idx - 1]
    c_hi = cdf[idx]
    t_lo = t[idx - 1]
    t_hi = t[idx]
    w = (u - c_lo) / (c_hi - c_lo + 1e-30)
    w = torch.clamp(w, 0.0, 1.0)
    return t_lo + w * (t_hi - t_lo)


def l2_misfit(pred, obs):
    """Plain squared-L2 residual ``||pred - obs||^2`` -- the objective that cycle-skips, kept for comparison."""
    torch = _torch()
    p = _as1d(pred, torch)
    o = _as1d(obs, torch)
    return torch.sum((p - o) ** 2)


_KINDS = {
    "l2": l2_misfit,
    "envelope": envelope_misfit,
    "xcorr": xcorr_traveltime_misfit,
    "w1": wasserstein1d_misfit,
    "w2": wasserstein1d_misfit,
}


def misfit(pred, obs, *, kind: str = "l2", **kw):
    """Dispatch to a misfit functional by name: ``kind`` in ``{l2, envelope, xcorr, w1}`` (``w2`` aliases
    ``w1``). Extra keywords (e.g. ``dt``) forward to the chosen functional."""
    try:
        fn = _KINDS[kind]
    except KeyError as e:
        raise ValueError(f"unknown misfit kind {kind!r}; choose from {sorted(_KINDS)}.") from e
    if kind in ("l2", "envelope"):
        return fn(pred, obs)
    return fn(pred, obs, **kw)
