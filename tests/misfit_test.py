"""Tests for cycle-skip-robust FWI misfit functionals.

The acceptance bar is the cycle-skipping property itself, asserted against a Ricker wavelet scanned over a
rigid time shift of +-1.5 dominant periods:

  1. the plain L2 misfit is non-convex, with a spurious local minimum at a nonzero lag (cycle skipping);
  2. the envelope, cross-correlation-traveltime, and 1D-Wasserstein misfits are single-basined over the same
     range -- exactly one local minimum, located at zero lag, with the global minimum there;
  3. the misfit gradient with respect to a scalar time-shift parameter is finite and points toward zero lag.
"""

import importlib.util
import unittest

import numpy as np

HAS_TORCH = importlib.util.find_spec("torch") is not None

if HAS_TORCH:
    import torch

    from mixle_pde.misfit import (
        envelope_misfit,
        hilbert_envelope,
        l2_misfit,
        misfit,
        wasserstein1d_misfit,
        xcorr_traveltime_misfit,
    )

# a Ricker wavelet: dominant frequency f0 -> dominant period T0 = 1/f0
F0 = 25.0
T0 = 1.0 / F0
FS = 500.0
DT = 1.0 / FS


def _time_grid():
    return np.arange(-0.2, 0.2, DT)


def _ricker_np(shift):
    """Ricker wavelet centered at time ``shift`` on the fixed grid (numpy)."""
    t = _time_grid()
    a = (np.pi * F0 * (t - shift)) ** 2
    return (1.0 - 2.0 * a) * np.exp(-a)


def _local_minima(values):
    """Indices i (interior) with values[i] strictly less than both neighbors."""
    return [i for i in range(1, len(values) - 1) if values[i] < values[i - 1] and values[i] < values[i + 1]]


@unittest.skipUnless(HAS_TORCH, "requires PyTorch")
class HilbertEnvelopeTestCase(unittest.TestCase):
    def setUp(self):
        torch.set_default_dtype(torch.float64)

    def test_envelope_of_cosine_is_constant_unit_amplitude(self):
        # analytic envelope of A cos(w t) is exactly A (away from edges): a sharp closed-form check.
        n = 512
        t = np.arange(n) / n
        x = 1.7 * np.cos(2 * np.pi * 20 * t)
        env = hilbert_envelope(torch.as_tensor(x)).numpy()
        interior = env[n // 8 : -n // 8]
        self.assertLess(float(np.abs(interior - 1.7).max()), 1e-3)

    def test_envelope_is_nonnegative_and_smooth_for_ricker(self):
        env = hilbert_envelope(torch.as_tensor(_ricker_np(0.0))).numpy()
        self.assertGreaterEqual(float(env.min()), 0.0)
        # the raw Ricker has three lobes (two sign changes); its envelope is single-peaked (no interior minima)
        self.assertEqual(_local_minima(env), [])


@unittest.skipUnless(HAS_TORCH, "requires PyTorch")
class CycleSkippingTestCase(unittest.TestCase):
    """The whole point: scan a shifted Ricker over +-1.5 periods and characterize each misfit's landscape."""

    def setUp(self):
        torch.set_default_dtype(torch.float64)
        self.lags = np.linspace(-1.5 * T0, 1.5 * T0, 61)
        self.zero_idx = int(np.argmin(np.abs(self.lags)))
        self.obs = torch.as_tensor(_ricker_np(0.0))

    def _scan(self, fn, **kw):
        vals = []
        for ls in self.lags:
            pred = torch.as_tensor(_ricker_np(ls))
            vals.append(float(fn(pred, self.obs, **kw)))
        return np.array(vals)

    def test_l2_has_spurious_local_minimum_at_nonzero_lag(self):
        # L2 cycle-skips: the landscape is non-convex with local minima one period from the true (zero) lag.
        vals = self._scan(l2_misfit)
        minima = _local_minima(vals)
        self.assertIn(self.zero_idx, minima)  # the true minimum is there
        spurious = [i for i in minima if i != self.zero_idx]
        self.assertTrue(spurious, "expected L2 to cycle-skip (a local minimum away from zero lag)")
        # a spurious minimum sits roughly a full period away from zero lag
        for i in spurious:
            self.assertGreater(abs(self.lags[i]), 0.5 * T0)

    def test_envelope_is_single_basined_with_argmin_at_zero_lag(self):
        vals = self._scan(envelope_misfit)
        self.assertEqual(_local_minima(vals), [self.zero_idx])  # exactly one basin, at zero lag
        self.assertEqual(int(np.argmin(vals)), self.zero_idx)  # global minimum at zero lag

    def test_xcorr_traveltime_is_single_basined_with_argmin_at_zero_lag(self):
        vals = self._scan(xcorr_traveltime_misfit, dt=DT)
        self.assertEqual(_local_minima(vals), [self.zero_idx])
        self.assertEqual(int(np.argmin(vals)), self.zero_idx)

    def test_wasserstein_is_single_basined_with_argmin_at_zero_lag(self):
        vals = self._scan(wasserstein1d_misfit, dt=DT)
        self.assertEqual(_local_minima(vals), [self.zero_idx])
        self.assertEqual(int(np.argmin(vals)), self.zero_idx)

    def test_envelope_and_xcorr_are_strictly_monotone_over_full_range(self):
        # envelope and cross-correlation traveltime are strictly single-basined over the entire +-1.5 T0 scan:
        # each arm decreases toward zero lag and increases away from it, with no flat spots.
        zi = self.zero_idx
        for kind, kw in [("envelope", {}), ("xcorr", {"dt": DT})]:
            vals = self._scan(lambda p, o, _kind=kind, **k: misfit(p, o, kind=_kind, **k), **kw)
            left = vals[: zi + 1]
            right = vals[zi:]
            self.assertTrue(np.all(np.diff(left) < 0), f"{kind}: left arm not decreasing toward zero lag")
            self.assertTrue(np.all(np.diff(right) > 0), f"{kind}: right arm not increasing after zero lag")

    def test_wasserstein_is_strictly_monotone_within_a_half_period(self):
        # For a compact wavelet the transport cost saturates once the packet has slid a full lobe past, so W is
        # strictly single-basined within the physically relevant core (|lag| <= half a dominant period), where
        # its value equals the transport distance and is the strongest cure for cycle skipping.
        vals = self._scan(wasserstein1d_misfit, dt=DT)
        core = np.abs(self.lags) <= 0.5 * T0 + 1e-12
        idx = np.where(core)[0]
        czi = idx[int(np.argmin(np.abs(self.lags[idx])))] - idx[0]
        core_vals = vals[idx]
        left = core_vals[: czi + 1]
        right = core_vals[czi:]
        self.assertTrue(np.all(np.diff(left) < 0), "W: left arm not decreasing toward zero lag in core")
        self.assertTrue(np.all(np.diff(right) > 0), "W: right arm not increasing after zero lag in core")


@unittest.skipUnless(HAS_TORCH, "requires PyTorch")
class DifferentiabilityTestCase(unittest.TestCase):
    """The gradient wrt a scalar time-shift parameter is finite and descends toward zero lag."""

    def setUp(self):
        torch.set_default_dtype(torch.float64)
        self.t = torch.arange(-0.2, 0.2, DT, dtype=torch.float64)
        self.obs = self._ricker(torch.tensor(0.0))

    def _ricker(self, shift):
        a = (np.pi * F0 * (self.t - shift)) ** 2
        return (1.0 - 2.0 * a) * torch.exp(-a)

    def _grad_at(self, kind, shift0, **kw):
        s = torch.tensor(shift0, dtype=torch.float64, requires_grad=True)
        pred = self._ricker(s)
        loss = misfit(pred, self.obs, kind=kind, **kw)
        loss.backward()
        return float(s.grad)

    def test_gradients_are_finite_and_point_toward_zero_lag(self):
        # a positive predicted shift: descent must reduce it, so dL/dshift > 0 at shift0 > 0.
        shift0 = 0.4 * T0
        for kind, kw in [("envelope", {}), ("xcorr", {"dt": DT}), ("w1", {"dt": DT})]:
            g_pos = self._grad_at(kind, shift0, **kw)
            self.assertTrue(np.isfinite(g_pos), f"{kind}: gradient not finite")
            self.assertGreater(g_pos, 0.0, f"{kind}: gradient does not descend toward zero lag from the right")
            # symmetric check on the other side: a negative shift has a negative gradient
            g_neg = self._grad_at(kind, -shift0, **kw)
            self.assertTrue(np.isfinite(g_neg))
            self.assertLess(g_neg, 0.0, f"{kind}: gradient does not descend toward zero lag from the left")

    def test_misfit_dispatch_matches_direct_calls(self):
        pred = self._ricker(torch.tensor(0.3 * T0))
        self.assertAlmostEqual(float(misfit(pred, self.obs, kind="l2")), float(l2_misfit(pred, self.obs)))
        self.assertAlmostEqual(float(misfit(pred, self.obs, kind="envelope")), float(envelope_misfit(pred, self.obs)))
        self.assertAlmostEqual(
            float(misfit(pred, self.obs, kind="xcorr", dt=DT)),
            float(xcorr_traveltime_misfit(pred, self.obs, dt=DT)),
        )
        self.assertAlmostEqual(
            float(misfit(pred, self.obs, kind="w1", dt=DT)),
            float(wasserstein1d_misfit(pred, self.obs, dt=DT)),
        )
        with self.assertRaises(ValueError):
            misfit(pred, self.obs, kind="nope")


if __name__ == "__main__":
    unittest.main()
