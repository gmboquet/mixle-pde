"""Differential with a cycle-skip-robust misfit: the L2 data term is multi-modal in a time shift while the
envelope / Wasserstein objectives stay single-basined, and gradient descent recovers a cycle-skipped shift
that L2 cannot."""

import unittest

import numpy as np
import torch
from mixle.ppl import free, joint
from scipy.signal import find_peaks

from mixle_pde import Differential
from mixle_pde.inverse import _DifferentialProxy

F0 = 20.0  # Hz, an oscillatory Gabor wavelet (several cycles) so L2 genuinely cycle-skips
SIGMA = 0.08  # s, gaussian envelope width
DT = 0.004
NT = 320
TC = 0.7  # wavelet center at zero shift
TVEC = np.arange(NT) * DT
PERIOD = 1.0 / F0  # 0.05 s
TRUE_SHIFT = 1.5 * PERIOD  # 0.075 s, well past half a period -> L2 skips a cycle from a zero start


def _gabor(shift):
    """A gaussian-windowed cosine centered at TC + shift, differentiable in the torch scalar ``shift``."""
    t = torch.as_tensor(TVEC, dtype=torch.float64)
    tau = t - TC - shift
    return torch.exp(-(tau**2) / (2.0 * SIGMA**2)) * torch.cos(2.0 * np.pi * F0 * tau)


def _forward(p, ops):
    s = p.shift
    if not torch.is_tensor(s):
        s = torch.as_tensor(np.asarray(s, dtype=float))
    return _gabor(s.reshape(()))


OBS = _gabor(torch.tensor(float(TRUE_SHIFT), dtype=torch.float64)).detach().numpy()


def _loglik_at(proxy, shift):
    params = {"shift": torch.tensor([float(shift)], dtype=torch.float64)}
    with torch.no_grad():
        return float(proxy.loglik(None, params, torch))


def _significant_peaks(vals):
    """Local maxima with real prominence, so float ripple on the flat tails is not counted."""
    vals = np.asarray(vals, dtype=float)
    prom = 0.05 * (vals.max() - vals.min())
    peaks, _ = find_peaks(vals, prominence=prom)
    return peaks


class MisfitInverseTest(unittest.TestCase):
    def _proxy(self, misfit):
        _field, proxy = Differential(OBS, forward=_forward, drivers=[free(1, name="shift")], scale=0.1, misfit=misfit)
        return proxy

    def test_l2_is_multimodal_but_robust_misfits_are_single_basined(self):
        wide = np.linspace(TRUE_SHIFT - 2.0 * PERIOD, TRUE_SHIFT + 2.0 * PERIOD, 161)
        step = wide[1] - wide[0]
        l2 = np.array([_loglik_at(self._proxy(None), s) for s in wide])
        # L2 has spurious local maxima one period apart (the cycle-skip), but its global max is still the truth.
        self.assertGreaterEqual(len(_significant_peaks(l2)), 3)
        self.assertLess(abs(wide[int(np.argmax(l2))] - TRUE_SHIFT), 1.5 * step)
        # the envelope misfit (a gaussian bowl) is single-basined across the whole cycle-skip range
        env = np.array([_loglik_at(self._proxy("envelope"), s) for s in wide])
        self.assertEqual(len(_significant_peaks(env)), 1)
        self.assertLess(abs(wide[int(np.argmax(env))] - TRUE_SHIFT), 1.5 * step)
        # the Wasserstein misfit is convex over its core window (its guaranteed regime)
        core = np.linspace(TRUE_SHIFT - 0.6 * PERIOD, TRUE_SHIFT + 0.6 * PERIOD, 61)
        cstep = core[1] - core[0]
        w1 = np.array([_loglik_at(self._proxy("w1"), s) for s in core])
        self.assertEqual(len(_significant_peaks(w1)), 1)
        self.assertLess(abs(core[int(np.argmax(w1))] - TRUE_SHIFT), 1.5 * cstep)

    def test_misfit_gradient_escapes_the_cycle_skip(self):
        # At one period below the truth (an L2 trap) the envelope-misfit gradient still points toward the truth.
        params = {"shift": torch.tensor([TRUE_SHIFT - PERIOD], dtype=torch.float64, requires_grad=True)}
        ll = self._proxy("envelope").loglik(None, params, torch)
        ll.backward()
        self.assertGreater(float(params["shift"].grad.item()), 0.0)

    def test_gradient_descent_recovers_where_l2_is_stuck(self):
        def optimize(misfit_name, steps=800, lr=0.003):
            s = torch.tensor([0.0], dtype=torch.float64, requires_grad=True)
            proxy = self._proxy(misfit_name)
            opt = torch.optim.Adam([s], lr=lr)
            for _ in range(steps):
                opt.zero_grad()
                loss = -proxy.loglik(None, {"shift": s}, torch)
                loss.backward()
                opt.step()
            return float(s.item())

        self.assertLess(abs(optimize("envelope") - TRUE_SHIFT), 0.2 * PERIOD)  # robust misfit recovers it
        self.assertGreater(abs(optimize(None) - TRUE_SHIFT), 0.5 * PERIOD)  # L2 is trapped a cycle away

    def test_fit_plumbing_accepts_misfit(self):
        # the joint()/fit path runs end to end with a misfit and returns a finite estimate
        obs = Differential(OBS, forward=_forward, drivers=[free(1, name="shift")], scale=0.1, misfit="envelope")
        mean, _sd = joint([obs]).fit(how="laplace").posterior("shift")
        self.assertTrue(np.isfinite(float(np.asarray(mean))))

    def test_misfit_changes_the_score(self):
        s = TRUE_SHIFT - 0.5 * PERIOD
        self.assertNotAlmostEqual(_loglik_at(self._proxy(None), s), _loglik_at(self._proxy("envelope"), s), places=3)

    def test_guards(self):
        # a scalar misfit has no residual vector, so Gauss-Newton is disabled
        self.assertIsNone(self._proxy("envelope").residual(None, {"shift": torch.zeros(1)}, torch))
        self.assertIsNotNone(self._proxy(None).residual(None, {"shift": torch.zeros(1)}, torch))
        # misfit is gaussian-only and real-only
        with self.assertRaises(ValueError):
            Differential(np.ones(4), forward=_forward, drivers=[free(1, name="shift")], family="poisson", misfit="w1")
        with self.assertRaises(ValueError):
            _DifferentialProxy(
                np.ones(4) + 1j,
                forward=_forward,
                observe=None,
                drivers=[],
                over_name=None,
                scale=1.0,
                family="gaussian",
                misfit_fn=lambda a, b: a,
            )


if __name__ == "__main__":
    unittest.main()
