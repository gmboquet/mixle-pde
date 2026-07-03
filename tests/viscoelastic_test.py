"""Tests for the 1D constant-Q viscoacoustic GSLS stepper: attenuation matches the analytic Q model.

The physics that defines a viscoacoustic medium is that a plane wave loses amplitude with distance as
``exp(-omega x / (2 Q c))`` and that a well-designed generalized standard linear solid keeps the quality
factor ``Q`` nearly frequency independent over the design band (constant-Q), unlike a single relaxation
mechanism whose ``Q(omega)`` is sharply peaked.

We measure ``Q`` the way field seismology does: the spectral-ratio method. The amplitude spectra at two
receivers a distance ``d`` apart obey ``ln(A2/A1) = -omega d / (2 Q c) + const``, so a straight-line fit of
``ln(A2/A1)`` against ``omega`` has slope ``-d / (2 Q c)``, from which ``Q`` is read off. Fitting that slope
over three narrow sub-bands recovers ``Q`` at three frequencies and checks it is constant.
"""

import unittest

import numpy as np

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

if HAS_TORCH:
    from mixle_pde.ops import make_ops
    from mixle_pde.viscoelastic import ViscoacousticWave1D, q_of_omega, tau_fit


def _run_two_receiver(q0=30.0, band=(10.0, 60.0), n_mech=3, fc=25.0):
    """Propagate a Ricker pulse; return spectra and geometry for spectral-ratio Q recovery."""
    ops = make_ops()
    c, rho = 2000.0, 1000.0
    nx, dx = 2000, 2.0
    dt = 0.4 * dx / c  # under the Courant limit dx / c
    nt = 9000

    m = ViscoacousticWave1D(
        nx,
        dt=dt,
        spacing=dx,
        c=c,
        rho=rho,
        q0=q0,
        band=band,
        n_mech=n_mech,
        absorb_width=40,
        absorb_strength=4.0,
    )
    src = m.ricker_source(100, fc, amplitude=1e6)
    rx1, rx2 = 400, 1200  # separation 800 nodes = 1600 m

    state = m.zeros(ops)
    tr1, tr2 = [], []
    for it in range(nt):
        state = m.step(state, ops, source=src(it * dt))
        sigma = m.stress(state)
        tr1.append(float(sigma[rx1]))
        tr2.append(float(sigma[rx2]))
    tr1 = np.asarray(tr1)
    tr2 = np.asarray(tr2)

    freqs = np.fft.rfftfreq(nt, dt)
    f1 = np.abs(np.fft.rfft(tr1))
    f2 = np.abs(np.fft.rfft(tr2))
    dist = (rx2 - rx1) * dx
    return dict(freqs=freqs, f1=f1, f2=f2, dist=dist, c=c, m=m, finite=bool(np.isfinite(tr2).all()))


def _spectral_ratio_q(res, flo, fhi):
    """Recover Q from the slope of ln(A2/A1) vs omega over [flo, fhi]."""
    freqs, f1, f2 = res["freqs"], res["f1"], res["f2"]
    band = (freqs >= flo) & (freqs <= fhi) & (f1 > 1e-4 * f1.max())
    w = 2.0 * np.pi * freqs[band]
    lr = np.log(f2[band] / f1[band])
    slope = np.polyfit(w, lr, 1)[0]
    return -res["dist"] / (2.0 * slope * res["c"])


@unittest.skipUnless(HAS_TORCH, "requires PyTorch")
class ViscoacousticAttenuationTestCase(unittest.TestCase):
    """A propagated pulse decays as exp(-omega x / (2 Q c)); recovered Q matches the target."""

    def setUp(self):
        torch.set_default_dtype(torch.float64)

    def test_amplitude_attenuation_recovers_q0(self):
        q0 = 30.0
        res = _run_two_receiver(q0=q0)
        self.assertTrue(res["finite"])
        # amplitude actually decayed between the two receivers
        self.assertLess(res["f2"].max(), res["f1"].max())
        q_meas = _spectral_ratio_q(res, 12.0, 55.0)
        rel_err = abs(q_meas - q0) / q0
        self.assertLess(rel_err, 0.15, f"spectral-ratio Q {q_meas:.2f} vs target {q0} (err {rel_err:.3f})")


@unittest.skipUnless(HAS_TORCH, "requires PyTorch")
class ConstantQTestCase(unittest.TestCase):
    """The GSLS fit gives near-frequency-independent Q across the band (constant-Q)."""

    def setUp(self):
        torch.set_default_dtype(torch.float64)

    def test_q_constant_across_band(self):
        res = _run_two_receiver(q0=30.0, n_mech=3)
        self.assertTrue(res["finite"])
        qs = [_spectral_ratio_q(res, lo, hi) for (lo, hi) in [(12.0, 20.0), (22.0, 30.0), (40.0, 52.0)]]
        for q in qs:
            self.assertGreater(q, 0.0)
        spread = (max(qs) - min(qs)) / np.mean(qs)
        self.assertLess(spread, 0.20, f"Q across band {np.round(qs, 2)} spread {spread:.3f} (want < 0.20)")

    def test_multi_mechanism_flatter_than_single(self):
        """The exact GSLS Q(omega) is far flatter for L=3 than the peaked single mechanism (L=1)."""
        f = np.array([12.0, 25.0, 55.0])
        w = 2.0 * np.pi * f

        def spread(n_mech):
            a, omega_l = tau_fit(30.0, 10.0, 60.0, n_mech)
            q = q_of_omega(a, omega_l, w)
            return (q.max() - q.min()) / q.mean(), q

        s1, q1 = spread(1)
        s3, q3 = spread(3)
        self.assertLess(s3, 0.10, f"L=3 design Q {np.round(q3, 2)} spread {s3:.3f}")
        self.assertGreater(s1, 0.30, f"L=1 design Q {np.round(q1, 2)} spread {s1:.3f}")
        self.assertLess(s3, s1)  # more mechanisms flatten Q(omega)


@unittest.skipUnless(HAS_TORCH, "requires PyTorch")
class DifferentiableTestCase(unittest.TestCase):
    """The stepper carries autograd gradients to the relaxation weights (Q is invertible)."""

    def setUp(self):
        torch.set_default_dtype(torch.float64)

    def test_gradient_flows_to_weights(self):
        ops = make_ops()
        c, rho, nx, dx = 2000.0, 1000.0, 300, 2.0
        dt = 0.4 * dx / c
        m = ViscoacousticWave1D(nx, dt=dt, spacing=dx, c=c, rho=rho, q0=30.0, band=(10.0, 60.0), n_mech=3)
        a = torch.tensor(m.a, requires_grad=True)
        src = m.ricker_source(50, 25.0, amplitude=1e6)
        rx = 150  # 100 nodes = 200 m from the source; the pulse must arrive for a nonzero gradient
        state = m.zeros(ops)
        energy = ops.zeros(1)[0]
        for it in range(320):
            state = m.step(state, ops, source=src(it * dt), a_override=a)
            if it >= 180:  # accumulate once the pulse has reached the receiver (avoids a zero-crossing loss)
                energy = energy + m.stress(state)[rx] ** 2
        loss = energy
        loss.backward()
        self.assertIsNotNone(a.grad)
        self.assertTrue(torch.isfinite(a.grad).all())
        self.assertGreater(float(a.grad.abs().sum()), 0.0)  # attenuation depends on the weights


if __name__ == "__main__":
    unittest.main()
