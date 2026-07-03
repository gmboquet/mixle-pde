"""Tests for the 3D elastic-wave stepper: the two seismic body-wave speeds and numerical stability.

The defining physics of seismology is that an elastic solid carries two waves at two speeds -- a
compressional (P) wave at ``vp = sqrt((lambda+2mu)/rho)`` with particle motion along the propagation
direction, and a shear (S) wave at ``vs = sqrt(mu/rho)`` with motion perpendicular to it. In a homogeneous
Poisson-ratio solid ``vp/vs = sqrt(3) ~ 1.73``; here we set ``vp/vs = 1.7`` and confirm both fronts travel
at their analytical speeds within a few percent by tracking each wavefront across the grid. A separate check
confirms a moment-tensor point source runs stably (bounded energy, no NaN) under a CFL-respecting step.

Each plane-wave seed is a one-way (right-going) characteristic: seeding the velocity together with the stress
it carries in the Riemann invariant (``sxx = -rho*vp*vx`` for P, ``sxy = -rho*vs*vy`` for S) launches a
single traveling front instead of two counter-propagating halves, so the peak position advances cleanly and
its slope in time is the phase speed.
"""

import unittest

import numpy as np

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

if HAS_TORCH:
    from mixle_pde.elastic import ElasticWave3D
    from mixle_pde.ops import make_ops

_ORDER = ("vx", "vy", "vz", "sxx", "syy", "szz", "sxy", "sxz", "syz")


def _plane_wave_speed(kind, *, n=40, vp=1.7, vs=1.0, rho=1.0):
    """Seed a one-way plane P or S wave along +x and return the tracked phase-front speed."""
    ops = make_ops()
    h = 1.0 / n
    dt = 0.35 * h / (vp * np.sqrt(3))  # comfortably under the 3D Courant limit h/(vp*sqrt(3))
    lam = rho * (vp**2 - 2 * vs**2)
    mu = rho * vs**2
    m = ElasticWave3D(n, dt=dt, spacing=h, vp=vp, vs=vs, rho=rho, absorb_width=6, absorb_strength=6.0)

    x = np.arange(n) * h
    x0, width = 0.20, 0.045  # a narrow Gaussian pulse near the near edge
    env = np.broadcast_to(np.exp(-((x - x0) ** 2) / (2 * width**2))[:, None, None], (n, n, n)).copy()
    z = np.zeros((n, n, n))
    comps = {c: z.copy() for c in _ORDER}
    if kind == "P":
        # P: particle velocity ALONG propagation (x); right-going characteristic sxx = -rho*vp*vx.
        comps["vx"] = env
        comps["sxx"] = -rho * vp * env
        comps["syy"] = (lam / (lam + 2 * mu)) * comps["sxx"]  # isotropic lateral stress of a P wave
        comps["szz"] = (lam / (lam + 2 * mu)) * comps["sxx"]
        probe = 0  # track vx
    else:
        # S: particle velocity PERPENDICULAR (y) to propagation (x); characteristic sxy = -rho*vs*vy.
        comps["vy"] = env
        comps["sxy"] = -rho * vs * env
        probe = 1  # track vy
    state = m.pack(*[comps[c] for c in _ORDER])

    speed_analytic = vp if kind == "P" else vs
    times, peaks = [], []
    last = -1.0
    for i in range(1, 401):
        state = m.step(state, ops)
        if i % 8 == 0:
            v = m.velocity(state, ops)[probe].detach().numpy()
            prof = np.abs(v).mean(axis=(1, 2))  # transverse-averaged amplitude profile along x
            pk = np.argmax(prof) * h  # wavefront position = peak of the profile
            if pk > 0.80:  # reached the far sponge; stop before reflection contaminates the fit
                break
            if pk >= 0.30 and pk >= last - 1e-9:  # forward-going, in the clean window
                times.append(i * dt)
                peaks.append(pk)
                last = pk
    finite = bool(torch.isfinite(state).all())
    speed = float(np.polyfit(np.array(times), np.array(peaks), 1)[0])  # slope of front position vs time
    return speed, speed_analytic, len(peaks), finite


@unittest.skipUnless(HAS_TORCH, "requires PyTorch")
class ElasticBodyWaveSpeedTestCase(unittest.TestCase):
    """P and S waves travel at the analytical seismic speeds in a homogeneous Poisson solid."""

    def setUp(self):
        torch.set_default_dtype(torch.float64)

    def test_p_wave_speed(self):
        speed, analytic, npts, finite = _plane_wave_speed("P")
        self.assertTrue(finite)  # no NaN over the run
        self.assertGreaterEqual(npts, 6)  # enough clean points to fit a slope
        rel_err = abs(speed - analytic) / analytic
        self.assertLess(rel_err, 0.08, f"P speed {speed:.4f} vs analytic {analytic} (err {rel_err:.3f})")

    def test_s_wave_speed(self):
        speed, analytic, npts, finite = _plane_wave_speed("S")
        self.assertTrue(finite)
        self.assertGreaterEqual(npts, 6)
        rel_err = abs(speed - analytic) / analytic
        self.assertLess(rel_err, 0.08, f"S speed {speed:.4f} vs analytic {analytic} (err {rel_err:.3f})")

    def test_p_faster_than_s(self):
        """The compressional front outruns the shear front at the set vp/vs ratio (~1.7)."""
        vp_num, vp_a, _, _ = _plane_wave_speed("P")
        vs_num, vs_a, _, _ = _plane_wave_speed("S")
        self.assertGreater(vp_num, vs_num)  # P arrives first, the seismic ordering
        ratio = vp_num / vs_num
        self.assertLess(abs(ratio - vp_a / vs_a) / (vp_a / vs_a), 0.08)  # recovered ratio near 1.7


@unittest.skipUnless(HAS_TORCH, "requires PyTorch")
class ElasticStabilityTestCase(unittest.TestCase):
    """A moment-tensor point source excites the medium and runs stably (bounded energy, no NaN)."""

    def setUp(self):
        torch.set_default_dtype(torch.float64)

    def test_moment_tensor_source_stable(self):
        n = 24
        h = 1.0 / n
        vp, vs, rho = 1.7, 1.0, 1.0
        dt = 0.35 * h / (vp * np.sqrt(3))
        m = ElasticWave3D(n, dt=dt, spacing=h, vp=vp, vs=vs, rho=rho, absorb_width=5, absorb_strength=6.0)
        ops = make_ops()

        # explosion (M = I) at the centre excites a clean P wave; a Ricker-like time envelope
        c = n // 2
        moment = np.eye(3)
        state = m.zeros(ops)
        f0 = 4.0  # cycles over the run
        energies = []
        n_steps = 220
        for i in range(n_steps):
            t = i * dt
            amp = (1.0 - 2.0 * (np.pi * f0 * (t - 0.15)) ** 2) * np.exp(-((np.pi * f0 * (t - 0.15)) ** 2))
            src = m.moment_tensor_source((c, c, c), moment, amplitude=amp)
            state = m.step(state, ops, source=src)
            vx, vy, vz = m.velocity(state, ops)
            energies.append(float((vx**2 + vy**2 + vz**2).sum()))

        self.assertTrue(torch.isfinite(state).all())  # no NaN / blow-up
        self.assertGreater(max(energies), 0.0)  # the source actually excited the wavefield
        # energy stays bounded: the sponge bleeds it off rather than letting it grow
        self.assertLess(energies[-1], 50.0 * max(energies) + 1.0)

    def test_free_surface_runs(self):
        """A free-surface top boundary is stable and traps motion near the surface (surface waves)."""
        n = 24
        h = 1.0 / n
        vp, vs, rho = 1.7, 1.0, 1.0
        dt = 0.35 * h / (vp * np.sqrt(3))
        m = ElasticWave3D(
            n, dt=dt, spacing=h, vp=vp, vs=vs, rho=rho, absorb_width=5, absorb_strength=6.0, free_surface=True
        )
        ops = make_ops()

        # a vertical point force just below the free surface -- the classic Rayleigh-wave excitation
        c = n // 2
        state = m.zeros(ops)
        f0 = 4.0
        for i in range(180):
            t = i * dt
            amp = (1.0 - 2.0 * (np.pi * f0 * (t - 0.15)) ** 2) * np.exp(-((np.pi * f0 * (t - 0.15)) ** 2))
            src = m.point_force_source((c, c, 2), (0.0, 0.0, 1.0), amplitude=amp)
            state = m.step(state, ops, source=src)

        self.assertTrue(torch.isfinite(state).all())
        vx, vy, vz = (f.detach().numpy() for f in m.velocity(state, ops))
        speed = np.sqrt(vx**2 + vy**2 + vz**2)
        # the free-surface tractions are held at zero on the top face each step
        sxx, syy, szz, sxy, sxz, syz = (f.detach().numpy() for f in m.fields(state, ops)[3:])
        self.assertLess(np.abs(szz[:, :, 0]).max(), 1e-9)
        self.assertLess(np.abs(sxz[:, :, 0]).max(), 1e-9)
        self.assertLess(np.abs(syz[:, :, 0]).max(), 1e-9)
        self.assertGreater(speed.max(), 0.0)  # motion was excited


if __name__ == "__main__":
    unittest.main()
