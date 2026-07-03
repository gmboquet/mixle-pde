"""Tests for the 3D anisotropic (VTI/TTI) elastic-wave stepper: exact Thomsen phase velocities on the axes.

The defining physics of a transversely isotropic solid is that a plane wave travels at a different speed
along the symmetry axis than across it. Along the two symmetry axes the phase velocities are the exact
Christoffel/Thomsen values

    vertical   P  = sqrt(c33/rho) = Vp0
    horizontal P  = sqrt(c11/rho) = Vp0 sqrt(1 + 2 epsilon)
    vertical   S  = sqrt(c44/rho) = Vs0
    horizontal SH = sqrt(c66/rho) = Vs0 sqrt(1 + 2 gamma)

We seed a one-way (right-going) plane wave along a symmetry axis with the stress it carries in the Riemann
invariant (``s_aa = -rho c v_a`` for the normal/P mode, ``s_ab = -rho c v_b`` for the shear mode), track the
wavefront by the centroid of the transverse-averaged amplitude profile, and fit its slope in time. Each is
required to match the analytical speed to ~1.5%. A TTI check tilts the symmetry axis by 90 degrees about y
and confirms vertical propagation now sees the fast (originally horizontal) P speed. Two structural checks
confirm the Thomsen->cij mapping and that the medium is differentiable through ``ops``.
"""

import unittest

import numpy as np

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

if HAS_TORCH:
    from mixle_pde.elastic_aniso import AnisotropicElasticWave3D, thomsen_to_cij
    from mixle_pde.ops import make_ops

_ORDER = ("vx", "vy", "vz", "sxx", "syy", "szz", "sxy", "sxz", "syz")
_SHEAR_NAME = {(0, 1): "sxy", (0, 2): "sxz", (1, 2): "syz"}
_SHEAR_VIDX = {"sxy": 5, "sxz": 4, "syz": 3}

# a representative VTI medium (Thomsen), all five parameters nonzero
_VP0, _VS0, _EPS, _DLT, _GAM, _RHO = 2.0, 1.0, 0.2, 0.1, 0.15, 1.0


def _axis_phase_speed(mode, prop_axis, *, tilt=0.0, n=60, sample=4):
    """Seed a one-way plane wave along ``prop_axis`` and return (numeric speed, analytic speed, npts, finite).

    ``mode`` is "P" (particle motion along propagation) or "S" (transverse motion). The analytic speed is
    read off the (possibly tilted) per-cell stiffness so the benchmark stays exact even for a TTI medium.
    """
    ops = make_ops()
    h = 1.0 / n
    c11, c33, c44, c66, c13 = thomsen_to_cij(_VP0, _VS0, _EPS, _DLT, _GAM, _RHO)
    vp_max = np.sqrt(max(c11, c33) / _RHO)
    dt = 0.3 * h / (vp_max * np.sqrt(3))  # comfortably under the 3D Courant limit
    m = AnisotropicElasticWave3D(
        n,
        dt=dt,
        spacing=h,
        vp0=_VP0,
        vs0=_VS0,
        epsilon=_EPS,
        delta=_DLT,
        gamma=_GAM,
        rho=_RHO,
        tilt=tilt,
        tilt_axis=1,
        absorb_width=8,
        absorb_strength=6.0,
    )

    coord = np.arange(n) * h
    x0, width = 0.18, 0.05  # a narrow Gaussian pulse near the near edge
    shp = [1, 1, 1]
    shp[prop_axis] = n
    env = np.broadcast_to(np.exp(-((coord - x0) ** 2) / (2 * width**2)).reshape(shp), (n, n, n)).copy()
    z = np.zeros((n, n, n))
    comps = {c: z.copy() for c in _ORDER}
    vnames = ("vx", "vy", "vz")

    C0 = m._C[0, 0, 0]  # homogeneous medium: read the analytic modulus off any cell
    if mode == "P":
        vc = prop_axis  # particle motion along propagation
        comps[vnames[vc]] = env
        c_nn = C0[prop_axis, prop_axis]  # normal-normal Voigt modulus along this axis
        speed_analytic = float(np.sqrt(c_nn / _RHO))
        comps[("sxx", "syy", "szz")[prop_axis]] = -_RHO * speed_analytic * env
        probe = vc
    else:
        vc = (prop_axis + 1) % 3  # a transverse motion direction
        comps[vnames[vc]] = env
        sname = _SHEAR_NAME[tuple(sorted((prop_axis, vc)))]
        c_ss = C0[_SHEAR_VIDX[sname], _SHEAR_VIDX[sname]]
        speed_analytic = float(np.sqrt(c_ss / _RHO))
        comps[sname] = -_RHO * speed_analytic * env
        probe = vc

    state = m.pack(*[comps[c] for c in _ORDER])
    axes = tuple(a for a in range(3) if a != prop_axis)  # transverse-average axes
    times, cents, last = [], [], -1.0
    for i in range(1, 900):
        state = m.step(state, ops)
        if i % sample == 0:
            v = m.velocity(state, ops)[probe].detach().numpy()
            prof = np.abs(v).mean(axis=axes)  # transverse-averaged amplitude profile along the axis
            pk = int(np.argmax(prof))
            pkpos = pk * h
            if pkpos > 0.80:  # reached the far sponge; stop before reflection contaminates the fit
                break
            if pkpos >= 0.28 and pkpos >= last - 1e-9:  # forward-going, in the clean window
                lo, hi = max(0, pk - 4), min(n, pk + 5)
                w, xs = prof[lo:hi], coord[lo:hi]
                cent = float((w * xs).sum() / w.sum())  # sub-grid centroid of the wavefront
                times.append(i * dt)
                cents.append(cent)
                last = pkpos
    finite = bool(torch.isfinite(state).all())
    speed = float(np.polyfit(np.array(times), np.array(cents), 1)[0])  # slope of front position vs time
    return speed, speed_analytic, len(cents), finite


@unittest.skipUnless(HAS_TORCH, "requires PyTorch")
class ThomsenMappingTestCase(unittest.TestCase):
    """The Thomsen -> VTI-constant map reproduces the axis phase velocities exactly."""

    def test_thomsen_to_cij_axis_velocities(self):
        c11, c33, c44, c66, c13 = thomsen_to_cij(_VP0, _VS0, _EPS, _DLT, _GAM, _RHO)
        self.assertAlmostEqual(float(np.sqrt(c33 / _RHO)), _VP0, places=10)  # vertical P
        self.assertAlmostEqual(float(np.sqrt(c11 / _RHO)), _VP0 * np.sqrt(1 + 2 * _EPS), places=10)
        self.assertAlmostEqual(float(np.sqrt(c44 / _RHO)), _VS0, places=10)  # vertical S
        self.assertAlmostEqual(float(np.sqrt(c66 / _RHO)), _VS0 * np.sqrt(1 + 2 * _GAM), places=10)


@unittest.skipUnless(HAS_TORCH, "requires PyTorch")
class VTIPhaseVelocityTestCase(unittest.TestCase):
    """Propagated plane waves travel at the analytical Thomsen speeds along the symmetry axes."""

    def setUp(self):
        torch.set_default_dtype(torch.float64)

    def _check(self, mode, prop_axis, tol=0.015):
        speed, analytic, npts, finite = _axis_phase_speed(mode, prop_axis)
        self.assertTrue(finite)  # no NaN over the run
        self.assertGreaterEqual(npts, 8)  # enough clean points to fit a slope
        rel = abs(speed - analytic) / analytic
        self.assertLess(rel, tol, f"{mode} axis {prop_axis}: {speed:.4f} vs {analytic:.4f} (err {rel:.4f})")
        return speed, analytic

    def test_vertical_p_speed(self):
        speed, analytic = self._check("P", 2)  # sqrt(c33/rho) = Vp0
        self.assertAlmostEqual(analytic, _VP0, places=10)

    def test_horizontal_p_speed(self):
        speed, analytic = self._check("P", 0)  # sqrt(c11/rho) = Vp0 sqrt(1+2 eps)
        self.assertAlmostEqual(analytic, _VP0 * np.sqrt(1 + 2 * _EPS), places=10)

    def test_vertical_s_speed(self):
        speed, analytic = self._check("S", 2)  # sqrt(c44/rho) = Vs0
        self.assertAlmostEqual(analytic, _VS0, places=10)

    def test_horizontal_sh_speed(self):
        speed, analytic = self._check("S", 0)  # sqrt(c66/rho) = Vs0 sqrt(1+2 gamma)
        self.assertAlmostEqual(analytic, _VS0 * np.sqrt(1 + 2 * _GAM), places=10)

    def test_horizontal_faster_than_vertical_p(self):
        """The anisotropy is real: the P wave runs faster across the layering than along the axis."""
        vpv, _ = self._check("P", 2)
        vph, _ = self._check("P", 0)
        self.assertGreater(vph, vpv)  # epsilon > 0 => horizontal P faster


@unittest.skipUnless(HAS_TORCH, "requires PyTorch")
class TTITiltTestCase(unittest.TestCase):
    """A Bond-rotated symmetry axis moves the fast direction with the tilt (TTI)."""

    def setUp(self):
        torch.set_default_dtype(torch.float64)

    def test_ninety_degree_tilt_swaps_p_speeds(self):
        """Tilting the symmetry axis 90 deg about y: vertical propagation now sees the fast (horizontal) P."""
        speed, analytic, npts, finite = _axis_phase_speed("P", 2, tilt=np.pi / 2)
        self.assertTrue(finite)
        self.assertGreaterEqual(npts, 8)
        # after the 90 deg tilt the analytic modulus along z is the original c11 (fast) speed
        self.assertAlmostEqual(analytic, _VP0 * np.sqrt(1 + 2 * _EPS), places=6)
        rel = abs(speed - analytic) / analytic
        self.assertLess(rel, 0.02, f"TTI90 vertical P {speed:.4f} vs {analytic:.4f} (err {rel:.4f})")


@unittest.skipUnless(HAS_TORCH, "requires PyTorch")
class DifferentiabilityTestCase(unittest.TestCase):
    """The anisotropic wavefield is differentiable w.r.t. the stiffness (anisotropic FWI sensitivity)."""

    def setUp(self):
        torch.set_default_dtype(torch.float64)

    def test_gradient_flows_to_stiffness(self):
        n, rho = 16, 1.0
        h = 1.0 / n
        c11, c33, c44, c66, c13 = thomsen_to_cij(_VP0, _VS0, _EPS, _DLT, _GAM, rho)
        dt = 0.3 * h / (np.sqrt(max(c11, c33) / rho) * np.sqrt(3))
        m = AnisotropicElasticWave3D(
            n, dt=dt, spacing=h, c11=c11, c33=c33, c44=c44, c66=c66, c13=c13, rho=rho, absorb_width=3
        )
        ops = make_ops()
        # make the per-cell Voigt stiffness a leaf tensor; ops.tensor keeps the graph on a tensor input
        m._C = torch.tensor(m._C, requires_grad=True)

        c = n // 2
        state = m.zeros(ops)
        f0 = 4.0
        for i in range(40):
            t = i * dt
            amp = (1.0 - 2.0 * (np.pi * f0 * (t - 0.15)) ** 2) * np.exp(-((np.pi * f0 * (t - 0.15)) ** 2))
            src = m.moment_tensor_source((c, c, c), np.eye(3), amplitude=amp)
            state = m.step(state, ops, source=src)
        vx, vy, vz = m.velocity(state, ops)
        loss = (vx**2 + vy**2 + vz**2).sum()
        loss.backward()

        self.assertTrue(torch.isfinite(m._C.grad).all())
        self.assertGreater(float(m._C.grad.norm()), 0.0)  # a real, nonzero anisotropy sensitivity


if __name__ == "__main__":
    unittest.main()
