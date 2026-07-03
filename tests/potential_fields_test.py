"""Tests for full-tensor gravity gradiometry and vector/gradient magnetic point-source kernels.

The bar is agreement with the exact analytic point-source closed forms:
  * gravity gradient    T_ij = G m (3 r_i r_j - |r|^2 delta_ij) / |r|^5   (r = obs - source),
  * Laplace trace-free  T_xx + T_yy + T_zz = 0 off the source,
  * consistency         T_zz = d(g_z)/dz of geophysics.gravity_point_sensitivity (finite difference),
plus the magnetic vector components against the dipole field and TMI = projection onto the field direction.
"""

import unittest

import numpy as np

from mixle_pde.geophysics import gravity_point_sensitivity, magnetic_dipole_sensitivity
from mixle_pde.potential_fields import (
    G_GRAV,
    gravity_gradient_components,
    gravity_gradient_tensor,
    magnetic_gradient_tensor,
    magnetic_vector_sensitivity,
    trace_free_zz,
)

EOTVOS = 1.0e9
_AXIS = {"x": 0, "y": 1, "z": 2}


def _analytic_ftg(obs, source, mass, i, j):
    """Exact gravity-gradient component T_ij (SI, s^-2) of a point mass. r = obs - source."""
    r = np.asarray(obs, float) - np.asarray(source, float)
    rn = np.linalg.norm(r)
    delta = 1.0 if i == j else 0.0
    return G_GRAV * mass * (3.0 * r[i] * r[j] - delta * rn**2) / rn**5


class GravityGradientAnalyticTestCase(unittest.TestCase):
    """The FTG kernel of a single point mass matches the closed form component-by-component to ~1e-10."""

    def test_all_components_match_closed_form(self):
        # a generic off-axis geometry so no r_i is zero and every component is exercised
        obs = np.array([[12.0, -7.0, 3.0]])
        source = np.array([[-4.0, 5.0, -9.0]])
        vol = 2.5e5
        rho = 1.0  # unit density contrast so volume == mass
        mass = vol * rho
        blocks = gravity_gradient_components(obs, source, vol, components=("xx", "xy", "xz", "yy", "yz", "zz"))
        for comp in ("xx", "xy", "xz", "yy", "yz", "zz"):
            i, j = _AXIS[comp[0]], _AXIS[comp[1]]
            got = float(blocks[comp][0, 0]) / EOTVOS  # kernel returns Eotvos; compare in SI
            want = _analytic_ftg(obs[0], source[0], mass, i, j)
            self.assertAlmostEqual(got, want, delta=1e-10 * max(1.0, abs(want)))

    def test_symmetry_xy_equals_yx(self):
        obs = np.array([[3.0, 8.0, 1.0]])
        source = np.array([[-2.0, 1.0, -6.0]])
        b = gravity_gradient_components(obs, source, 1.0e4, components=("xy", "yx", "xz", "zx"))
        self.assertAlmostEqual(float(b["xy"][0, 0]), float(b["yx"][0, 0]), delta=1e-12)
        self.assertAlmostEqual(float(b["xz"][0, 0]), float(b["zx"][0, 0]), delta=1e-12)

    def test_si_units_scale(self):
        obs = np.array([[1.0, 2.0, 10.0]])
        source = np.array([[0.0, 0.0, 0.0]])
        e = gravity_gradient_components(obs, source, 1.0e3, components=("xx",), units="eotvos")["xx"]
        s = gravity_gradient_components(obs, source, 1.0e3, components=("xx",), units="si")["xx"]
        self.assertAlmostEqual(float(e[0, 0]), EOTVOS * float(s[0, 0]), delta=1e-9 * abs(float(e[0, 0])))


class GravityGradientTraceFreeTestCase(unittest.TestCase):
    """Laplace: Txx + Tyy + Tzz = 0 in the source-free region, to ~1e-10 (relative)."""

    def test_trace_free_single_source(self):
        obs = np.array([[5.0, -3.0, 7.0]])
        source = np.array([[-1.0, 2.0, -4.0]])
        b = gravity_gradient_components(obs, source, 3.3e5, components=("xx", "yy", "zz"))
        trace = float(b["xx"][0, 0] + b["yy"][0, 0] + b["zz"][0, 0])
        scale = max(abs(float(b["xx"][0, 0])), abs(float(b["yy"][0, 0])), abs(float(b["zz"][0, 0])))
        self.assertLess(abs(trace), 1e-10 * scale)

    def test_trace_free_many_sources_and_stations(self):
        rng = np.random.RandomState(0)
        obs = rng.uniform(-20, 20, (7, 3))
        cells = rng.uniform(-40, -10, (11, 3))  # sources well below the stations (source-free region above)
        vols = rng.uniform(1e3, 1e5, 11)
        b = gravity_gradient_components(obs, cells, vols, components=("xx", "yy", "zz"))
        trace = b["xx"] + b["yy"] + b["zz"]  # per (station, cell)
        scale = np.maximum.reduce([np.abs(b["xx"]), np.abs(b["yy"]), np.abs(b["zz"])])
        self.assertLess(np.max(np.abs(trace) / scale), 1e-10)

    def test_trace_free_helper_matches(self):
        obs = np.array([[2.0, 2.0, 9.0]])
        source = np.array([[-3.0, 1.0, -2.0]])
        b = gravity_gradient_components(obs, source, 1.0e4, components=("xx", "yy", "zz"))
        zz_from_helper = trace_free_zz(b["xx"], b["yy"])
        self.assertAlmostEqual(float(zz_from_helper[0, 0]), float(b["zz"][0, 0]), delta=1e-12)


class GravityGradientConsistencyTestCase(unittest.TestCase):
    """T_zz equals d(g_z)/dz of geophysics.gravity_point_sensitivity (finite-difference check).

    The stated tensor is T_ij = -d2U/dx_i dx_j = d(g_i)/dx_j with g = -grad U the attraction. The existing
    gravity_point_sensitivity returns g_z = 1e5 G V dz/r^3 with dz = obs_z - source_z, i.e. the +dU/dz sign
    (it grows as the observer moves away above the source), so its finite difference gives -T. We assert that
    sign relationship exactly, converting units mGal/m -> Eotvos via 1e9/1e5.
    """

    def test_tzz_is_minus_dgz_dz(self):
        source = np.array([[10.0, -5.0, -80.0]])
        vol = 1.0e5
        eps = 1e-3  # metres
        obs0 = np.array([[3.0, 4.0, 20.0]])
        up = obs0 + np.array([0.0, 0.0, eps])
        dn = obs0 - np.array([0.0, 0.0, eps])
        gz_up = float(gravity_point_sensitivity(up, source, vol)[0, 0])
        gz_dn = float(gravity_point_sensitivity(dn, source, vol)[0, 0])
        dgz_dz = (gz_up - gz_dn) / (2.0 * eps)  # mGal / m
        tzz = float(gravity_gradient_tensor(obs0, source, vol, components=("zz",))[0, 0])  # Eotvos
        self.assertAlmostEqual(-dgz_dz * (EOTVOS / 1.0e5), tzz, delta=1e-4 * abs(tzz))

    def test_txz_is_minus_dgz_dx(self):
        source = np.array([[10.0, -5.0, -80.0]])
        vol = 1.0e5
        eps = 1e-3
        obs0 = np.array([[3.0, 4.0, 20.0]])
        px = obs0 + np.array([eps, 0.0, 0.0])
        mx = obs0 - np.array([eps, 0.0, 0.0])
        gz_px = float(gravity_point_sensitivity(px, source, vol)[0, 0])
        gz_mx = float(gravity_point_sensitivity(mx, source, vol)[0, 0])
        dgz_dx = (gz_px - gz_mx) / (2.0 * eps)
        txz = float(gravity_gradient_tensor(obs0, source, vol, components=("xz",))[0, 0])
        self.assertAlmostEqual(-dgz_dx * (EOTVOS / 1.0e5), txz, delta=1e-4 * abs(txz))


class GravityGradientStackTestCase(unittest.TestCase):
    """The stacked full-tensor operator has the right block layout and stays linear (d = G @ rho)."""

    def test_stacked_shape_and_block_order(self):
        rng = np.random.RandomState(1)
        obs = rng.uniform(-10, 10, (4, 3))
        cells = rng.uniform(-30, -5, (6, 3))
        vols = rng.uniform(1e3, 1e4, 6)
        comps = ("xx", "xy", "xz", "yy", "yz")
        G = gravity_gradient_tensor(obs, cells, vols, components=comps)
        self.assertEqual(G.shape, (4 * len(comps), 6))
        blocks = gravity_gradient_components(obs, cells, vols, components=comps)
        for k, comp in enumerate(comps):
            np.testing.assert_allclose(G[k * 4 : (k + 1) * 4], blocks[comp])

    def test_linear_forward(self):
        rng = np.random.RandomState(2)
        obs = rng.uniform(-10, 10, (3, 3))
        cells = rng.uniform(-30, -5, (5, 3))
        G = gravity_gradient_tensor(obs, cells, 1e4)
        rho = rng.randn(5)
        d = G @ rho
        self.assertEqual(d.shape, (3 * 5,))
        # linearity: doubling the model doubles the data
        np.testing.assert_allclose(G @ (2.0 * rho), 2.0 * d)


class MagneticVectorTestCase(unittest.TestCase):
    """Vector-component kernel matches the dipole field; TMI = projection onto the field direction."""

    def test_vector_matches_dipole_field(self):
        # B = (T0 V / 4pi) (3 (b.rhat) rhat - b) / |r|^3 * kappa ; check each component of one source.
        obs = np.array([[6.0, -2.0, 4.0]])
        source = np.array([[-3.0, 5.0, -7.0]])
        vol, T0 = 1.5e4, 51000.0
        inc, dec = -55.0, 7.0
        inc_r, dec_r = np.radians(inc), np.radians(dec)
        b = np.array([np.cos(inc_r) * np.sin(dec_r), np.cos(inc_r) * np.cos(dec_r), -np.sin(inc_r)])
        r = obs[0] - source[0]
        rn = np.linalg.norm(r)
        rhat = r / rn
        want = (T0 * vol / (4.0 * np.pi)) * (3.0 * (b @ rhat) * rhat - b) / rn**3
        got = magnetic_vector_sensitivity(
            obs, source, vol, inclination=inc, declination=dec, field_nt=T0, components=("x", "y", "z")
        )
        for k, comp in enumerate(("x", "y", "z")):
            self.assertAlmostEqual(float(got[comp][0, 0]), want[k], delta=1e-10 * max(1.0, abs(want[k])))

    def test_tmi_is_projection_onto_field(self):
        # geophysics.magnetic_dipole_sensitivity (scalar TMI) == b . B_vector, to ~1e-12 relative.
        rng = np.random.RandomState(3)
        obs = rng.uniform(-15, 15, (5, 3))
        cells = rng.uniform(-40, -10, (4, 3))
        vols = rng.uniform(1e3, 1e4, 4)
        inc, dec, T0 = -63.0, 11.0, 49000.0
        b = np.array(
            [
                np.cos(np.radians(inc)) * np.sin(np.radians(dec)),
                np.cos(np.radians(inc)) * np.cos(np.radians(dec)),
                -np.sin(np.radians(inc)),
            ]
        )
        vec = magnetic_vector_sensitivity(obs, cells, vols, inclination=inc, declination=dec, field_nt=T0)
        proj = b[0] * vec["x"] + b[1] * vec["y"] + b[2] * vec["z"]
        tmi = magnetic_dipole_sensitivity(obs, cells, vols, inclination=inc, declination=dec, field_nt=T0)
        np.testing.assert_allclose(proj, tmi, rtol=1e-12, atol=1e-12)

    def test_component_subset(self):
        obs = np.array([[1.0, 1.0, 10.0]])
        cells = np.array([[0.0, 0.0, -5.0]])
        only_z = magnetic_vector_sensitivity(obs, cells, 1.0e3, inclination=-60.0, declination=0.0, components=("z",))
        self.assertEqual(set(only_z), {"z"})
        self.assertTrue(np.isfinite(only_z["z"]).all())


class MagneticGradientTestCase(unittest.TestCase):
    """The magnetic gradient tensor dB_i/dx_j matches a finite difference of the vector kernel, and is trace-free."""

    def test_gradient_matches_finite_difference(self):
        source = np.array([[-4.0, 3.0, -30.0]])
        vol, T0 = 2.0e4, 50000.0
        inc, dec = -58.0, 9.0
        obs0 = np.array([[5.0, -2.0, 12.0]])
        eps = 1e-3

        def field_component(obs, comp):
            v = magnetic_vector_sensitivity(
                obs, source, vol, inclination=inc, declination=dec, field_nt=T0, components=(comp,)
            )
            return float(v[comp][0, 0])

        grad = magnetic_gradient_tensor(
            obs0, source, vol, inclination=inc, declination=dec, field_nt=T0, components=("xx", "xy", "xz", "yy", "yz")
        )
        for comp in ("xx", "xy", "xz", "yy", "yz"):
            i, j = _AXIS[comp[0]], _AXIS[comp[1]]
            step = np.zeros(3)
            step[j] = eps
            fd = (field_component(obs0 + step, comp[0]) - field_component(obs0 - step, comp[0])) / (2.0 * eps)
            got = float(grad[comp][0, 0])
            self.assertAlmostEqual(got, fd, delta=1e-5 * max(1.0, abs(got)))

    def test_gradient_trace_free(self):
        # sum_i dB_i/dx_i = div(B) = 0 off the source
        source = np.array([[1.0, -1.0, -25.0]])
        obs = np.array([[4.0, 6.0, 15.0]])
        g = magnetic_gradient_tensor(
            obs, source, 1.0e4, inclination=-50.0, declination=5.0, components=("xx", "yy", "zz")
        )
        trace = float(g["xx"][0, 0] + g["yy"][0, 0] + g["zz"][0, 0])
        scale = max(abs(float(g["xx"][0, 0])), abs(float(g["yy"][0, 0])), abs(float(g["zz"][0, 0])))
        self.assertLess(abs(trace), 1e-8 * scale)

    def test_gradient_symmetry(self):
        source = np.array([[0.0, 0.0, -20.0]])
        obs = np.array([[3.0, 2.0, 10.0]])
        g = magnetic_gradient_tensor(obs, source, 1.0e4, inclination=-45.0, declination=0.0, components=("xy", "yx"))
        self.assertAlmostEqual(float(g["xy"][0, 0]), float(g["yx"][0, 0]), delta=1e-10)


if __name__ == "__main__":
    unittest.main()
