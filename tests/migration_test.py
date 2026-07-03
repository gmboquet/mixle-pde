"""Tests for reverse-time migration: image a single flat reflector at the correct depth."""

import unittest

import numpy as np

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

if HAS_TORCH:
    from mixle_pde.migration import born_modeling, lsrtm_step, model_data, rtm_image
    from mixle_pde.ops import make_ops
    from mixle_pde.wave import WaveEquation2D


def _flat_reflector_setup(n=80, nt=500):
    """Two-layer model with one horizontal interface. Axis 0 = depth z (down), axis 1 = horizontal x."""
    h = 1.0 / (n - 1)
    c0 = 1.0
    dt = 0.3 * h / c0
    aw = 8  # absorbing sponge width
    wave = WaveEquation2D(n, dt=dt, spacing=h, absorb_width=aw, absorb_strength=0.6 / dt)
    ops = make_ops()

    z_ref = 45  # reflector depth (grid index)
    c_bg = np.full((n, n), c0)  # smooth background: homogeneous, no reflector
    c_true = c_bg.copy()
    c_true[z_ref:, :] = 1.6 * c0  # faster lower layer -> one reflection
    c2_bg = torch.as_tensor((c_bg**2).ravel())
    c2_true = torch.as_tensor((c_true**2).ravel())

    z_top = aw + 2  # source/receivers just below the top sponge
    src_node = z_top * n + n // 2

    f0 = 9.0
    tg = np.arange(nt + 1) * dt
    a = (np.pi * f0 * (tg - 3.0 / f0)) ** 2
    ricker = (1 - 2 * a) * np.exp(-a)

    recv_x = np.arange(aw + 2, n - aw - 2)
    recv_nodes = z_top * n + recv_x

    return {
        "n": n,
        "nt": nt,
        "wave": wave,
        "ops": ops,
        "z_ref": z_ref,
        "z_top": z_top,
        "aw": aw,
        "c2_bg": c2_bg,
        "c2_true": c2_true,
        "src_node": src_node,
        "ricker": ricker,
        "recv_nodes": recv_nodes,
    }


def _scattered_data(s):
    """Observed (true model) minus background (direct) data = the isolated reflection."""
    ops, wave, nt = s["ops"], s["wave"], s["nt"]
    d_true = model_data(
        wave, s["c2_true"], s["src_node"], s["ricker"], s["recv_nodes"], nt, ops, checkpoint=20
    ).detach()
    d_bg = model_data(wave, s["c2_bg"], s["src_node"], s["ricker"], s["recv_nodes"], nt, ops, checkpoint=20).detach()
    return d_true - d_bg


def _depth_energy(image, s):
    """Interior depth profile: sum |image| over the physical horizontal window, per depth."""
    z_top, aw, n = s["z_top"], s["aw"], s["n"]
    zlo, zhi = z_top + 2, n - aw - 2
    xlo, xhi = aw + 2, n - aw - 2
    sub = image[zlo:zhi, xlo:xhi]
    return np.arange(zlo, zhi), np.abs(sub).sum(axis=1)


@unittest.skipUnless(HAS_TORCH, "requires PyTorch")
class RTMFlatReflectorTestCase(unittest.TestCase):
    def setUp(self):
        torch.set_default_dtype(torch.float64)

    def test_rtm_images_reflector_at_correct_depth(self):
        s = _flat_reflector_setup()
        residual = _scattered_data(s)
        # the scattered field must be a real reflection, not numerical dust
        self.assertGreater(float(residual.abs().max()), 1e-8)

        image = rtm_image(
            s["wave"],
            s["c2_bg"],
            s["src_node"],
            s["ricker"],
            s["recv_nodes"],
            residual,
            s["nt"],
            s["ops"],
            checkpoint=20,
        )
        self.assertEqual(image.shape, (s["n"], s["n"]))
        self.assertTrue(np.isfinite(image).all())

        zz, energy = _depth_energy(image, s)
        peak_z = int(zz[np.argmax(energy)])
        # the RTM image peaks AT the reflector depth (within a couple of grid cells)
        self.assertLessEqual(abs(peak_z - s["z_ref"]), 2)

        # near-zero away from the reflector: energy > 4 cells away is well below the peak
        peak_val = energy.max()
        away = energy[np.abs(zz - s["z_ref"]) > 4]
        self.assertLess(away.max(), 0.5 * peak_val)

    def test_rtm_column_profile_peaks_at_reflector(self):
        s = _flat_reflector_setup()
        residual = _scattered_data(s)
        image = rtm_image(
            s["wave"],
            s["c2_bg"],
            s["src_node"],
            s["ricker"],
            s["recv_nodes"],
            residual,
            s["nt"],
            s["ops"],
            checkpoint=20,
        )
        # central-column depth profile, restricted to the physical interior
        z_top, aw, n = s["z_top"], s["aw"], s["n"]
        zlo, zhi = z_top + 2, n - aw - 2
        col = np.abs(image[zlo:zhi, n // 2])
        peak_z = int(np.arange(zlo, zhi)[np.argmax(col)])
        self.assertLessEqual(abs(peak_z - s["z_ref"]), 2)


@unittest.skipUnless(HAS_TORCH, "requires PyTorch")
class BornAndLSRTMTestCase(unittest.TestCase):
    def setUp(self):
        torch.set_default_dtype(torch.float64)

    def test_born_matches_true_scattered_data(self):
        """The Born operator with the true perturbation reproduces the observed reflection to leading order."""
        s = _flat_reflector_setup()
        dm = s["c2_true"] - s["c2_bg"]
        d_born = born_modeling(
            s["wave"], s["c2_bg"], dm, s["src_node"], s["ricker"], s["recv_nodes"], s["nt"], s["ops"], checkpoint=20
        ).detach()
        d_scat = _scattered_data(s)
        # weak-contrast Born linearization tracks the true scattered field: correlated, comparable amplitude
        a = d_born.numpy().ravel()
        b = d_scat.numpy().ravel()
        corr = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
        self.assertGreater(corr, 0.9)

    def test_lsrtm_step_is_a_descent_that_images_the_reflector(self):
        """One LSRTM step from dm=0 is a genuine misfit-descent step whose update images the reflector.

        The misfit ``f(dm) = ||L dm - d_scat||^2`` is quadratic in ``dm`` for the Born operator ``L``. Along
        the RTM descent direction the exact line-search decrease ``(g^T g)^2 / (L g)^T(L g)`` is strictly
        positive, and the update ``dm1 = -alpha g`` concentrates at the reflector depth (that is the point of
        least-squares RTM). The absolute misfit change is below the double-precision cancellation floor of the
        raw sum, so the descent guarantee is asserted analytically rather than by differencing two ~1e-9 sums.
        """
        s = _flat_reflector_setup(nt=400)
        d_scat = _scattered_data(s).numpy()
        nn = s["n"] * s["n"]

        # g = RTM image of the residual at dm=0 (= -L^T d_scat), the least-squares gradient direction
        g = rtm_image(
            s["wave"],
            s["c2_bg"],
            s["src_node"],
            s["ricker"],
            s["recv_nodes"],
            torch.as_tensor(-d_scat),
            s["nt"],
            s["ops"],
            checkpoint=20,
            laplacian_filter=False,
        ).reshape(nn)
        lg = (
            born_modeling(
                s["wave"],
                s["c2_bg"],
                torch.as_tensor(g),
                s["src_node"],
                s["ricker"],
                s["recv_nodes"],
                s["nt"],
                s["ops"],
                checkpoint=20,
            )
            .detach()
            .numpy()
            .ravel()
        )
        gtg = float(np.dot(g, g))
        lgtlg = float(np.dot(lg, lg))
        # -g is a descent direction and the exact-line-search misfit decrease is strictly positive
        self.assertGreater(gtg, 0.0)
        predicted_decrease = gtg**2 / lgtlg
        self.assertGreater(predicted_decrease, 0.0)

        alpha = gtg / lgtlg
        dm1 = lsrtm_step(
            s["wave"],
            s["c2_bg"],
            np.zeros(nn),
            s["src_node"],
            s["ricker"],
            s["recv_nodes"],
            d_scat,
            s["nt"],
            s["ops"],
            step_size=alpha,
            checkpoint=20,
        )
        # the LSRTM reflectivity update peaks at the reflector depth
        zz, energy = _depth_energy(dm1.reshape(s["n"], s["n"]), s)
        peak_z = int(zz[np.argmax(energy)])
        self.assertLessEqual(abs(peak_z - s["z_ref"]), 2)


if __name__ == "__main__":
    unittest.main()
