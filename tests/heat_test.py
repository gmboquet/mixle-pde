"""Transient heterogeneous heat conduction: analytical-benchmark tests for TransientHeat."""

import unittest

import numpy as np

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

if HAS_TORCH:
    from mixle_pde.heat import TransientHeat
    from mixle_pde.ops import make_ops


@unittest.skipUnless(HAS_TORCH, "requires PyTorch")
class FourierDecayTestCase(unittest.TestCase):
    """Benchmark 1: a homogeneous decaying Fourier mode on [0, L] with Dirichlet ends.

    T(x, 0) = sin(pi x / L) decays as exp(-alpha (pi/L)^2 t), alpha = k / rho c. The measured decay rate
    of the numerical solution must match the analytical rate.
    """

    def setUp(self):
        torch.set_default_dtype(torch.float64)

    def test_decay_rate_matches_analytical(self):
        L = 2.0
        m = 81
        k_val, rho_c = 0.7, 1.3
        alpha = k_val / rho_c
        xs = np.linspace(0.0, L, m)
        h = xs[1] - xs[0]
        dt = 0.4 * h**2 / (2.0 * alpha)  # inside the CFL limit
        nt = 400

        heat = TransientHeat(m, dt=dt, spacing=h)
        ops = make_ops()
        T0 = torch.as_tensor(np.sin(np.pi * xs / L))
        k = torch.full((m,), k_val)

        def stepper(y, i):
            return heat.step(y, k, ops, rho_c=rho_c)

        rec = ops.integrate_record(stepper, T0, nt, lambda y, i: y, checkpoint=None).detach().numpy()

        # amplitude at the mode's peak (midpoint) over time; fit log-amplitude vs t for the decay rate
        mid = m // 2
        amp = rec[:, mid]
        t = np.arange(nt + 1) * dt
        rate_num = -np.polyfit(t, np.log(amp), 1)[0]
        rate_exact = alpha * (np.pi / L) ** 2
        rel_err = abs(rate_num - rate_exact) / rate_exact
        self.assertLess(rel_err, 0.02)  # within 2%

    def test_full_field_matches_separable_solution(self):
        """The whole field, not just the peak, matches T(x,t) = sin(pi x/L) exp(-alpha (pi/L)^2 t)."""
        L = 1.0
        m = 61
        k_val, rho_c = 1.0, 1.0
        alpha = k_val / rho_c
        xs = np.linspace(0.0, L, m)
        h = xs[1] - xs[0]
        dt = 0.4 * h**2 / (2.0 * alpha)
        nt = 300

        heat = TransientHeat(m, dt=dt, spacing=h)
        ops = make_ops()
        T0 = torch.as_tensor(np.sin(np.pi * xs / L))
        k = torch.full((m,), k_val)
        Tn = T0
        for _ in range(nt):
            Tn = heat.step(Tn, k, ops, rho_c=rho_c)
        t_end = nt * dt
        exact = np.sin(np.pi * xs / L) * np.exp(-alpha * (np.pi / L) ** 2 * t_end)
        err = np.max(np.abs(Tn.detach().numpy() - exact))
        self.assertLess(err, 2e-3)


@unittest.skipUnless(HAS_TORCH, "requires PyTorch")
class FlashErfcTestCase(unittest.TestCase):
    """Benchmark 2: semi-infinite flash-heated surface response.

    A constant surface heat flux Q on the face of a semi-infinite solid (k, rho c, alpha = k / rho c) gives
    the surface-temperature rise (Carslaw & Jaeger):

        T_surf(t) - T0 = (2 Q / k) sqrt(alpha t / pi).

    On a finite grid this holds until the thermal front reaches the far end. We compare the numerical
    surface temperature to the closed form over that early window.
    """

    def setUp(self):
        torch.set_default_dtype(torch.float64)

    def test_surface_temperature_matches_erfc_law(self):
        Lz = 1.0
        m = 201
        k_val, rho_c = 2.0, 2.5
        alpha = k_val / rho_c
        Q = 3.0
        zs = np.linspace(0.0, Lz, m)
        h = zs[1] - zs[0]
        dt = 0.4 * h**2 / (2.0 * alpha)
        nt = 600  # keep the front well inside the domain: sqrt(alpha t) << Lz

        heat = TransientHeat(m, dt=dt, spacing=h, flux_face=(0, 0))
        ops = make_ops()
        T0 = torch.zeros(m)
        k = torch.full((m,), k_val)

        def stepper(y, i):
            return heat.step(y, k, ops, rho_c=rho_c, flux=Q)

        rec = ops.integrate_record(stepper, T0, nt, lambda y, i: heat.surface(y)[0], checkpoint=None)
        surf = rec.detach().numpy()
        t = np.arange(nt + 1) * dt
        exact = (2.0 * Q / k_val) * np.sqrt(alpha * t / np.pi)

        # front depth ~ sqrt(alpha * t_end); confirm it is still inside the domain
        front = np.sqrt(alpha * t[-1])
        self.assertLess(front, 0.6 * Lz)

        # compare over the window after the first few steps (early transient is grid-resolution limited)
        sel = slice(20, nt + 1)
        rel = np.abs(surf[sel] - exact[sel]) / np.maximum(exact[sel], 1e-12)
        self.assertLess(float(np.max(rel)), 0.03)  # within 3%


@unittest.skipUnless(HAS_TORCH, "requires PyTorch")
class HeterogeneousInverseTestCase(unittest.TestCase):
    """Recover a localized low-conductivity defect from a surface thermogram (the inverse demonstration)."""

    def setUp(self):
        torch.set_default_dtype(torch.float64)

    def test_recovers_defect_contrast(self):
        m = 41
        rho_c = 1.0
        k_bg, k_defect = 1.0, 0.3
        zs = np.linspace(0.0, 1.0, m)
        h = zs[1] - zs[0]
        alpha_max = k_bg / rho_c
        dt = 0.4 * h**2 / (2.0 * alpha_max)
        nt = 400
        Q = 1.0

        # true field: a shallow low-k slab (a subsurface delamination) that dams heat and lifts the surface
        # temperature -- the near-surface geometry where thermography actually resolves the defect
        defect = (zs > 0.1) & (zs < 0.25)
        k_true = np.where(defect, k_defect, k_bg)

        heat = TransientHeat(m, dt=dt, spacing=h, flux_face=(0, 0))
        ops = make_ops()
        T0 = torch.zeros(m)

        def surface_series(k_field):
            def stepper(y, i):
                return heat.step(y, k_field, ops, rho_c=rho_c, flux=Q)

            return ops.integrate_record(stepper, T0, nt, lambda y, i: heat.surface(y)[0], checkpoint=25)

        obs = surface_series(torch.as_tensor(k_true)).detach()

        # invert for the defect conductivity (a single scalar contrast), everything else known
        log_kd = torch.tensor(np.log(0.7), requires_grad=True)
        opt = torch.optim.Adam([log_kd], lr=0.15)
        for _ in range(200):
            opt.zero_grad()
            kd = torch.exp(log_kd)
            k_field = torch.where(torch.as_tensor(defect), kd, torch.tensor(k_bg))
            pred = surface_series(k_field)
            loss = ((pred - obs) ** 2).sum()
            loss.backward()
            opt.step()
        recovered = float(torch.exp(log_kd).detach())
        self.assertLess(abs(recovered - k_defect), 0.05)


if __name__ == "__main__":
    unittest.main()
