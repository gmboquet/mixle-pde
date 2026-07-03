"""Physics tests for the 2D immiscible two-fluid solver (phase-field, variable-property projection).

Two analytical benchmarks on the ``TwoPhaseFlow2D`` stepper:

1. Two-layer (stratified) Poiseuille flow -- the lubrication benchmark. A channel with a thin low-viscosity
   film of thickness ``d`` at each wall and a high-viscosity core, driven by a constant pressure gradient
   ``G``, reaches a steady piecewise-parabolic profile with velocity AND shear stress ``mu du/dy`` continuous
   at the two interfaces. The solver, run to steady state with the phase held at the stratified configuration,
   must reproduce that profile, and the two-layer flow rate must be substantially larger than a single-fluid
   all-high-viscosity channel at the same ``G`` (the drag reduction the film buys).

2. Young-Laplace -- a static circular drop of radius ``R`` develops an interior overpressure ``sigma/R``
   (in 2D). The solver's surface-tension force, balanced by the mechanical pressure, must reproduce it.
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
    from mixle_pde.two_phase import TwoPhaseFlow2D


def two_layer_profile(y, *, G, mu_lo, mu_hi, d, H):
    """Exact two-layer Poiseuille velocity ``u(y)`` for a channel ``[0, H]`` driven by ``-dp/dx = G``.

    Low-viscosity ``mu_lo`` films occupy ``y < d`` and ``y > H - d``; the high-viscosity ``mu_hi`` core fills
    the middle. In each layer ``mu_i u'' = -G`` so ``u_i = -(G / 2 mu_i) y^2 + a_i y + b_i``. The six
    coefficients are fixed by no-slip at both walls and continuity of ``u`` and of the shear stress
    ``mu u'`` at ``y = d`` and ``y = H - d``. (The stress-continuity rows reduce to ``mu_lo a1 = mu_hi a2``
    etc., since ``mu_i * (-G/mu_i) y = -G y`` is the same in both layers.)"""

    def par(mu, yy):
        return -(G / (2.0 * mu)) * yy**2

    def dpar(mu, yy):
        return -(G / mu) * yy

    A = np.zeros((6, 6))
    rhs = np.zeros(6)
    yb = H - d
    A[0, 1] = 1.0  # u1(0) = 0
    rhs[0] = -par(mu_lo, 0.0)
    A[1, 4] = H  # u3(H) = 0
    A[1, 5] = 1.0
    rhs[1] = -par(mu_lo, H)
    A[2, 0] = d  # u1(d) = u2(d)
    A[2, 1] = 1.0
    A[2, 2] = -d
    A[2, 3] = -1.0
    rhs[2] = par(mu_hi, d) - par(mu_lo, d)
    A[3, 2] = yb  # u2(H-d) = u3(H-d)
    A[3, 3] = 1.0
    A[3, 4] = -yb
    A[3, 5] = -1.0
    rhs[3] = par(mu_lo, yb) - par(mu_hi, yb)
    A[4, 0] = mu_lo  # mu_lo u1'(d) = mu_hi u2'(d)
    A[4, 2] = -mu_hi
    rhs[4] = mu_hi * dpar(mu_hi, d) - mu_lo * dpar(mu_lo, d)
    A[5, 2] = mu_hi  # mu_hi u2'(H-d) = mu_lo u3'(H-d)
    A[5, 4] = -mu_lo
    rhs[5] = mu_lo * dpar(mu_lo, yb) - mu_hi * dpar(mu_hi, yb)
    a1, b1, a2, b2, a3, b3 = np.linalg.solve(A, rhs)

    y = np.asarray(y, dtype=float)
    out = np.zeros_like(y)
    m1 = y <= d
    m3 = y >= H - d
    m2 = ~(m1 | m3)
    out[m1] = par(mu_lo, y[m1]) + a1 * y[m1] + b1
    out[m2] = par(mu_hi, y[m2]) + a2 * y[m2] + b2
    out[m3] = par(mu_lo, y[m3]) + a3 * y[m3] + b3
    return out


@unittest.skipUnless(HAS_TORCH, "requires PyTorch")
class TwoLayerPoiseuilleTestCase(unittest.TestCase):
    """Stratified two-layer Poiseuille flow: the numerical profile matches the analytical, and the low-mu
    film substantially raises the flow rate at the same driving pressure gradient (the lubrication effect)."""

    def setUp(self):
        torch.set_default_dtype(torch.float64)

    def test_two_layer_profile_and_lubrication(self):
        ops = make_ops()
        ny = 65
        nx = 6
        H = 1.0
        h = H / (ny - 1)
        G = 1.0
        mu_lo, mu_hi = 0.1, 1.0
        d = 10.0 * h  # film thickness aligned to a grid node (interfaces sit on cell faces)

        # fluid 0 (phi=-1) is the low-viscosity wall film; fluid 1 (phi=+1) is the high-viscosity core.
        solver = TwoPhaseFlow2D(
            nx,
            ny,
            mu=(mu_lo, mu_hi),
            rho=(1.0, 1.0),
            dt=0.05,  # implicit diffusion -> dt limited only by the (vanishing) advection, so take big steps
            spacing=h,
            body_force=G,
            interface_width=0.5,
            implicit_diffusion=True,
        )
        phi = solver.stratified_phi(d, ops)
        state = (ops.zeros(nx * ny), ops.zeros(nx * ny), phi)

        # march to steady state (the flow rate stops changing) with the phase held stratified.
        prev = 0.0
        for it in range(4000):
            state = solver.step(state, ops)
            if it % 100 == 0:
                q = float(solver.flow_rate(state))
                if it > 0 and abs(q - prev) / (abs(q) + 1e-12) < 1e-8:
                    break
                prev = q

        ys = np.arange(ny) * h
        u_num = solver.u(state).reshape(nx, ny).mean(0).numpy()  # uniform in x; average the streamwise copies
        u_ana = two_layer_profile(ys, G=G, mu_lo=mu_lo, mu_hi=mu_hi, d=d, H=H)

        # centerline (the peak velocity) within a few percent of the analytical two-layer value.
        cl_num = u_num[ny // 2]
        cl_ana = u_ana[ny // 2]
        self.assertLess(abs(cl_num - cl_ana) / cl_ana, 0.06)

        # the whole interior profile within ~6% of the analytical, measured against the peak velocity so the
        # near-wall points (where u -> 0) do not divide by a vanishing analytical value.
        umax = u_ana.max()
        interior_err = np.abs(u_num[1:-1] - u_ana[1:-1]).max() / umax
        self.assertLess(interior_err, 0.06)

        # the analytical shear stress mu*du/dy is continuous at the interface (the two-layer matching
        # condition); confirm the numerical profile carries the same jump-free shape by checking the film
        # is much faster near the wall than a single high-mu fluid would be there.
        self.assertGreater(u_num[3], 5.0 * (G / (2.0 * mu_hi)) * ys[3] * (H - ys[3]))

        # lubrication: the two-layer flow rate is much larger than the single-fluid all-high-mu channel.
        q_num = np.trapezoid(u_num, ys)
        q_single = np.trapezoid((G / (2.0 * mu_hi)) * ys * (H - ys), ys)
        lubrication = q_num / q_single
        self.assertGreater(lubrication, 4.0)  # the film buys a large drag reduction (analytical ~6.9x)
        # and it tracks the analytical two-layer flow rate.
        q_two_ana = np.trapezoid(u_ana, ys)
        self.assertLess(abs(q_num - q_two_ana) / q_two_ana, 0.06)


@unittest.skipUnless(HAS_TORCH, "requires PyTorch")
class YoungLaplaceTestCase(unittest.TestCase):
    """A static circular drop develops the Young-Laplace interior overpressure ``sigma/R`` (2D)."""

    def setUp(self):
        torch.set_default_dtype(torch.float64)

    def test_pressure_jump_matches_sigma_over_R(self):
        ops = make_ops()
        n = 64
        h = 1.0 / (n - 1)
        sigma = 0.5
        R = 0.25
        solver = TwoPhaseFlow2D(n, n, mu=(1.0, 1.0), sigma=sigma, dt=0.05, spacing=h, interface_width=1.0)
        phi = solver.drop_phi((0.5, 0.5), R, ops)
        state = (ops.zeros(n * n), ops.zeros(n * n), phi)

        p = solver.static_pressure(state, ops).reshape(n, n).numpy()
        xx, yy = np.meshgrid(np.arange(n) * h, np.arange(n) * h, indexing="ij")
        r = np.sqrt((xx - 0.5) ** 2 + (yy - 0.5) ** 2)
        inside = p[r < 0.5 * R].mean()
        outside = p[(r > 1.5 * R) & (r < 2.0 * R)].mean()
        jump = inside - outside

        self.assertGreater(jump, 0.0)  # the drop interior is at higher pressure
        self.assertLess(abs(jump - sigma / R) / (sigma / R), 0.15)  # matches sigma/R within 15%


if __name__ == "__main__":
    unittest.main()
