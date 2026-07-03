"""Tests for the differentiable nonlinear steady solver (Newton + implicit-function-theorem backward).

Acceptance bar is agreement with the manufactured analytical solution and a central finite difference for the
implicit-adjoint gradient, not merely that the solver runs.
"""

import importlib.util
import unittest

import numpy as np

HAS_TORCH = importlib.util.find_spec("torch") is not None

if HAS_TORCH:
    import torch

    from mixle_pde.nonlinear import nonlinear_solve, reaction_diffusion_residual


def _grid(m):
    h = 1.0 / (m - 1)
    xx, yy = np.meshgrid(np.linspace(0, 1, m), np.linspace(0, 1, m), indexing="ij")
    return h, xx, yy


@unittest.skipUnless(HAS_TORCH, "requires PyTorch")
class ManufacturedNonlinearPoissonTestCase(unittest.TestCase):
    def setUp(self):
        torch.set_default_dtype(torch.float64)

    def test_cubic_reaction_matches_manufactured_solution(self):
        # -lap u + theta * u^3 = f  on [0,1]^2, Dirichlet.  Manufactured u* = sin(pi x) sin(pi y).
        m = 41
        h, xx, yy = _grid(m)
        shape = (m, m)
        theta = torch.tensor(1.7, dtype=torch.float64)
        ustar = np.sin(np.pi * xx) * np.sin(np.pi * yy)
        # f = -lap u* + theta u*^3 = 2 pi^2 u* + theta u*^3
        f = 2 * np.pi**2 * ustar + float(theta) * ustar**3
        # boundary nodes carry the Dirichlet value (u* = 0 there) via the source
        f_full = f.ravel().copy()
        bnd = np.zeros((m, m), bool)
        bnd[0] = bnd[-1] = bnd[:, 0] = bnd[:, -1] = True
        f_full[bnd.ravel()] = ustar.ravel()[bnd.ravel()]

        def g(u, th):
            return th * u**3

        def dg(u, th):
            return 3.0 * th * u**2

        residual_fn, jac_fn = reaction_diffusion_residual(shape, f_full, g, dg, spacing=h)
        u0 = torch.zeros(m * m, dtype=torch.float64)
        u = nonlinear_solve(residual_fn, jac_fn, u0, theta, max_its=50)
        err = float((u - torch.as_tensor(ustar.ravel())).abs().max())
        self.assertLess(err, 1e-3)

    def test_bratu_matches_manufactured_solution(self):
        # -lap u + lambda exp(u) = f  (Bratu-type nonlinearity), manufactured u* = sin(pi x) sin(pi y).
        m = 41
        h, xx, yy = _grid(m)
        shape = (m, m)
        lam = torch.tensor(0.9, dtype=torch.float64)
        ustar = np.sin(np.pi * xx) * np.sin(np.pi * yy)
        f = 2 * np.pi**2 * ustar + float(lam) * np.exp(ustar)
        f_full = f.ravel().copy()
        bnd = np.zeros((m, m), bool)
        bnd[0] = bnd[-1] = bnd[:, 0] = bnd[:, -1] = True
        f_full[bnd.ravel()] = ustar.ravel()[bnd.ravel()]

        def g(u, th):
            return th * torch.exp(u)

        def dg(u, th):
            return th * torch.exp(u)

        residual_fn, jac_fn = reaction_diffusion_residual(shape, f_full, g, dg, spacing=h)
        u0 = torch.zeros(m * m, dtype=torch.float64)
        u = nonlinear_solve(residual_fn, jac_fn, u0, lam, max_its=50)
        err = float((u - torch.as_tensor(ustar.ravel())).abs().max())
        self.assertLess(err, 1e-3)


@unittest.skipUnless(HAS_TORCH, "requires PyTorch")
class ImplicitAdjointGradientTestCase(unittest.TestCase):
    def setUp(self):
        torch.set_default_dtype(torch.float64)

    def test_gradient_matches_central_difference(self):
        # loss = 0.5 ||u(theta) - u_target||^2 for the reaction coefficient theta; d loss/d theta from
        # the implicit-adjoint backward must match a central finite difference.
        m = 21
        h, xx, yy = _grid(m)
        shape = (m, m)
        # a fixed source and an (arbitrary, off-solution) target so the loss has a nonzero gradient in theta
        f_full = np.full(m * m, 5.0)  # constant interior forcing
        bnd = np.zeros((m, m), bool)
        bnd[0] = bnd[-1] = bnd[:, 0] = bnd[:, -1] = True
        f_full[bnd.ravel()] = 0.0  # homogeneous Dirichlet
        u_target = torch.as_tensor((np.sin(2 * np.pi * xx) * np.sin(np.pi * yy)).ravel())
        u0 = torch.zeros(m * m, dtype=torch.float64)

        theta0 = 0.8
        theta_r = torch.tensor(theta0, dtype=torch.float64, requires_grad=True)

        def g(u, th):
            return th * u**3

        def dg(u, th):
            return 3.0 * th * u**2

        residual_fn, jac_fn = reaction_diffusion_residual(shape, f_full, g, dg, spacing=h)
        u = nonlinear_solve(residual_fn, jac_fn, u0, theta_r, max_its=50)
        loss = 0.5 * ((u - u_target) ** 2).sum()
        loss.backward()
        g_auto = float(theta_r.grad)

        eps = 1e-5

        def loss_at(tv):
            t = torch.tensor(tv, dtype=torch.float64)
            uu = nonlinear_solve(residual_fn, jac_fn, u0, t, max_its=50)
            return float(0.5 * ((uu - u_target) ** 2).sum())

        g_fd = (loss_at(theta0 + eps) - loss_at(theta0 - eps)) / (2 * eps)
        self.assertLess(abs(g_auto - g_fd), 1e-4 * max(1.0, abs(g_fd)))

    def test_vector_theta_gradient_matches_finite_difference(self):
        # theta is a length-2 vector: g(u; theta) = theta0 * u^3 + theta1 * u.  Checks the VJP handles a
        # vector parameter (the reusable-for-PNP requirement).
        m = 17
        h, xx, yy = _grid(m)
        shape = (m, m)
        f_full = np.full(m * m, 4.0)
        bnd = np.zeros((m, m), bool)
        bnd[0] = bnd[-1] = bnd[:, 0] = bnd[:, -1] = True
        f_full[bnd.ravel()] = 0.0
        u_target = torch.as_tensor((np.sin(np.pi * xx) * np.cos(np.pi * yy) * 0.3).ravel())
        u0 = torch.zeros(m * m, dtype=torch.float64)

        def g(u, th):
            return th[0] * u**3 + th[1] * u

        def dg(u, th):
            return 3.0 * th[0] * u**2 + th[1]

        residual_fn, jac_fn = reaction_diffusion_residual(shape, f_full, g, dg, spacing=h)
        theta0 = np.array([0.7, 1.3])
        theta_r = torch.tensor(theta0, dtype=torch.float64, requires_grad=True)
        u = nonlinear_solve(residual_fn, jac_fn, u0, theta_r, max_its=50)
        loss = 0.5 * ((u - u_target) ** 2).sum()
        loss.backward()
        g_auto = theta_r.grad.detach().numpy().copy()

        eps = 1e-5
        g_fd = np.zeros(2)
        for k in range(2):
            tp = theta0.copy()
            tp[k] += eps
            tm = theta0.copy()
            tm[k] -= eps
            up = nonlinear_solve(residual_fn, jac_fn, u0, torch.tensor(tp), max_its=50)
            um = nonlinear_solve(residual_fn, jac_fn, u0, torch.tensor(tm), max_its=50)
            lp = float(0.5 * ((up - u_target) ** 2).sum())
            lm = float(0.5 * ((um - u_target) ** 2).sum())
            g_fd[k] = (lp - lm) / (2 * eps)
        self.assertLess(float(np.abs(g_auto - g_fd).max()), 1e-4 * max(1.0, float(np.abs(g_fd).max())))


if __name__ == "__main__":
    unittest.main()
