"""Edge-preserving priors over a PDE/ODE Differential forward: complex-valued data and multistart.

The Gaussian-forward TotalVariation/Potts tests stay in mixle (priors_test.py); these use the
``Differential`` forward operator from pysparkplug-pde.
"""

import unittest

import numpy as np

try:
    import torch  # noqa: F401

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

if HAS_TORCH:
    from mixle.ppl import GP, RandomWalk, free, joint, multistart

    from mixle_pde import Differential


@unittest.skipUnless(HAS_TORCH, "requires PyTorch")
class ComplexDataTestCase(unittest.TestCase):
    def setUp(self):
        torch.set_default_dtype(torch.float64)

    def test_recovers_field_from_complex_observations(self):
        m = 20
        rng = np.random.RandomState(1)
        A = rng.randn(12, m) + 1j * rng.randn(12, m)
        f_true = np.sin(2 * np.pi * np.linspace(0, 1, m))
        y = A @ f_true + 0.01 * (rng.randn(12) + 1j * rng.randn(12))
        At = torch.as_tensor(A)
        fld = GP("h", index=np.arange(m), kernel=RandomWalk(scale=0.5, ridge=3.0))
        obs = Differential(y, over=fld, scale=0.01, forward=lambda p, ops: At @ p.field.to(torch.complex128))
        hm, hs = joint([obs]).fit(how="laplace").posterior("h")
        self.assertGreater(np.corrcoef(hm, f_true)[0, 1], 0.95)
        self.assertTrue(np.all(hs > 0))


@unittest.skipUnless(HAS_TORCH, "requires PyTorch")
class MultistartTestCase(unittest.TestCase):
    def setUp(self):
        torch.set_default_dtype(torch.float64)

    def test_escapes_local_minimum(self):
        t = np.linspace(0, 2, 40)
        w_true = 3.0
        y = np.sin(w_true * t) + 0.02 * np.random.RandomState(2).randn(40)
        w = free(1, name="w", support="positive")
        model = joint(
            [
                Differential(
                    y, drivers=[w], y0=0.0, t_grid=t, scale=0.02, rhs=lambda u, tt, p, ops: p.w * ops.cos(p.w * tt)
                )
            ]
        )
        bad = model.fit(how="laplace", init={"w": np.log(7.0)}).mean("w")
        best = multistart(model, [{"w": np.log(7.0)}, {"w": np.log(2.0)}], how="laplace").mean("w")
        self.assertGreater(abs(bad - w_true), 1.0)  # the bad init is stuck in a local mode
        self.assertLess(abs(best - w_true), 0.5)  # multistart finds the global one


if __name__ == "__main__":
    unittest.main()
