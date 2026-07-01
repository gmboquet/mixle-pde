"""Mechanistic field reasoning: reconstruct a PDE field from sparse sensors (mixle_pde.reasoning)."""

import unittest

import numpy as np

from mixle_pde.dynamics import DiffusionOperator
from mixle_pde.reasoning import MechanisticFieldReasoner


def _true_diffusion(op, dt, steps, x0):
    A = op.transition_matrix(dt)
    u = [np.asarray(x0, dtype=float)]
    for _ in range(1, steps):
        u.append(A @ u[-1])
    return np.array(u)  # (steps, n)


class MechanisticFieldReasonerTest(unittest.TestCase):
    def setUp(self):
        self.n, self.steps, self.dt = 12, 8, 0.5
        self.op = DiffusionOperator(0.15, self.n, length=float(self.n), bc="neumann")
        x0 = np.exp(-((np.arange(self.n) - 6.0) ** 2) / 2.0)  # a bump
        self.u_true = _true_diffusion(self.op, self.dt, self.steps, x0)
        self.reasoner = MechanisticFieldReasoner(self.op, dt=self.dt, steps=self.steps, x0_sd=2.0, process_sd=0.01)

    def test_prior_shape(self):
        self.assertEqual(np.size(self.reasoner.prior.mean()), self.n * self.steps)

    def test_sparse_sensors_reconstruct_the_field(self):
        # a handful of sensors in space and time; the diffusion physics fills the rest.
        rng = np.random.RandomState(0)
        sensors = []
        for step in (0, 4):  # only two time slices observed
            for cell in (2, 4, 6, 8, 10):  # five sensors spanning (and including) the bump
                sensors.append(
                    self.reasoner.sensor(
                        cell=cell,
                        step=step,
                        value=self.u_true[step, cell] + rng.normal(0, 0.01),
                        noise_sd=0.02,
                    )
                )
        ans = self.reasoner.reason(sensors)
        recon = self.reasoner.field(ans)
        # whole space-time field recovered from 10 sensors at 2 times over 96 unknowns
        self.assertGreater(np.corrcoef(recon.ravel(), self.u_true.ravel())[0, 1], 0.9)
        # including an unobserved time slice (t=5) filled in by the dynamics
        self.assertLess(np.linalg.norm(recon[5] - self.u_true[5]) / self.n, 0.05)

    def test_uncertainty_lower_near_sensors(self):
        s = [self.reasoner.sensor(cell=6, step=0, value=self.u_true[0, 6], noise_sd=0.02)]
        ans = self.reasoner.reason(s)
        sd = self.reasoner.uncertainty(ans)
        prior_sd = self.reasoner.uncertainty(self.reasoner.reason([]))
        # the observed cell/time is much more certain than under the prior
        self.assertLess(sd[0, 6], prior_sd[0, 6])
        # and information propagates forward in time (later steps also sharpen)
        self.assertLess(sd[3, 6], prior_sd[3, 6])

    def test_no_sensors_returns_prior(self):
        ans = self.reasoner.reason([])
        np.testing.assert_allclose(ans.mean, self.reasoner.prior.mean(), atol=1e-12)


if __name__ == "__main__":
    unittest.main()
