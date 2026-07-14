"""C1 Definition-of-Done: adjoint Jacobians replace the O(n_model) finite-difference loop.

The primary check (``DCResistivityAdjointTest``) is the literal Definition of Done: on a 4x4x4 ERT
toy (``n_model=64``), ``adjoint=True`` must cost O(1) forward factorizations (one, since every
quadrupole here shares a single current injection) instead of the ``2 * n_model`` the finite-difference
Jacobian needs, and the two Jacobians must agree closely (the adjoint is the exact discrete gradient of
the same differentiable forward the finite-difference loop only approximates).

Electrodes/edges must be genuinely INTERIOR to the grid: :func:`mixle_pde.geophysics.dc_resistivity`
(and the 3-D EM solves) impose Dirichlet conditions on the outer boundary, so a corner/boundary
electrode is decoupled from the model entirely (a degenerate all-zero Jacobian that would validate
nothing) -- exactly the trap the module docstrings warn about.

The supplementary MT-3D and CSEM-3D checks confirm the same adjoint machinery generalizes across the
registry's curl-curl operators. MT-3D's comparison runs with ``gauge=0`` because
:func:`mixle_pde.em_diffusion_3d.assemble_curl_curl_3d`'s Coulomb-gauge stabilizer intentionally uses a
DETACHED conductivity (its own docstring: "constant, uses detached sigma") to keep the operator
well-conditioned -- a deliberate, pre-existing non-differentiability of that one term, not something C1
is scoped to touch (no new physics forwards). With the stabilizer disabled the forward is fully
differentiable end to end and the adjoint matches finite differences closely.
"""

from __future__ import annotations

import unittest

import numpy as np

from mixle_pde import pde_solve
from mixle_pde.latent import Field3D
from mixle_pde.observations import (
    Observation,
    csem_3d_forward_operator,
    dc_resistivity_forward_operator,
    mt_3d_forward_operator,
)

try:
    import torch  # noqa: F401

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class _SolveCounter:
    """Wraps a callable, counting invocations -- used to monkeypatch ``pde_solve.sparse_solve``."""

    def __init__(self, real):
        self.real = real
        self.count = 0

    def __call__(self, *args, **kwargs):
        self.count += 1
        return self.real(*args, **kwargs)


class _CountsSolves:
    """Context manager: monkeypatches ``mixle_pde.pde_solve.sparse_solve`` and counts calls."""

    def __enter__(self):
        self._orig = pde_solve.sparse_solve
        self.counter = _SolveCounter(self._orig)
        pde_solve.sparse_solve = self.counter
        return self.counter

    def __exit__(self, *exc):
        pde_solve.sparse_solve = self._orig
        return False


@unittest.skipUnless(HAS_TORCH, "requires PyTorch")
class DCResistivityAdjointTest(unittest.TestCase):
    """The C1 Definition of Done."""

    def _toy(self):
        shape = (4, 4, 4)
        idx = np.arange(np.prod(shape)).reshape(shape)
        # a single current injection at two INTERIOR electrodes, read at three interior potential
        # dipoles -- one unique (a, b) injection means one forward factorization, adjoint=True or not.
        a, b = int(idx[1, 1, 1]), int(idx[2, 2, 2])
        schedule = [
            (a, b, int(idx[1, 1, 2]), int(idx[2, 1, 1])),
            (a, b, int(idx[1, 2, 1]), int(idx[2, 2, 1])),
            (a, b, int(idx[2, 1, 2]), int(idx[1, 2, 2])),
        ]
        coords = np.array([[x, y, z] for x in range(4) for y in range(4) for z in range(4)], dtype=float)
        grid = Field3D(coords, spacing=1.0, units="log(S/m)", property_name="log_conductivity")
        n_model = int(np.prod(shape))
        rng = np.random.default_rng(0)
        values = 0.05 * rng.standard_normal(n_model)
        locations = np.zeros((len(schedule), 3))
        observation = Observation("dc_resistivity", locations, np.zeros(len(schedule)), np.ones(len(schedule)))
        return shape, schedule, grid, values, observation, n_model

    def test_adjoint_matches_finite_difference_with_far_fewer_solves(self):
        shape, schedule, grid, values, observation, n_model = self._toy()

        op_fd = dc_resistivity_forward_operator(shape, schedule, sigma_ref=0.02, finite_difference_step=1.0e-6)
        with _CountsSolves() as counter_fd:
            jac_fd = op_fd.local_jacobian(grid, values, observation)
        self.assertEqual(counter_fd.count, 2 * n_model)  # the FD reference: 2 evals per model parameter
        self.assertGreater(np.count_nonzero(jac_fd), 0)  # sanity: electrodes are interior, not degenerate

        op_adjoint = dc_resistivity_forward_operator(shape, schedule, sigma_ref=0.02, adjoint=True)
        with _CountsSolves() as counter_adjoint:
            jac_adjoint = op_adjoint.local_jacobian(grid, values, observation)

        self.assertLessEqual(counter_adjoint.count, 2)
        self.assertLess(counter_adjoint.count, counter_fd.count)
        np.testing.assert_allclose(jac_adjoint, jac_fd, rtol=1.0e-4)

        self.assertEqual(op_adjoint.jacobian_kind, "adjoint")
        self.assertTrue(op_adjoint.has_true_adjoint)
        report = op_adjoint.capability_report()
        self.assertTrue(report["has_true_adjoint"])
        self.assertEqual(report["jacobian_kind"], "adjoint")

    def test_default_operator_is_unchanged_finite_difference(self):
        # adjoint=False (the default) must be byte-for-byte the pre-C1 behavior.
        shape, schedule, grid, values, observation, _ = self._toy()
        op = dc_resistivity_forward_operator(shape, schedule, sigma_ref=0.02)
        self.assertEqual(op.jacobian_kind, "finite_difference")
        self.assertFalse(op.has_true_adjoint)


@unittest.skipUnless(HAS_TORCH, "requires PyTorch")
class MT3DAdjointTest(unittest.TestCase):
    def test_adjoint_matches_finite_difference_with_far_fewer_solves(self):
        shape = (3, 3, 6)
        freqs = np.array([5.0, 20.0])
        nx, ny, nz = shape
        coords = np.array([[float(i), float(j), -float(k)] for i in range(nx) for j in range(ny) for k in range(nz)])
        grid = Field3D(coords, spacing=50.0, units="log(S/m)", property_name="log_conductivity_3d")
        values = np.zeros(grid.n)
        locations = np.column_stack([freqs, np.zeros_like(freqs), np.zeros_like(freqs)])
        observation = Observation(
            "mt_3d_log_apparent_resistivity", locations, np.zeros(len(freqs)), np.ones(len(freqs))
        )

        # gauge=0 disables the Coulomb-gauge stabilizer, which intentionally uses a detached
        # conductivity (assemble_curl_curl_3d's docstring) and so is not part of the differentiable
        # graph; with it off the forward is fully differentiable and the adjoint is exact.
        op_fd = mt_3d_forward_operator(
            shape, freqs, spacing=50.0, sigma_ref=0.05, gauge=0.0, finite_difference_step=1.0e-6
        )
        jac_fd = op_fd.local_jacobian(grid, values, observation)
        self.assertGreater(np.count_nonzero(jac_fd), 0)

        op_adjoint = mt_3d_forward_operator(shape, freqs, spacing=50.0, sigma_ref=0.05, gauge=0.0, adjoint=True)
        with _CountsSolves() as counter:
            jac_adjoint = op_adjoint.local_jacobian(grid, values, observation)

        self.assertLessEqual(counter.count, len(freqs))  # one factorization per sounding frequency
        np.testing.assert_allclose(jac_adjoint, jac_fd, rtol=1.0e-3, atol=5.0e-2)
        self.assertEqual(op_adjoint.jacobian_kind, "adjoint")
        self.assertTrue(op_adjoint.has_true_adjoint)


@unittest.skipUnless(HAS_TORCH, "requires PyTorch")
class CSEM3DAdjointTest(unittest.TestCase):
    def test_adjoint_matches_finite_difference_with_far_fewer_solves(self):
        from mixle_pde.em_diffusion_3d import _edge_layout

        shape = (5, 5, 5)
        nx, ny, nz = shape
        _, _, _, _, (sx, _, _) = _edge_layout(shape)
        ic, jc, kc = nx // 2, ny // 2, nz // 2
        src_edge = (ic * sx[1] + jc) * sx[2] + kc
        # receivers offset from the source (not the same near-null edge) so the "real" component is
        # well away from a field zero -- log_amplitude at a field null would make BOTH the
        # finite-difference and the exact derivative blow up (a log-of-near-zero artifact, not an
        # adjoint bug), so this uses the plain real-part component instead.
        rcv_edges = [
            ((ic + 1) * sx[1] + jc) * sx[2] + kc,
            (ic * sx[1] + (jc + 1)) * sx[2] + kc,
        ]
        coords = np.array([[float(i), float(j), -float(k)] for i in range(nx) for j in range(ny) for k in range(nz)])
        grid = Field3D(coords, spacing=1.0, units="log(S/m)", property_name="log_conductivity_3d")
        values = np.zeros(grid.n)
        locations = np.zeros((len(rcv_edges), 3))
        observation = Observation("csem_3d_real", locations, np.zeros(len(rcv_edges)), np.ones(len(rcv_edges)))

        op_fd = csem_3d_forward_operator(
            shape, 50.0, [src_edge], rcv_edges, component="real", sigma_ref=0.05, finite_difference_step=1.0e-6
        )
        jac_fd = op_fd.local_jacobian(grid, values, observation)
        self.assertGreater(np.count_nonzero(jac_fd), 0)

        op_adjoint = csem_3d_forward_operator(
            shape, 50.0, [src_edge], rcv_edges, component="real", sigma_ref=0.05, adjoint=True
        )
        with _CountsSolves() as counter:
            jac_adjoint = op_adjoint.local_jacobian(grid, values, observation)

        self.assertLessEqual(counter.count, 1)  # one shared curl-curl factorization for the one source
        np.testing.assert_allclose(jac_adjoint, jac_fd, rtol=1.0e-4, atol=1.0e-8)
        self.assertEqual(op_adjoint.jacobian_kind, "adjoint")
        self.assertTrue(op_adjoint.has_true_adjoint)


@unittest.skipUnless(HAS_TORCH, "requires PyTorch")
class TorchAdjointJVPTest(unittest.TestCase):
    def test_jvp_matches_full_jacobian_contraction(self):
        from mixle_pde.adjoint import torch_adjoint_jacobian, torch_adjoint_jvp
        from mixle_pde.geophysics import dc_resistivity

        shape = (4, 4, 4)
        idx = np.arange(np.prod(shape)).reshape(shape)
        a, b = int(idx[1, 1, 1]), int(idx[2, 2, 2])
        schedule = [
            (a, b, int(idx[1, 1, 2]), int(idx[2, 1, 1])),
            (a, b, int(idx[1, 2, 1]), int(idx[2, 2, 1])),
        ]

        def predict_torch(x_t):
            return dc_resistivity(x_t, shape, schedule, sigma_ref=0.02, log_data=True)

        rng = np.random.default_rng(3)
        x0 = 0.05 * rng.standard_normal(int(np.prod(shape)))
        v0 = rng.standard_normal(int(np.prod(shape)))

        jvp = torch_adjoint_jvp(predict_torch, x0, v0)
        jac = torch_adjoint_jacobian(predict_torch, x0, n_obs=len(schedule))
        np.testing.assert_allclose(jvp, jac @ v0, rtol=1.0e-3)


if __name__ == "__main__":
    unittest.main()
