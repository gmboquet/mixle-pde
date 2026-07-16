"""Checkpoint/restart for metropolis_field_invert (MP-I9 checkpoint/restart slice).

Acceptance evidence: a checkpointed run at an awkward (non-dividing) cadence must reproduce an
uninterrupted run bit-for-bit, and a run genuinely interrupted mid-chain, serialized to disk, and
resumed from a freshly restored RNG in what stands in for a new process must reproduce the same
uninterrupted reference too.
"""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from mixle_pde.field_inversion import FieldGaussianPrior
from mixle_pde.field_mcmc import metropolis_field_invert
from mixle_pde.latent import Field3D, PosteriorFieldSamples3D
from mixle_pde.mcmc_checkpoint import (
    MCMCCheckpoint,
    load_checkpoint,
    resume_checkpointed,
    run_checkpointed,
    save_checkpoint,
)
from mixle_pde.observations import ForwardOperatorRegistry, Observation, borehole_forward_operator


class _StopAfter(Exception):
    """Raised from an ``on_checkpoint`` callback to simulate a process being interrupted right after
    a checkpoint was captured (and, in the disk-backed test, already persisted)."""

    def __init__(self, checkpoint: MCMCCheckpoint):
        super().__init__(f"stopped at iteration={checkpoint.iteration}")
        self.checkpoint = checkpoint


def _stop_once_past(iteration_threshold: int):
    def _on_checkpoint(checkpoint: MCMCCheckpoint) -> None:
        if checkpoint.iteration >= iteration_threshold:
            raise _StopAfter(checkpoint)

    return _on_checkpoint


class MCMCCheckpointTest(unittest.TestCase):
    def _bounded_problem(self):
        grid = Field3D(
            coordinates=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            spacing=1.0,
            units="fraction",
            property_name="porosity",
            bounds=(0.0, 1.0),
        )
        registry = ForwardOperatorRegistry()
        registry.register(borehole_forward_operator())
        truth = np.array([0.25, 0.75])
        observation = Observation("borehole", grid.coordinates, truth, np.full(2, 0.03**2))
        prior = FieldGaussianPrior(
            mean=grid.to_unconstrained(np.full(grid.n, 0.5)),
            smoothness_precision=0.02,
            marginal_precision=0.1,
            length_scale=1.0,
        )
        return grid, [observation], registry, prior

    def test_checkpointed_run_matches_uninterrupted_run_at_awkward_cadence(self):
        grid, observations, registry, prior = self._bounded_problem()
        n_samples, burn_in, thin, step_scale = 300, 80, 3, np.full(grid.n, 0.25)
        seed = 1234

        reference_posterior, reference_report = metropolis_field_invert(
            grid,
            observations,
            registry,
            prior,
            n_samples=n_samples,
            burn_in=burn_in,
            thin=thin,
            step_scale=step_scale,
            rng=np.random.default_rng(seed),
        )

        checkpoints: list[MCMCCheckpoint] = []
        posterior, report, final_checkpoint = run_checkpointed(
            grid,
            observations,
            registry,
            prior,
            n_samples=n_samples,
            burn_in=burn_in,
            thin=thin,
            step_scale=step_scale,
            rng=np.random.default_rng(seed),
            checkpoint_every=37,  # does not evenly divide burn_in, thin, or n_samples * thin
            on_checkpoint=checkpoints.append,
        )

        # A genuine multi-segment run happened (37 does not divide 980 total raw steps).
        self.assertGreater(len(checkpoints), 1)
        self.assertEqual(checkpoints[-1].iteration, final_checkpoint.iteration)
        self.assertEqual(final_checkpoint.iteration, final_checkpoint.total_steps)

        self.assertIsInstance(posterior, PosteriorFieldSamples3D)
        np.testing.assert_array_equal(posterior.samples, reference_posterior.samples)
        np.testing.assert_array_equal(posterior.log_posterior, reference_posterior.log_posterior)
        np.testing.assert_array_equal(posterior.map, reference_posterior.map)
        self.assertEqual(report.accepted, reference_report.accepted)
        self.assertEqual(report.stored_samples, reference_report.stored_samples)
        self.assertEqual(report.final_log_posterior, reference_report.final_log_posterior)
        self.assertEqual(report.best_log_posterior, reference_report.best_log_posterior)

    def test_interrupted_run_resumes_bit_reproducibly_from_a_disk_checkpoint(self):
        grid, observations, registry, prior = self._bounded_problem()
        n_samples, burn_in, thin, step_scale = 300, 80, 3, np.full(grid.n, 0.25)
        seed = 777

        reference_posterior, reference_report = metropolis_field_invert(
            grid,
            observations,
            registry,
            prior,
            n_samples=n_samples,
            burn_in=burn_in,
            thin=thin,
            step_scale=step_scale,
            rng=np.random.default_rng(seed),
        )

        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_path = Path(tmp) / "chain"
            try:
                run_checkpointed(
                    grid,
                    observations,
                    registry,
                    prior,
                    n_samples=n_samples,
                    burn_in=burn_in,
                    thin=thin,
                    step_scale=step_scale,
                    rng=np.random.default_rng(seed),
                    checkpoint_every=29,
                    on_checkpoint=lambda cp: (save_checkpoint(cp, checkpoint_path), _stop_once_past(133)(cp)),
                )
                self.fail("expected the run to be interrupted before completion")
            except _StopAfter as stop:
                interrupted_at = stop.checkpoint.iteration
            self.assertGreater(interrupted_at, 0)
            self.assertLess(interrupted_at, reference_report.iterations)

            # Simulate a new process: nothing survives but the file on disk.
            loaded = load_checkpoint(checkpoint_path)
            self.assertEqual(loaded.iteration, interrupted_at)
            self.assertEqual(loaded.rng_bit_generator, "PCG64")

            posterior, report, final_checkpoint = resume_checkpointed(
                loaded,
                grid,
                observations,
                registry,
                prior,
                checkpoint_every=41,  # a different cadence than the interrupted run used
            )

        self.assertEqual(final_checkpoint.iteration, final_checkpoint.total_steps)
        np.testing.assert_array_equal(posterior.samples, reference_posterior.samples)
        np.testing.assert_array_equal(posterior.log_posterior, reference_posterior.log_posterior)
        np.testing.assert_array_equal(posterior.map, reference_posterior.map)
        self.assertEqual(report.accepted, reference_report.accepted)
        self.assertEqual(report.stored_samples, reference_report.stored_samples)
        self.assertEqual(report.best_log_posterior, reference_report.best_log_posterior)

    def test_resuming_an_already_complete_checkpoint_is_a_no_op(self):
        grid, observations, registry, prior = self._bounded_problem()
        _, _, checkpoint = run_checkpointed(
            grid,
            observations,
            registry,
            prior,
            n_samples=50,
            burn_in=10,
            thin=1,
            step_scale=np.full(grid.n, 0.25),
            rng=np.random.default_rng(5),
            checkpoint_every=1000,
        )
        self.assertEqual(checkpoint.iteration, checkpoint.total_steps)

        posterior, report, resumed_checkpoint = resume_checkpointed(checkpoint, grid, observations, registry, prior)
        np.testing.assert_array_equal(posterior.samples, checkpoint.stored_samples)
        self.assertEqual(report.accepted, checkpoint.accepted)
        self.assertEqual(resumed_checkpoint.iteration, checkpoint.iteration)

    def test_resume_rejects_a_mismatched_problem(self):
        grid, observations, registry, prior = self._bounded_problem()
        _, _, checkpoint = run_checkpointed(
            grid,
            observations,
            registry,
            prior,
            n_samples=20,
            burn_in=5,
            thin=1,
            step_scale=np.full(grid.n, 0.25),
            rng=np.random.default_rng(2),
            checkpoint_every=1000,
        )

        other_grid = Field3D(
            coordinates=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            spacing=1.0,
            units="fraction",
            property_name="density",  # deliberately different property than _bounded_problem's "porosity"
            bounds=(0.0, 1.0),
        )
        with self.assertRaises(ValueError):
            resume_checkpointed(checkpoint, other_grid, observations, registry, prior)

    def test_checkpoint_rejects_malformed_arrays(self):
        grid, observations, registry, prior = self._bounded_problem()
        _, _, checkpoint = run_checkpointed(
            grid,
            observations,
            registry,
            prior,
            n_samples=20,
            burn_in=5,
            thin=1,
            step_scale=np.full(grid.n, 0.25),
            rng=np.random.default_rng(9),
            checkpoint_every=1000,
        )
        with self.assertRaises(ValueError):
            MCMCCheckpoint(
                schema=checkpoint.schema,
                iteration=checkpoint.iteration,
                total_steps=checkpoint.total_steps,
                burn_in=checkpoint.burn_in,
                thin=checkpoint.thin,
                n_samples=checkpoint.n_samples,
                grid_n=checkpoint.grid_n,
                step_scale=checkpoint.step_scale,
                current=np.zeros(checkpoint.grid_n + 1),  # wrong shape
                current_logp=checkpoint.current_logp,
                best=checkpoint.best,
                best_logp=checkpoint.best_logp,
                accepted=checkpoint.accepted,
                stored_samples=checkpoint.stored_samples,
                stored_log_posterior=checkpoint.stored_log_posterior,
                rng_bit_generator=checkpoint.rng_bit_generator,
                rng_state=checkpoint.rng_state,
                config_digest=checkpoint.config_digest,
            )


if __name__ == "__main__":
    unittest.main()
