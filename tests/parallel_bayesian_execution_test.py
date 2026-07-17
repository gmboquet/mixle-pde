"""Parallel, resumable, reproducible multi-chain Bayesian execution (MP-I9 baseline).

Acceptance evidence exercised here, mirroring MP-I9's own accept criterion:

* "serial/parallel analytic posterior summaries agree within Monte Carlo error": pooled samples from
  several chains genuinely run in separate OS processes are compared against
  :func:`mixle_pde.field_inversion.linear_gaussian_invert`'s exact closed-form posterior (the model is
  linear-Gaussian by construction: an identity-transform field + a linear borehole observation + a
  Gaussian prior), not an approximation of one.
* "interrupted multi-chain ... jobs resume without duplicating samples or changing the target
  distribution": a batch is genuinely interrupted mid-run (mirroring
  tests/mcmc_checkpoint_test.py's own interruption pattern), resumed across fresh worker processes, and
  checked bit-for-bit against an uninterrupted reference run with the same seed.
* "every forward call is attributable": every outcome carries which worker process (OS pid) ran it, how
  many attempts it took, and its wall-clock time.

Also covers genuine multiprocessing (distinct OS pids), failure-aware retry (transient and terminal),
and wall-clock budget/cancellation.
"""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from mixle_pde.field_inversion import FieldGaussianPrior, linear_gaussian_invert
from mixle_pde.job_governance import ResourceBudget
from mixle_pde.latent import Field3D
from mixle_pde.mcmc_checkpoint import MCMCCheckpoint, save_checkpoint
from mixle_pde.observations import ForwardOperatorRegistry, Observation, borehole_forward_operator
from mixle_pde.parallel_bayesian_execution import (
    ChainOutcome,
    ParallelChainRun,
    resume_parallel_chains,
    run_parallel_chains,
    spawn_counter_streams,
)
from mixle_pde.verification.mcmc_diagnostics import chains_from_posterior_samples, evaluate_chain_convergence

# ``ProcessPoolExecutor`` pickles every argument it sends to a worker, so the "problem" a scheduled
# chain samples must be rebuilt from a module-level (picklable) factory rather than passed as an
# already-built object -- see mixle_pde/parallel_bayesian_execution.py's module docstring for why
# (ForwardOperatorRegistry holds closures that plain pickle cannot serialize).

_TRUTH = np.array([2.0, -1.0])
_NOISE_VAR = 0.25
_STEP_SCALE = 0.5


def _analytic_problem() -> tuple[Field3D, list[Observation], ForwardOperatorRegistry, FieldGaussianPrior]:
    """Two-cell, identity-transform (bounds=None), linear-Gaussian field inversion problem.

    ``bounds=None`` keeps the unconstrained-space transform the identity, and ``borehole_forward_operator``
    is a fixed linear (0/1 selection) operator, so this model is exactly conjugate:
    :func:`mixle_pde.field_inversion.linear_gaussian_invert` gives the *exact* posterior, not an
    approximation, to compare sampled output against.
    """
    grid = Field3D(
        coordinates=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        spacing=1.0,
        units="kg/m^3",
        property_name="density",
    )
    registry = ForwardOperatorRegistry()
    registry.register(borehole_forward_operator())
    observation = Observation("borehole", grid.coordinates, value=_TRUTH.copy(), noise_cov=np.full(2, _NOISE_VAR))
    prior = FieldGaussianPrior(mean=0.0, smoothness_precision=0.4, marginal_precision=1.0)
    return grid, [observation], registry, prior


class _CountedFailureFactory:
    """Picklable callable: raises for the first ``fail_first_n_calls`` invocations (tracked via a
    marker file, so it works across independently-spawned worker processes), then delegates to
    :func:`_analytic_problem`. A plain module-level function bound via ``functools.partial`` would work
    too; a small callable class keeps the marker-file bookkeeping self-contained.
    """

    def __init__(self, marker_dir: str, fail_first_n_calls: int) -> None:
        self.marker_dir = marker_dir
        self.fail_first_n_calls = fail_first_n_calls

    def __call__(self):
        marker_dir = Path(self.marker_dir)
        marker_dir.mkdir(parents=True, exist_ok=True)
        marker = marker_dir / f"attempt-{len(list(marker_dir.iterdir()))}.marker"
        marker.write_text("x")
        attempt_number = len(list(marker_dir.iterdir()))
        if attempt_number <= self.fail_first_n_calls:
            raise RuntimeError(f"synthetic transient failure (attempt {attempt_number})")
        return _analytic_problem()


def _total_steps(n_samples: int, burn_in: int, thin: int) -> int:
    return burn_in + n_samples * thin


class SpawnCounterStreamsTest(unittest.TestCase):
    def test_streams_are_independent_and_reproducible(self):
        streams_a = spawn_counter_streams(2024, 4)
        streams_b = spawn_counter_streams(2024, 4)
        self.assertEqual(len(streams_a), 4)
        for sa, sb in zip(streams_a, streams_b, strict=True):
            np.testing.assert_array_equal(sa.random(50), sb.random(50))

        # Different spawn index -> a genuinely different stream, not the same sequence replayed.
        fresh = spawn_counter_streams(2024, 4)
        draws = [g.random(50) for g in fresh]
        for i in range(4):
            for j in range(i + 1, 4):
                self.assertFalse(np.array_equal(draws[i], draws[j]))

        # Counter-based (Philox), not the package default.
        for g in spawn_counter_streams(1, 2):
            self.assertEqual(type(g.bit_generator).__name__, "Philox")

    def test_rejects_non_positive_stream_count(self):
        with self.assertRaises(ValueError):
            spawn_counter_streams(1, 0)


class RunParallelChainsTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp(prefix="mp_i9_")
        self.addCleanup(shutil.rmtree, self._tmp, ignore_errors=True)

    def _checkpoint_dir(self, name: str) -> str:
        path = Path(self._tmp) / name
        path.mkdir(parents=True, exist_ok=True)
        return str(path)

    def test_pooled_parallel_posterior_matches_analytic_reference_with_real_multiprocessing(self):
        grid, observations, registry, prior = _analytic_problem()
        reference = linear_gaussian_invert(grid, observations, registry, prior)

        n_chains, n_samples, burn_in = 4, 2500, 500
        result = run_parallel_chains(
            _analytic_problem,
            n_chains=n_chains,
            n_samples=n_samples,
            burn_in=burn_in,
            thin=1,
            step_scale=_STEP_SCALE,
            seed=20260717,
            checkpoint_dir=self._checkpoint_dir("pooled"),
            max_workers=4,
        )

        self.assertIsInstance(result, ParallelChainRun)
        self.assertEqual(result.n_chains, n_chains)
        self.assertEqual(result.status_counts(), {"completed": n_chains, "failed": 0, "cancelled": 0})
        for chain_id, outcome in enumerate(result.outcomes):
            self.assertEqual(outcome.chain_id, chain_id)
            self.assertEqual(outcome.status, "completed")
            self.assertEqual(outcome.posterior.samples.shape, (n_samples, 2))
            self.assertEqual(len(outcome.worker_pids), 1)

        # Genuine OS-level multiprocessing: more than one distinct worker process ran the batch.
        pids = {pid for outcome in result.outcomes for pid in outcome.worker_pids}
        self.assertGreater(len(pids), 1, "expected chains to run in more than one OS process")

        pooled = result.pooled_samples()
        self.assertEqual(pooled.shape, (n_chains * n_samples, 2))
        pooled_mean = pooled.mean(axis=0)
        pooled_var = pooled.var(axis=0)
        np.testing.assert_allclose(pooled_mean, reference.mean, atol=0.1)
        np.testing.assert_allclose(pooled_var, np.diag(reference.dense_cov), rtol=0.35)

        # Independent chains actually mixed to the same distribution (MP-I8's own diagnostics, reused
        # rather than reimplemented): R-hat/ESS pass on the per-chain stacked draws.
        stacked = chains_from_posterior_samples(result.posteriors())
        diagnostics = evaluate_chain_convergence(stacked, parameter_names=("cell_0", "cell_1"))
        self.assertTrue(diagnostics.converged, diagnostics.detail)

    def test_interrupted_multi_chain_run_resumes_without_duplicating_or_changing_target(self):
        n_chains, n_samples, burn_in, thin = 3, 400, 100, 2
        seed = 4242
        total_steps = _total_steps(n_samples, burn_in, thin)

        # Uninterrupted reference: the whole batch in one go.
        reference = run_parallel_chains(
            _analytic_problem,
            n_chains=n_chains,
            n_samples=n_samples,
            burn_in=burn_in,
            thin=thin,
            step_scale=_STEP_SCALE,
            seed=seed,
            checkpoint_dir=self._checkpoint_dir("reference"),
            checkpoint_every=total_steps,  # single segment, i.e. an ordinary uninterrupted run
            max_workers=3,
        )
        for outcome in reference.outcomes:
            self.assertEqual(outcome.status, "completed")

        # Manufacture a genuinely interrupted batch: run each chain's *own* scheduler-spawned stream
        # directly (matching mcmc_checkpoint_test.py's own interruption pattern) and stop partway,
        # leaving a real, partial on-disk checkpoint -- exactly what a process crash mid-batch would.
        resume_dir = Path(self._checkpoint_dir("interrupted"))
        grid, observations, registry, prior = _analytic_problem()
        streams = spawn_counter_streams(seed, n_chains)
        from mixle_pde.mcmc_checkpoint import run_checkpointed

        class _StopAfter(Exception):
            pass

        for chain_id, rng in enumerate(streams):
            path = resume_dir / f"chain_{chain_id:04d}"

            def _on_checkpoint(checkpoint: MCMCCheckpoint, path=path) -> None:
                save_checkpoint(checkpoint, path)
                if checkpoint.iteration >= 150:
                    raise _StopAfter()

            try:
                run_checkpointed(
                    grid,
                    observations,
                    registry,
                    prior,
                    n_samples=n_samples,
                    burn_in=burn_in,
                    thin=thin,
                    step_scale=_STEP_SCALE,
                    rng=rng,
                    checkpoint_every=37,  # awkward cadence, matching mcmc_checkpoint_test.py's own case
                    on_checkpoint=_on_checkpoint,
                )
                self.fail(f"expected chain {chain_id} to be interrupted before completion")
            except _StopAfter:
                pass

        # Confirm the manufactured checkpoints are genuinely partial before resuming them.
        from mixle_pde.mcmc_checkpoint import load_checkpoint

        for chain_id in range(n_chains):
            loaded = load_checkpoint(resume_dir / f"chain_{chain_id:04d}")
            self.assertGreater(loaded.iteration, 0)
            self.assertLess(loaded.iteration, loaded.total_steps)

        resumed = resume_parallel_chains(
            _analytic_problem,
            n_chains=n_chains,
            checkpoint_dir=str(resume_dir),
            checkpoint_every=41,  # a different cadence than either the reference or the interrupted run used
            max_workers=3,
        )

        self.assertEqual(resumed.status_counts(), {"completed": n_chains, "failed": 0, "cancelled": 0})
        for chain_id in range(n_chains):
            ref_outcome = reference.outcomes[chain_id]
            res_outcome = resumed.outcomes[chain_id]
            # Exact stored-sample count: no duplication, nothing missing.
            self.assertEqual(res_outcome.posterior.samples.shape[0], n_samples)
            # Bit-for-bit identical to the uninterrupted reference: resuming did not change the target
            # distribution or re-derive a different (even if statistically similar) chain.
            np.testing.assert_array_equal(res_outcome.posterior.samples, ref_outcome.posterior.samples)
            np.testing.assert_array_equal(res_outcome.posterior.log_posterior, ref_outcome.posterior.log_posterior)
            np.testing.assert_array_equal(res_outcome.posterior.map, ref_outcome.posterior.map)
            self.assertEqual(res_outcome.report.accepted, ref_outcome.report.accepted)
            self.assertEqual(res_outcome.report.stored_samples, ref_outcome.report.stored_samples)

    def test_resuming_an_already_complete_batch_dispatches_no_worker_and_adds_nothing(self):
        n_chains = 2
        result = run_parallel_chains(
            _analytic_problem,
            n_chains=n_chains,
            n_samples=60,
            burn_in=20,
            thin=1,
            step_scale=_STEP_SCALE,
            seed=7,
            checkpoint_dir=self._checkpoint_dir("already_complete"),
            max_workers=2,
        )
        self.assertEqual(result.status_counts()["completed"], n_chains)

        resumed = resume_parallel_chains(
            _analytic_problem,
            n_chains=n_chains,
            checkpoint_dir=self._checkpoint_dir("already_complete"),
            max_workers=2,
        )
        for chain_id in range(n_chains):
            outcome = resumed.outcomes[chain_id]
            self.assertEqual(outcome.status, "completed")
            self.assertEqual(outcome.attempts, 0)
            self.assertEqual(outcome.worker_pids, ())
            np.testing.assert_array_equal(outcome.posterior.samples, result.outcomes[chain_id].posterior.samples)

    def test_resume_requires_every_chain_to_have_a_checkpoint(self):
        empty_dir = self._checkpoint_dir("empty")
        with self.assertRaises(FileNotFoundError):
            resume_parallel_chains(_analytic_problem, n_chains=2, checkpoint_dir=empty_dir)

    def test_transient_failure_is_retried_and_the_chain_still_completes(self):
        marker_dir = str(Path(self._tmp) / "markers_transient")
        factory = _CountedFailureFactory(marker_dir, fail_first_n_calls=1)

        result = run_parallel_chains(
            factory,
            n_chains=1,
            n_samples=100,
            burn_in=20,
            thin=1,
            step_scale=_STEP_SCALE,
            seed=11,
            checkpoint_dir=self._checkpoint_dir("transient"),
            max_workers=1,
            retry_limit=1,
        )
        outcome = result.outcomes[0]
        self.assertEqual(outcome.status, "completed")
        self.assertEqual(outcome.attempts, 2)  # one failed attempt, one successful retry
        self.assertEqual(outcome.posterior.samples.shape, (100, 2))

    def test_failure_exhausting_retries_is_reported_and_does_not_crash_the_batch(self):
        marker_dir = str(Path(self._tmp) / "markers_terminal")
        # fail_first_n_calls larger than retry_limit + 1 attempts guarantees every attempt fails.
        factory = _CountedFailureFactory(marker_dir, fail_first_n_calls=100)

        result = run_parallel_chains(
            factory,
            n_chains=2,
            n_samples=80,
            burn_in=20,
            thin=1,
            step_scale=_STEP_SCALE,
            seed=13,
            checkpoint_dir=self._checkpoint_dir("terminal"),
            max_workers=1,
            retry_limit=1,
        )
        self.assertEqual(result.status_counts(), {"completed": 0, "failed": 2, "cancelled": 0})
        for outcome in result.outcomes:
            self.assertEqual(outcome.status, "failed")
            self.assertIsNone(outcome.posterior)
            self.assertEqual(outcome.attempts, 2)  # initial attempt + 1 retry, both failed
            self.assertIn("synthetic transient failure", outcome.detail)

    def test_wall_clock_budget_cancels_chains_not_yet_dispatched(self):
        result = run_parallel_chains(
            _analytic_problem,
            n_chains=5,
            n_samples=50,
            burn_in=10,
            thin=1,
            step_scale=_STEP_SCALE,
            seed=17,
            checkpoint_dir=self._checkpoint_dir("budget"),
            max_workers=1,  # serialize so the budget reliably bites before every chain starts
            wall_clock_budget_seconds=1e-9,
        )
        counts = result.status_counts()
        self.assertEqual(counts["completed"] + counts["failed"] + counts["cancelled"], 5)
        self.assertGreater(counts["cancelled"], 0, "an effectively-zero budget should cancel at least one chain")
        for outcome in result.outcomes:
            if outcome.status == "cancelled":
                self.assertIsNone(outcome.posterior)
                self.assertIn("budget", outcome.detail)

    def test_rejects_fewer_than_one_chain(self):
        with self.assertRaises(ValueError):
            run_parallel_chains(
                _analytic_problem,
                n_chains=0,
                n_samples=10,
                burn_in=1,
                thin=1,
                step_scale=1.0,
                seed=1,
                checkpoint_dir=self._checkpoint_dir("invalid"),
            )

    def test_custom_resource_budget_is_accepted(self):
        result = run_parallel_chains(
            _analytic_problem,
            n_chains=1,
            n_samples=30,
            burn_in=10,
            thin=1,
            step_scale=_STEP_SCALE,
            seed=3,
            checkpoint_dir=self._checkpoint_dir("custom_budget"),
            max_workers=1,
            chain_budget=ResourceBudget(cpu_cores=2.0, memory_mb=1024.0, wall_clock_seconds=60.0),
        )
        self.assertEqual(result.status_counts()["completed"], 1)


class ChainOutcomeValidationTest(unittest.TestCase):
    def test_completed_outcome_requires_posterior_report_and_checkpoint(self):
        with self.assertRaises(ValueError):
            ChainOutcome(
                chain_id=0,
                status="completed",
                posterior=None,
                report=None,
                checkpoint=None,
                attempts=1,
                worker_pids=(123,),
                wall_clock_seconds=0.1,
                detail="",
            )

    def test_rejects_unknown_status(self):
        with self.assertRaises(ValueError):
            ChainOutcome(
                chain_id=0,
                status="running",
                posterior=None,
                report=None,
                checkpoint=None,
                attempts=1,
                worker_pids=(),
                wall_clock_seconds=0.0,
                detail="",
            )


class ParallelChainRunValidationTest(unittest.TestCase):
    def _cancelled_outcome(self, chain_id: int) -> ChainOutcome:
        return ChainOutcome(
            chain_id=chain_id,
            status="cancelled",
            posterior=None,
            report=None,
            checkpoint=None,
            attempts=0,
            worker_pids=(),
            wall_clock_seconds=0.0,
            detail="test fixture",
        )

    def test_rejects_length_mismatch(self):
        with self.assertRaises(ValueError):
            ParallelChainRun(
                n_chains=2,
                outcomes=(self._cancelled_outcome(0),),
                seed=1,
                checkpoint_dir="/tmp/does-not-matter",
                wall_clock_seconds=0.0,
            )

    def test_rejects_out_of_order_chain_ids(self):
        with self.assertRaises(ValueError):
            ParallelChainRun(
                n_chains=2,
                outcomes=(self._cancelled_outcome(1), self._cancelled_outcome(0)),
                seed=1,
                checkpoint_dir="/tmp/does-not-matter",
                wall_clock_seconds=0.0,
            )

    def test_pooled_samples_requires_a_completed_chain(self):
        run = ParallelChainRun(
            n_chains=1,
            outcomes=(self._cancelled_outcome(0),),
            seed=1,
            checkpoint_dir="/tmp/does-not-matter",
            wall_clock_seconds=0.0,
        )
        with self.assertRaises(ValueError):
            run.pooled_samples()


if __name__ == "__main__":
    unittest.main()
