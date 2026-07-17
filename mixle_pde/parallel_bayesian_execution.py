"""Parallel, resumable, reproducible multi-chain Bayesian execution (MP-I9 baseline).

``docs/reconciliation/mp-task-ledger.md`` records MP-I9 ("Scalable, restartable, reproducible
Bayesian execution") as ``not-started``, citing :mod:`mixle_pde.verification.capability_inventory`'s
own methodology note: "a repo-wide sweep found no ``mpi4py``, no ``multiprocessing``, and no
``concurrent.futures``/``joblib`` usage anywhere under ``mixle_pde/``." :mod:`mixle_pde.mcmc_checkpoint`
(MP-I9's only prior commit, PR #99) closes a narrower slice -- checkpoint/restart for a single chain in
a single process -- and says so explicitly in its own docstring: "It does not implement MPI/
multiprocessing/distributed execution, counter-based (splittable) random streams, or checkpointing for
any sampler other than metropolis_field_invert." Re-checked fresh against ``origin/release/0.8.0``
before starting: every entry in ``CAPABILITY_INVENTORY`` still reports ``parallel_status ==
"single_process"``, confirming the gap is real, not stale.

This module closes the concrete, buildable slice of MP-I9's full bullet list: it actually schedules
multiple independent Markov chains **across OS processes** (via the standard-library
:class:`concurrent.futures.ProcessPoolExecutor` -- the literal thing the capability-inventory sweep
found completely absent), seeds each chain from an independent **counter-based** random stream
(NumPy's ``Philox`` bit generator, keyed by a spawned :class:`numpy.random.SeedSequence` per chain --
the standard, documented mechanism for parallel-safe reproducible streams), persists **chunked**
per-chain posterior storage after every segment (by reusing :mod:`mixle_pde.mcmc_checkpoint` --
:func:`run_checkpointed`/:func:`resume_checkpointed` are called, never reimplemented), applies
**failure-aware retries** and **budget/cancellation** policy via :mod:`mixle_pde.job_governance`'s
already-merged, real executor-facing bookkeeping (:class:`~mixle_pde.job_governance.JobRegistry`,
:class:`~mixle_pde.job_governance.ResourceBudget`), and **deterministically reduces** every chain's
outcome by ``chain_id`` -- never by completion order, which is inherently nondeterministic across
processes.

What this deliberately does NOT claim (see Limitations below and the capability-inventory entry for
this module): particle/SMC/ensemble scheduling, adjoint-checkpoint scheduling, GPU/device scheduling,
and true multi-node (MPI) execution are all out of scope. This is a genuine, verified first slice of an
8-person-week work-plan item, matching the "baseline"/"(partial)" precedent already established by
sibling MP-* PRs in this repo (e.g. MP-N5 baseline, MP-C8's GCL-check half, MP-N6's drift-monitoring
piece) rather than a claim to have finished the whole item.

Design
------
:func:`spawn_counter_streams` is the counter-based-stream primitive: one root
:class:`numpy.random.SeedSequence` is spawned into ``n`` children (NumPy's own documented
parallel-stream recipe), each wrapped in an independent ``Philox`` bit generator. Philox is a genuine
counter-based RNG -- unlike PCG64/MT19937's mutable-state design, its state is a fixed key plus a
counter, so streams from different spawn keys are guaranteed statistically independent and
non-overlapping regardless of how many draws each stream consumes, and are reproducible from the key
alone. :func:`run_parallel_chains` uses this same spawning routine internally, so a caller can
reproduce the exact per-chain stream a scheduled run used just by calling
``spawn_counter_streams(seed, n_chains)`` itself.

Each chain's actual sampling is exactly one call to :func:`mixle_pde.mcmc_checkpoint.run_checkpointed`
(fresh) or :func:`~mixle_pde.mcmc_checkpoint.resume_checkpointed` (continuing from a prior checkpoint),
dispatched to a worker process via ``ProcessPoolExecutor``. Because ``ProcessPoolExecutor`` pickles
its arguments, and :class:`mixle_pde.observations.ForwardOperatorRegistry` holds closures (a registered
:class:`~mixle_pde.observations.ForwardOperator`'s ``predict``/``jacobian`` are typically returned from
a local factory function, e.g. :func:`mixle_pde.observations.borehole_forward_operator`) that plain
``pickle`` cannot serialize, callers supply a **problem factory** -- a module-level, picklable
zero-argument callable that rebuilds ``(grid, observations, registry, prior)`` -- rather than the
already-built objects. Each worker calls it once, locally, in its own process. This is standard
practice for multiprocessing scientific code (reconstruct unpicklable resources in the worker; never
try to ship live closures across the process boundary) and is verified directly in the test suite
(:func:`pickle.dumps` on a registry with a registered closure-based operator raises ``PicklingError``;
the factory pattern sidesteps it entirely).

The scheduler (:func:`run_parallel_chains`/:func:`resume_parallel_chains`, sharing one internal loop)
tracks every chain through a :class:`~mixle_pde.job_governance.JobRegistry`: ``submit`` before
dispatch, ``heartbeat`` on dispatch/redispatch, ``complete``/``fail`` when a future resolves.
A worker exception is caught by ``concurrent.futures`` itself and surfaces via ``Future.exception()``;
this module never lets one chain's exception crash the batch. ``JobRegistry.fail`` applies the
already-implemented bounded-retry policy: while retries remain, the chain is resubmitted --  from its
own last on-disk checkpoint if one was ever written for it, or from scratch with its *original* seed
sequence if it failed before its first checkpoint (both cases reproduce the identical intended chain,
never silently mutating the target distribution or duplicating stored samples, because
``resume_checkpointed`` only ever appends draws for raw steps beyond ``checkpoint.iteration``). Once
retries are exhausted the chain is terminally ``failed`` and excluded from the pooled posterior, but
still reported with a full, attributable outcome record. A wall-clock budget is checked before every
(re)dispatch; a chain not yet dispatched when the budget is exceeded is marked ``cancelled`` rather
than silently dropped or forced through -- already-running chains are allowed to finish their current
segment (cooperative, not preemptive; see Limitations).

Every :class:`ChainOutcome` records ``chain_id``, every worker PID that ever ran an attempt of it, the
attempt count, and wall-clock time -- attributable at the chain/segment granularity this module
actually schedules. :func:`ParallelChainRun.posteriors` and the ``chain_id``-sorted
:attr:`ParallelChainRun.outcomes` tuple are the deterministic reduction: identical regardless of which
process happened to finish first. The result feeds directly into
:mod:`mixle_pde.verification.mcmc_diagnostics` (already-merged, MP-I8) via
:func:`~mixle_pde.verification.mcmc_diagnostics.chains_from_posterior_samples`.

Limitations
-----------
* Multi-chain MCMC (:func:`~mixle_pde.field_mcmc.metropolis_field_invert`, via
  :mod:`mixle_pde.mcmc_checkpoint`) only. Particle filters / SMC / ensemble Kalman
  (:mod:`mixle_pde.field_assimilation`) and adjoint-checkpoint scheduling are not scheduled or made
  resumable here -- a genuinely separate, unclaimed remainder of MP-I9's full scope.
* Multi-process (single machine), not multi-node/MPI. ``max_workers`` bounds OS processes on the
  calling machine; there is no distributed/cluster dispatch and no GPU/device scheduling.
* Cancellation is cooperative and pre-dispatch only: a chain whose worker process is already running
  is never killed when the wall-clock budget is exceeded; only not-yet-dispatched chains are stopped
  from starting. A true preemptive cancellation would need to interrupt ``metropolis_field_invert``
  mid-call, which this module (matching ``mcmc_checkpoint.py``'s own boundary) treats as a black box.
* Attribution is chain/segment-level (which worker process ran which chain, for how long, across how
  many attempts), not per-individual-proposal. Instrumenting individual Metropolis proposal/accept
  calls would require modifying ``metropolis_field_invert`` itself, which this module never does.
* ``JobRegistry`` is single-threaded bookkeeping by its own design; this module's scheduler loop is
  the one thread driving it, consistent with that module's documented usage contract.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from mixle_pde.field_inversion import FieldGaussianPrior
from mixle_pde.field_mcmc import MCMCReport
from mixle_pde.job_governance import FailureReason, JobRegistry, JobStatus, ResourceBudget
from mixle_pde.latent import Field3D, PosteriorFieldSamples3D
from mixle_pde.mcmc_checkpoint import MCMCCheckpoint, load_checkpoint, resume_checkpointed, run_checkpointed
from mixle_pde.observations import ForwardOperatorRegistry, Observation

__all__ = [
    "ChainOutcome",
    "ParallelChainRun",
    "ProblemFactory",
    "spawn_counter_streams",
    "run_parallel_chains",
    "resume_parallel_chains",
]

ProblemFactory = Callable[[], "tuple[Field3D, list[Observation], ForwardOperatorRegistry, FieldGaussianPrior]"]

_TERMINAL_STATUSES = ("completed", "failed", "cancelled")


def _spawn_seed_sequences(seed: int | np.random.SeedSequence, n_streams: int) -> tuple[np.random.SeedSequence, ...]:
    if n_streams < 1:
        raise ValueError("n_streams must be >= 1.")
    root = seed if isinstance(seed, np.random.SeedSequence) else np.random.SeedSequence(seed)
    return tuple(root.spawn(n_streams))


def spawn_counter_streams(seed: int | np.random.SeedSequence, n_streams: int) -> tuple[np.random.Generator, ...]:
    """``n_streams`` independent, reproducible counter-based (``Philox``) RNG streams from one root seed.

    Uses NumPy's own documented parallel-stream recipe: a root :class:`~numpy.random.SeedSequence` is
    spawned into ``n_streams`` children (a deterministic function of ``seed`` and the spawn index
    alone), each wrapped in its own ``Philox`` bit generator. :func:`run_parallel_chains` calls the same
    underlying spawn routine for its per-chain streams, so a caller can reproduce (or independently
    pre-seed, e.g. to manufacture a test checkpoint) exactly the stream chain ``i`` of a scheduled run
    used by calling ``spawn_counter_streams(seed, n_chains)[i]``.
    """
    return tuple(np.random.Generator(np.random.Philox(child)) for child in _spawn_seed_sequences(seed, n_streams))


@dataclass(frozen=True)
class _ChainJob:
    """Picklable per-attempt dispatch instruction for one chain, sent to a worker process."""

    chain_id: int
    checkpoint_path: str
    checkpoint_every: int
    mode: str  # "fresh" or "resume"
    seed_sequence: np.random.SeedSequence | None  # set iff mode == "fresh"
    n_samples: int | None = None
    burn_in: int | None = None
    thin: int | None = None
    step_scale: Any = None

    def __post_init__(self) -> None:
        if self.mode not in ("fresh", "resume"):
            raise ValueError(f"mode must be 'fresh' or 'resume', got {self.mode!r}.")
        if self.mode == "fresh" and self.seed_sequence is None:
            raise ValueError("a 'fresh' job needs a seed_sequence.")


@dataclass(frozen=True)
class _WorkerChainResult:
    """Picklable payload a worker process returns for one completed chain attempt."""

    chain_id: int
    posterior: PosteriorFieldSamples3D
    report: MCMCReport
    checkpoint: MCMCCheckpoint
    worker_pid: int
    wall_clock_seconds: float


def _worker_run_chain(problem_factory: ProblemFactory, job: _ChainJob) -> _WorkerChainResult:
    """Runs in a worker process: rebuild the problem locally, then run or resume one chain.

    ``metropolis_field_invert`` is never called directly here -- :func:`~mixle_pde.mcmc_checkpoint.
    run_checkpointed`/:func:`~mixle_pde.mcmc_checkpoint.resume_checkpointed` are, exactly as a
    single-process caller would use them, so this module inherits their already-verified bit-for-bit
    segment-cadence-independence and resume guarantees rather than re-deriving them.
    """
    start = time.monotonic()
    grid, observations, registry, prior = problem_factory()
    path = Path(job.checkpoint_path)

    def _persist(checkpoint: MCMCCheckpoint) -> None:
        from mixle_pde.mcmc_checkpoint import save_checkpoint

        save_checkpoint(checkpoint, path)

    if job.mode == "fresh":
        assert job.seed_sequence is not None
        rng = np.random.Generator(np.random.Philox(job.seed_sequence))
        posterior, report, checkpoint = run_checkpointed(
            grid,
            observations,
            registry,
            prior,
            n_samples=job.n_samples,
            burn_in=job.burn_in,
            thin=job.thin,
            step_scale=job.step_scale,
            rng=rng,
            checkpoint_every=job.checkpoint_every,
            on_checkpoint=_persist,
        )
    else:
        loaded = load_checkpoint(path)
        posterior, report, checkpoint = resume_checkpointed(
            loaded,
            grid,
            observations,
            registry,
            prior,
            checkpoint_every=job.checkpoint_every,
            on_checkpoint=_persist,
        )

    return _WorkerChainResult(
        chain_id=job.chain_id,
        posterior=posterior,
        report=report,
        checkpoint=checkpoint,
        worker_pid=os.getpid(),
        wall_clock_seconds=time.monotonic() - start,
    )


@dataclass(frozen=True)
class ChainOutcome:
    """The final, attributable outcome of one scheduled chain.

    ``status`` is one of ``"completed"``, ``"failed"``, or ``"cancelled"``. Only a ``"completed"``
    chain has a non-``None`` ``posterior``/``report``/``checkpoint`` -- a failed or cancelled chain
    is still reported (never silently dropped), with ``posterior=None`` making it impossible to
    accidentally fold a non-result into a posterior summary.
    """

    chain_id: int
    status: str
    posterior: PosteriorFieldSamples3D | None
    report: MCMCReport | None
    checkpoint: MCMCCheckpoint | None
    attempts: int
    worker_pids: tuple[int, ...]
    wall_clock_seconds: float
    detail: str

    def __post_init__(self) -> None:
        if self.status not in _TERMINAL_STATUSES:
            raise ValueError(f"status must be one of {_TERMINAL_STATUSES}, got {self.status!r}.")
        if self.status == "completed" and (self.posterior is None or self.report is None or self.checkpoint is None):
            raise ValueError("a completed ChainOutcome must carry posterior, report, and checkpoint.")


@dataclass(frozen=True)
class ParallelChainRun:
    """The deterministic, chain_id-ordered result of one scheduled multi-chain run.

    ``outcomes`` always has length ``n_chains`` and is indexed by ``chain_id`` (``outcomes[i].chain_id
    == i``), regardless of which process happened to finish first -- the deterministic-reduction
    guarantee MP-I9's accept criterion asks for.
    """

    n_chains: int
    outcomes: tuple[ChainOutcome, ...]
    seed: int | None
    checkpoint_dir: str
    wall_clock_seconds: float

    def __post_init__(self) -> None:
        if len(self.outcomes) != self.n_chains:
            raise ValueError(f"expected {self.n_chains} outcomes, got {len(self.outcomes)}.")
        for i, outcome in enumerate(self.outcomes):
            if outcome.chain_id != i:
                raise ValueError(f"outcomes must be ordered by chain_id; outcomes[{i}].chain_id == {outcome.chain_id}.")

    def completed(self) -> tuple[ChainOutcome, ...]:
        return tuple(o for o in self.outcomes if o.status == "completed")

    def posteriors(self) -> tuple[PosteriorFieldSamples3D, ...]:
        """Completed chains' posteriors, in ``chain_id`` order -- feed directly into
        :func:`mixle_pde.verification.mcmc_diagnostics.chains_from_posterior_samples`."""
        return tuple(o.posterior for o in self.completed())  # type: ignore[misc]

    def pooled_samples(self) -> np.ndarray:
        """All completed chains' stored draws stacked into one ``(total_draws, n_parameters)`` array."""
        posteriors = self.posteriors()
        if not posteriors:
            raise ValueError("no completed chains to pool.")
        return np.concatenate([p.samples for p in posteriors], axis=0)

    def status_counts(self) -> dict[str, int]:
        counts = dict.fromkeys(_TERMINAL_STATUSES, 0)
        for outcome in self.outcomes:
            counts[outcome.status] += 1
        return counts


def _checkpoint_path(checkpoint_dir: Path, chain_id: int) -> Path:
    return checkpoint_dir / f"chain_{chain_id:04d}"


def _schedule(
    *,
    problem_factory: ProblemFactory,
    initial_jobs: dict[int, _ChainJob],
    n_chains: int,
    checkpoint_dir: Path,
    checkpoint_every: int,
    max_workers: int | None,
    retry_limit: int,
    wall_clock_budget_seconds: float | None,
    chain_budget: ResourceBudget | None,
    seed: int | None,
    preset_outcomes: dict[int, ChainOutcome] | None = None,
) -> ParallelChainRun:
    start_time = time.monotonic()
    registry = JobRegistry()
    budget = chain_budget or ResourceBudget()

    job_by_chain: dict[int, _ChainJob] = dict(initial_jobs)
    original_seed_sequence: dict[int, np.random.SeedSequence | None] = {
        chain_id: job.seed_sequence for chain_id, job in initial_jobs.items()
    }
    job_id_by_chain: dict[int, str] = {}
    attempts: dict[int, int] = {chain_id: 0 for chain_id in initial_jobs}
    worker_pids: dict[int, list[int]] = {chain_id: [] for chain_id in initial_jobs}
    outcomes: dict[int, ChainOutcome] = dict(preset_outcomes or {})
    not_yet_dispatched: set[int] = set(initial_jobs)

    for chain_id in initial_jobs:
        record = registry.submit(budget=budget, retry_limit=retry_limit, metadata={"chain_id": chain_id})
        job_id_by_chain[chain_id] = record.id

    def _elapsed() -> float:
        return time.monotonic() - start_time

    def _budget_exhausted() -> bool:
        return wall_clock_budget_seconds is not None and _elapsed() > wall_clock_budget_seconds

    def _finalize(chain_id: int, *, status: str, result: _WorkerChainResult | None, detail: str) -> None:
        outcomes[chain_id] = ChainOutcome(
            chain_id=chain_id,
            status=status,
            posterior=result.posterior if result is not None else None,
            report=result.report if result is not None else None,
            checkpoint=result.checkpoint if result is not None else None,
            attempts=attempts[chain_id],
            worker_pids=tuple(worker_pids[chain_id]),
            wall_clock_seconds=result.wall_clock_seconds if result is not None else 0.0,
            detail=detail,
        )

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures: dict[Any, int] = {}

        def _dispatch(chain_id: int) -> None:
            job_id = job_id_by_chain[chain_id]
            if _budget_exhausted():
                registry.cancel(job_id, reason="wall_clock_budget_exceeded before dispatch")
                _finalize(
                    chain_id,
                    status="cancelled",
                    result=None,
                    detail=f"cancelled before dispatch: wall-clock budget "
                    f"({wall_clock_budget_seconds}s) already exceeded at {_elapsed():.3f}s.",
                )
                not_yet_dispatched.discard(chain_id)
                return
            registry.heartbeat(job_id)
            attempts[chain_id] += 1
            future = executor.submit(_worker_run_chain, problem_factory, job_by_chain[chain_id])
            futures[future] = chain_id
            not_yet_dispatched.discard(chain_id)

        for chain_id in list(not_yet_dispatched):
            _dispatch(chain_id)

        while futures:
            done, _pending = wait(list(futures.keys()), return_when=FIRST_COMPLETED)
            for future in done:
                chain_id = futures.pop(future)
                job_id = job_id_by_chain[chain_id]
                exc = future.exception()
                if exc is None:
                    result = future.result()
                    worker_pids[chain_id].append(result.worker_pid)
                    registry.heartbeat(job_id)
                    registry.complete(job_id)
                    _finalize(
                        chain_id,
                        status="completed",
                        result=result,
                        detail=f"chain {chain_id} completed after {attempts[chain_id]} attempt(s).",
                    )
                    continue

                record = registry.fail(job_id, reason=FailureReason.EXECUTOR_ERROR, detail=str(exc))
                if record.status is JobStatus.QUEUED and not _budget_exhausted():
                    path = Path(job_by_chain[chain_id].checkpoint_path)
                    has_checkpoint = path.with_suffix(".json").exists()
                    if has_checkpoint:
                        job_by_chain[chain_id] = _ChainJob(
                            chain_id=chain_id,
                            checkpoint_path=job_by_chain[chain_id].checkpoint_path,
                            checkpoint_every=checkpoint_every,
                            mode="resume",
                            seed_sequence=None,
                        )
                    else:
                        job_by_chain[chain_id] = _ChainJob(
                            chain_id=chain_id,
                            checkpoint_path=job_by_chain[chain_id].checkpoint_path,
                            checkpoint_every=checkpoint_every,
                            mode="fresh",
                            seed_sequence=original_seed_sequence[chain_id],
                            n_samples=job_by_chain[chain_id].n_samples,
                            burn_in=job_by_chain[chain_id].burn_in,
                            thin=job_by_chain[chain_id].thin,
                            step_scale=job_by_chain[chain_id].step_scale,
                        )
                    registry.heartbeat(job_id)
                    attempts[chain_id] += 1
                    future2 = executor.submit(_worker_run_chain, problem_factory, job_by_chain[chain_id])
                    futures[future2] = chain_id
                    continue

                if record.status is JobStatus.QUEUED and _budget_exhausted():
                    registry.cancel(job_id, reason="wall_clock_budget_exceeded before retry dispatch")
                    _finalize(
                        chain_id,
                        status="cancelled",
                        result=None,
                        detail=f"cancelled before retry: wall-clock budget "
                        f"({wall_clock_budget_seconds}s) exceeded at {_elapsed():.3f}s; last error: {exc}",
                    )
                else:
                    _finalize(
                        chain_id,
                        status="failed",
                        result=None,
                        detail=f"chain {chain_id} failed terminally after {attempts[chain_id]} attempt(s) "
                        f"(retry_limit={retry_limit}): {exc}",
                    )

    ordered = tuple(outcomes[i] for i in range(n_chains))
    return ParallelChainRun(
        n_chains=n_chains,
        outcomes=ordered,
        seed=seed,
        checkpoint_dir=str(checkpoint_dir),
        wall_clock_seconds=_elapsed(),
    )


def run_parallel_chains(
    problem_factory: ProblemFactory,
    *,
    n_chains: int,
    n_samples: int,
    burn_in: int = 1000,
    thin: int = 1,
    step_scale: float | np.ndarray = 1.0,
    seed: int,
    checkpoint_dir: str | Path,
    checkpoint_every: int | None = None,
    max_workers: int | None = None,
    retry_limit: int = 1,
    wall_clock_budget_seconds: float | None = None,
    chain_budget: ResourceBudget | None = None,
) -> ParallelChainRun:
    """Schedule ``n_chains`` independent :func:`~mixle_pde.field_mcmc.metropolis_field_invert` chains
    across worker processes, each from its own counter-based (``Philox``) random stream spawned from
    ``seed``.

    ``problem_factory`` must be a module-level (picklable) zero-argument callable returning
    ``(grid, observations, registry, prior)`` -- see the module docstring for why a live registry
    cannot be passed directly. ``checkpoint_dir`` is created if needed; every chain writes (and
    overwrites, each segment) ``{checkpoint_dir}/chain_{chain_id:04d}.npz``/``.json``, so
    :func:`resume_parallel_chains` can later continue an interrupted or cancelled run without
    duplicating any already-stored sample. ``checkpoint_every`` defaults to the full chain length
    (one segment); pass a smaller value to checkpoint (and therefore risk losing at most that many
    steps of progress on a genuine mid-segment crash) more often.

    Returns a :class:`ParallelChainRun` whose ``outcomes`` are always ordered by ``chain_id`` --
    deterministic regardless of which worker process happens to finish first.
    """
    if n_chains < 1:
        raise ValueError("n_chains must be >= 1.")
    total_steps = int(burn_in) + int(n_samples) * int(thin)
    if total_steps < 1:
        raise ValueError("burn_in + n_samples * thin must be >= 1.")
    resolved_checkpoint_every = total_steps if checkpoint_every is None else int(checkpoint_every)

    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    seed_sequences = _spawn_seed_sequences(seed, n_chains)
    jobs = {
        chain_id: _ChainJob(
            chain_id=chain_id,
            checkpoint_path=str(_checkpoint_path(checkpoint_dir, chain_id)),
            checkpoint_every=resolved_checkpoint_every,
            mode="fresh",
            seed_sequence=seed_sequences[chain_id],
            n_samples=int(n_samples),
            burn_in=int(burn_in),
            thin=int(thin),
            step_scale=step_scale,
        )
        for chain_id in range(n_chains)
    }
    return _schedule(
        problem_factory=problem_factory,
        initial_jobs=jobs,
        n_chains=n_chains,
        checkpoint_dir=checkpoint_dir,
        checkpoint_every=resolved_checkpoint_every,
        max_workers=max_workers,
        retry_limit=retry_limit,
        wall_clock_budget_seconds=wall_clock_budget_seconds,
        chain_budget=chain_budget,
        seed=seed,
    )


def resume_parallel_chains(
    problem_factory: ProblemFactory,
    *,
    n_chains: int,
    checkpoint_dir: str | Path,
    checkpoint_every: int | None = None,
    max_workers: int | None = None,
    retry_limit: int = 1,
    wall_clock_budget_seconds: float | None = None,
    chain_budget: ResourceBudget | None = None,
) -> ParallelChainRun:
    """Continue a batch :func:`run_parallel_chains` previously wrote to ``checkpoint_dir`` -- in fresh
    worker processes that may share nothing with whatever ran before beyond the files on disk.

    Every ``chain_id in range(n_chains)`` must have a checkpoint already on disk (even a chain that
    completed in one segment during the original call has one -- ``run_parallel_chains`` always
    persists at least the final state). A chain whose checkpoint shows ``iteration == total_steps`` is
    already complete: it is finalized directly from the checkpoint's own stored state, with **no
    worker process dispatched for it and no sample re-added** -- structurally, not just behaviorally,
    ruling out duplication -- which is exactly the "resume without duplicating samples" guarantee.
    Only chains with ``iteration < total_steps`` are actually redispatched to continue sampling.
    """
    if n_chains < 1:
        raise ValueError("n_chains must be >= 1.")
    checkpoint_dir = Path(checkpoint_dir)

    jobs: dict[int, _ChainJob] = {}
    preset_outcomes: dict[int, ChainOutcome] = {}
    total_steps_seen: set[int] = set()
    problem: tuple[Field3D, list[Observation], ForwardOperatorRegistry, FieldGaussianPrior] | None = None

    for chain_id in range(n_chains):
        path = _checkpoint_path(checkpoint_dir, chain_id)
        if not path.with_suffix(".json").exists():
            raise FileNotFoundError(
                f"no checkpoint found for chain {chain_id} under {checkpoint_dir}; resume_parallel_chains "
                "requires every chain id in range(n_chains) to have been submitted at least once by "
                "run_parallel_chains (a chain cancelled before its worker ever started has no checkpoint -- "
                "re-run the whole batch, or a fresh single chain, instead)."
            )
        loaded = load_checkpoint(path)
        total_steps_seen.add(loaded.total_steps)

        if loaded.iteration >= loaded.total_steps:
            if problem is None:
                problem = problem_factory()
            grid, observations, registry, prior = problem
            posterior, report, checkpoint = resume_checkpointed(loaded, grid, observations, registry, prior)
            preset_outcomes[chain_id] = ChainOutcome(
                chain_id=chain_id,
                status="completed",
                posterior=posterior,
                report=report,
                checkpoint=checkpoint,
                attempts=0,
                worker_pids=(),
                wall_clock_seconds=0.0,
                detail=f"chain {chain_id} was already complete on disk; no worker dispatched.",
            )
            continue

        resolved_every = int(checkpoint_every) if checkpoint_every is not None else max(loaded.total_steps, 1)
        jobs[chain_id] = _ChainJob(
            chain_id=chain_id,
            checkpoint_path=str(path),
            checkpoint_every=resolved_every,
            mode="resume",
            seed_sequence=None,
        )

    resolved_checkpoint_every = (
        int(checkpoint_every) if checkpoint_every is not None else (max(total_steps_seen) if total_steps_seen else 1)
    )
    return _schedule(
        problem_factory=problem_factory,
        initial_jobs=jobs,
        n_chains=n_chains,
        checkpoint_dir=checkpoint_dir,
        checkpoint_every=resolved_checkpoint_every,
        max_workers=max_workers,
        retry_limit=retry_limit,
        wall_clock_budget_seconds=wall_clock_budget_seconds,
        chain_budget=chain_budget,
        seed=None,
        preset_outcomes=preset_outcomes,
    )
