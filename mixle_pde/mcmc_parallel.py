"""Embarrassingly-parallel multi-chain execution for :mod:`mixle_pde.field_mcmc` samplers (MP-I9).

``docs/reconciliation/mp-task-ledger.md`` records ``MP-I9`` ("Scalable, restartable, reproducible
Bayesian execution") as not-started, citing :mod:`mixle_pde.verification.capability_inventory`'s own
methodology note: *"parallel_status is 'single_process' for every entry: a repo-wide sweep found no
mpi4py, no multiprocessing, and no concurrent.futures/joblib usage anywhere under mixle_pde/"* -- an
honest, self-declared gap. :mod:`mixle_pde.mcmc_checkpoint` (``MP-I9`` PR #99) closed the
checkpoint/restart slice of that gap and explicitly left the rest: *"It does not implement
MPI/multiprocessing/distributed execution... for any sampler other than metropolis_field_invert."*
:mod:`mixle_pde.verification.mcmc_diagnostics` names the same remaining slice from the diagnostics
side: *"No resumability, checkpointing, counter-based random streams, or distributed/multi-process
execution is implemented (the broader MP-I9 scope)."* This module closes that slice: real parallel
execution of independent chains, gated as an opt-in capability, not a change to any existing sampler's
default (single-process) behavior.

Why parallel *chains*, not a distributed *solve*
-------------------------------------------------
``docs/adr/0001-mp-a4-backend-selection.md`` (the ``MP-A4`` backend-selection ADR) already surveyed
this exact question for MPI and answered it directly. Its capability-gap section: *"the one class of
workload this repo actually has that could use parallelism today -- marginal_std_cg's Hutchinson
probes... and multi-chain MCMC (mixle_pde/field_mcmc.py) -- is same-machine, embarrassingly parallel,
and does not need MPI's distributed-memory model at all."* Its decision: *"Defer standalone,
hand-rolled mpi4py use inside mixle-pde -- route embarrassingly-parallel ensemble/multi-chain
workloads to mixle-mlops or stdlib concurrent.futures first."* This module is that routing, for the
multi-chain case specifically: :mod:`mixle_pde.field_mcmc` already ships four independent single-chain
samplers (Random-Walk Metropolis, pCN, MALA, HMC); running ``N`` of them from different seeds/starting
points across worker processes is an outer loop around each sampler, not a rewrite of any solver
internals -- no chain ever needs to see another chain's state.

Why ``joblib``, not bare stdlib ``multiprocessing``/``concurrent.futures``
---------------------------------------------------------------------------
The ADR's stated preference is "zero extra dependency" stdlib. That was tried first here and found to
not actually work for this repo's real forward operators, not merely deferred on packaging taste.
:class:`~mixle_pde.observations.ForwardOperatorRegistry` holds :class:`~mixle_pde.observations.ForwardOperator`
instances built by factories such as :func:`~mixle_pde.observations.gravity_forward_operator` and
:func:`~mixle_pde.observations.borehole_forward_operator` -- every one of them returns a *closure*
(``predict``/``jacobian`` defined inside the factory function), not a plain top-level function.
Confirmed directly (see this module's tests): the standard library's ``pickle`` -- and therefore both
``multiprocessing.Pool.map`` and ``concurrent.futures.ProcessPoolExecutor.submit`` under macOS/Windows'
default ``spawn`` start method -- cannot serialize such a closure at all (``PicklingError: Can't pickle
local object ...``); this fails identically even under an explicitly requested ``fork`` start method
for any argument passed *per task* through a ``Pool``'s task queue (only values baked into a worker
*before* the fork, e.g. ``Pool(initializer=..., initargs=...)``, ride along for free -- fork copies
memory, it does not remove the need to pickle values sent to an already-running worker afterward).
``joblib``'s default ``loky`` backend uses ``cloudpickle`` to serialize the submitted call, which
serializes closures transparently and works under ``spawn`` -- i.e. cross-platform, not fork-only.
``joblib`` cannot be lower-cost than the stdlib in the abstract, but it is already an indirect
dependency the moment ``scikit-learn`` is (this repo's own ``inverse``/``surrogate``/``all`` extras
already pull ``scikit-learn`` in, and ``scikit-learn`` requires ``joblib``); this module formalizes it
as its own optional ``parallel`` extra (lazily imported inside :func:`run_parallel_chains`, never at
module scope, matching every other accelerator in this package) rather than silently piggybacking on a
transitive pin that could change out from under it.

What this does *not* do
------------------------
* Does not implement ``mpi4py`` in any form -- the ADR's MPI verdict (defer standalone use; the ``mpi``
  extra remains reserved for ``petsc4py``'s transitive dependency, on a separate trigger) is unchanged
  by this module.
* Does not modify :mod:`mixle_pde.field_mcmc` or :mod:`mixle_pde.mcmc_checkpoint` -- every sampler is
  called exactly as a single-process caller would call it; this is purely an outer dispatch loop.
* Does not compute convergence diagnostics itself. :attr:`MultiChainResult.posteriors` is exactly the
  ``Sequence[PosteriorFieldSamples3D]`` shape
  :func:`mixle_pde.verification.mcmc_diagnostics.chains_from_posterior_samples` expects, so a caller
  gets split-R-hat/ESS by composing the two modules (see this module's tests for the composition), not
  by this module reaching into ``mixle_pde.verification`` itself -- no existing top-level
  ``mixle_pde`` module imports from ``mixle_pde.verification``, and this one does not become the first.
* Does not eliminate -- and cannot amortize away on the very first call -- the cold-start cost of a
  freshly spawned worker process. Measured directly on this repo's own import graph: ``import
  mixle_pde`` alone costs on the order of 20-40 seconds in a fresh interpreter (``python -X importtime
  -c "import mixle_pde"`` attributes most of it to transitively importing ``torch`` and deep
  ``scipy.stats``/``scipy.sparse``/``scipy.special`` submodules the package's own ``__init__.py``
  pulls in eagerly) -- a pre-existing characteristic of this package's import graph, unrelated to
  ``joblib`` or to this module, that every freshly ``spawn``-ed worker pays once. ``joblib``'s ``loky``
  workers persist and are reused across repeated calls to :func:`run_parallel_chains` within one
  process, so this cost is a one-time tax per worker pool, not a per-call or per-chain one; a
  short-lived script that calls this function exactly once will not see a speedup, and this module does
  not pretend otherwise (see this module's own performance test, which warms the pool before timing,
  exactly as any long-lived analysis session would).
"""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from mixle_pde.field_inversion import FieldGaussianPrior
from mixle_pde.field_mcmc import MCMCReport
from mixle_pde.latent import Field3D, PosteriorFieldSamples3D
from mixle_pde.observations import ForwardOperatorRegistry, Observation

__all__ = [
    "ChainResult",
    "MultiChainResult",
    "run_parallel_chains",
]

#: The signature every sampler in :mod:`mixle_pde.field_mcmc` shares:
#: ``sampler(grid, observations, registry, prior, *, initial_unconstrained=..., rng=..., **kwargs)``.
Sampler = Callable[..., tuple[PosteriorFieldSamples3D, MCMCReport]]


def _require_joblib():
    """Import ``joblib`` or raise a clear, actionable error naming the extra to install.

    Matches the ``_require``/``_require_joblib``-style guarded-import convention already used for
    other optional accelerators in this package (e.g. :func:`mixle_pde.env_data._require`,
    :mod:`mixle_pde.reductions`); never imported at module scope, so a zero-extras install still
    imports :mod:`mixle_pde.mcmc_parallel` cleanly and only calling :func:`run_parallel_chains`
    itself requires the dependency.
    """
    try:
        import joblib
    except ImportError as exc:
        raise ImportError(
            "run_parallel_chains needs the optional dependency 'joblib' to dispatch chains across "
            "worker processes. Install it with: pip install 'mixle-pde[parallel]'  (or: pip install joblib)."
        ) from exc
    return joblib


def _spawn_seeds(seed: int | np.random.SeedSequence | None, n_chains: int) -> list[int]:
    """``n_chains`` statistically-independent integer seeds derived from one root seed.

    Uses :class:`numpy.random.SeedSequence`'s ``spawn`` mechanism -- NumPy's own recommended
    construction for independent parallel streams -- rather than a naive ``seed + i``, so adjacent
    chains are never at risk of correlated streams. Each spawned child is reduced to one ``uint32``
    via ``generate_state`` so the result is a plain, printable, picklable ``int`` (matching every
    other seed in this package's public surface, e.g. ``rng=np.random.default_rng(1234)``), not a
    ``SeedSequence`` object threaded through the public API.
    """
    root = seed if isinstance(seed, np.random.SeedSequence) else np.random.SeedSequence(seed)
    return [int(child.generate_state(1)[0]) for child in root.spawn(n_chains)]


def _run_one_chain(
    sampler: Sampler,
    grid: Field3D,
    observations: list[Observation],
    registry: ForwardOperatorRegistry,
    prior: FieldGaussianPrior,
    seed: int,
    initial_unconstrained: np.ndarray | None,
    sampler_kwargs: dict[str, Any],
) -> tuple[PosteriorFieldSamples3D, MCMCReport]:
    """The per-chain unit of work dispatched to each worker via ``joblib.delayed``.

    A top-level (picklable-by-reference) function, called identically whether ``joblib`` runs it
    in-process (``n_jobs=1``) or in a worker process (``n_jobs>1``): builds a fresh
    ``np.random.Generator`` from ``seed`` inside whichever process actually executes it (a live
    ``Generator`` is never itself sent across the process boundary) and calls ``sampler`` exactly as
    a single-process caller would. Stamps ``posterior.provenance["worker_pid"]`` with
    ``os.getpid()`` so a caller can directly confirm more than one real OS process executed a batch
    (see this module's tests), not merely that ``joblib`` reported success.
    """
    rng = np.random.default_rng(seed)
    posterior, report = sampler(
        grid,
        observations,
        registry,
        prior,
        initial_unconstrained=initial_unconstrained,
        rng=rng,
        **sampler_kwargs,
    )
    posterior.provenance["worker_pid"] = os.getpid()
    return posterior, report


@dataclass(frozen=True)
class ChainResult:
    """One independent chain's output, plus the seed that reproduces it standalone.

    ``seed`` alone (fed to ``np.random.default_rng(seed)``) exactly reproduces this chain outside
    :func:`run_parallel_chains` -- e.g. to re-run only the one chain that looked suspicious.
    """

    chain_id: int
    seed: int
    posterior: PosteriorFieldSamples3D
    report: MCMCReport


@dataclass(frozen=True)
class MultiChainResult:
    """``n_chains`` independent chains, plus their pooled (unconstrained-space) draws.

    ``combined`` concatenates every chain's stored draws into one :class:`PosteriorFieldSamples3D` --
    a plain pooling, not a convergence verdict. Pooling chains that have not actually mixed to the
    same distribution silently produces an overconfident posterior; check convergence first (e.g.
    ``mixle_pde.verification.mcmc_diagnostics.evaluate_chain_convergence`` over
    :attr:`posteriors`, the exact shape
    ``mixle_pde.verification.mcmc_diagnostics.chains_from_posterior_samples`` expects) before
    trusting ``combined`` for anything beyond a quick look.
    """

    chains: tuple[ChainResult, ...]
    combined: PosteriorFieldSamples3D
    n_jobs: int

    @property
    def posteriors(self) -> tuple[PosteriorFieldSamples3D, ...]:
        """Each chain's own posterior, in run (``chain_id``) order -- feed this directly to
        ``mixle_pde.verification.mcmc_diagnostics.chains_from_posterior_samples``."""
        return tuple(c.posterior for c in self.chains)


def run_parallel_chains(
    sampler: Sampler,
    grid: Field3D,
    observations: list[Observation],
    registry: ForwardOperatorRegistry,
    prior: FieldGaussianPrior,
    *,
    n_chains: int,
    seed: int | np.random.SeedSequence | None = None,
    initial_unconstrained: Sequence[np.ndarray | None] | None = None,
    n_jobs: int = -1,
    **sampler_kwargs: Any,
) -> MultiChainResult:
    """Run ``n_chains`` independent calls to ``sampler`` across ``n_jobs`` worker processes.

    ``sampler`` is any of :func:`mixle_pde.field_mcmc.metropolis_field_invert`,
    :func:`~mixle_pde.field_mcmc.pcn_field_invert`, :func:`~mixle_pde.field_mcmc.mala_field_invert`,
    :func:`~mixle_pde.field_mcmc.hmc_field_invert`, or any other callable sharing their
    ``(grid, observations, registry, prior, *, initial_unconstrained=, rng=, **kwargs)`` contract.
    ``grid``/``observations``/``registry``/``prior`` are shared, read-only across every chain,
    exactly as they would be for ``n_chains`` sequential single-process calls; only the RNG seed and
    (optionally) the starting point differ per chain.

    ``n_jobs`` follows ``joblib``'s own convention: ``1`` runs every chain sequentially in the
    calling process (no worker spawned at all -- the cheapest possible "serial" reference), a
    positive integer requests exactly that many worker processes, and ``-1`` (the default) requests
    one worker per CPU core. Every chain's result is bit-for-bit identical regardless of ``n_jobs``
    for the same ``seed`` -- no chain's computation depends on how many workers ran alongside it (see
    this module's tests).

    ``seed`` (an ``int``, a ``np.random.SeedSequence``, or ``None`` for fresh OS entropy) is expanded
    into ``n_chains`` independent child seeds via :class:`numpy.random.SeedSequence.spawn`; pass an
    explicit integer for a reproducible batch. ``initial_unconstrained``, if given, must have length
    ``n_chains`` (one starting point per chain, ``None`` entries falling back to ``sampler``'s own
    default); passing genuinely distinct, over-dispersed starting points is what makes a downstream
    R-hat/ESS check meaningful -- identical starting points make chains look converged regardless of
    whether they have actually mixed.

    Raises :class:`ImportError` if ``joblib`` is not installed (``pip install 'mixle-pde[parallel]'``)
    and :class:`ValueError` for a non-positive ``n_chains`` or a mismatched ``initial_unconstrained``
    length.
    """
    n_chains = int(n_chains)
    if n_chains < 1:
        raise ValueError("n_chains must be positive.")
    if initial_unconstrained is not None:
        initial_unconstrained = list(initial_unconstrained)
        if len(initial_unconstrained) != n_chains:
            raise ValueError(
                f"initial_unconstrained must have length n_chains ({n_chains}), got {len(initial_unconstrained)}."
            )
        inits = initial_unconstrained
    else:
        inits = [None] * n_chains

    seeds = _spawn_seeds(seed, n_chains)

    joblib = _require_joblib()
    raw_results = joblib.Parallel(n_jobs=n_jobs, backend="loky")(
        joblib.delayed(_run_one_chain)(sampler, grid, observations, registry, prior, chain_seed, init, sampler_kwargs)
        for chain_seed, init in zip(seeds, inits)
    )

    chains = tuple(
        ChainResult(chain_id=i, seed=s, posterior=post, report=rep)
        for i, (s, (post, rep)) in enumerate(zip(seeds, raw_results))
    )

    combined_samples = np.concatenate([c.posterior.samples for c in chains], axis=0)
    has_logp = all(c.posterior.log_posterior is not None for c in chains)
    combined_logp = np.concatenate([c.posterior.log_posterior for c in chains], axis=0) if has_logp else None
    best = max(chains, key=lambda c: c.report.best_log_posterior)
    combined = PosteriorFieldSamples3D(
        grid=grid,
        samples=combined_samples,
        log_posterior=combined_logp,
        map=best.posterior.map,
        provenance={
            "method": "run_parallel_chains",
            "n_chains": n_chains,
            "seeds": list(seeds),
            "n_jobs": n_jobs,
            "best_chain_id": best.chain_id,
        },
    )
    return MultiChainResult(chains=chains, combined=combined, n_jobs=n_jobs)
