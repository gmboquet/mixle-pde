"""Checkpoint/restart for :func:`mixle_pde.field_mcmc.metropolis_field_invert` (MP-I9).

MP-I9 ("Scalable, restartable, reproducible Bayesian execution") is recorded not-started in
``docs/reconciliation/mp-task-ledger.md``, and :mod:`mixle_pde.verification.capability_inventory`'s
own methodology note confirms a repo-wide sweep found no ``mpi4py``, ``multiprocessing``, or
``concurrent.futures``/``joblib`` usage anywhere under ``mixle_pde/``. :mod:`mixle_pde.verification.
mcmc_diagnostics` names the same gap explicitly: "No resumability, checkpointing, counter-based
random streams, or distributed/multi-process execution is implemented (the broader MP-I9 scope)."

This module closes the **checkpoint/restart** slice of that gap, and only that slice: a long-running
:func:`~mixle_pde.field_mcmc.metropolis_field_invert` chain can be paused at a caller-chosen cadence,
its exact numerical state (including the sampler RNG's bit-generator state) serialized to disk, and
resumed -- in a later process, with a freshly constructed RNG -- to continue the SAME chain
bit-for-bit, as if it had never been interrupted. It does **not** implement MPI/multiprocessing/
distributed execution, counter-based (splittable) random streams, or checkpointing for any sampler
other than :func:`~mixle_pde.field_mcmc.metropolis_field_invert`; ``parallel_status`` for every
capability-inventory entry remains accurately ``"single_process"`` after this change.

Design: :func:`~mixle_pde.field_mcmc.metropolis_field_invert` is treated as a black box and is never
modified. Advancing a chain by ``L`` raw steps from state ``(current, rng)`` is exactly one call to
that function with ``burn_in=0, thin=1, n_samples=L, initial_unconstrained=current, rng=rng`` -- the
per-step proposal/accept-reject logic and its RNG draw order are untouched, so splitting one
continuous run into segments at any cadence (aligned or not to the caller's own ``burn_in``/``thin``)
never changes the sequence of accept/reject decisions. This module supplies the part
``metropolis_field_invert`` does not: deciding which raw steps the caller's ``(burn_in, thin)``
schedule actually stores, carrying the running "best" sample and acceptance count across segments,
and capturing/restoring the RNG's ``bit_generator.state`` (numpy's own supported mechanism for
resuming a ``Generator`` bit-for-bit; verified against every built-in bit generator, including the
ones whose state contains a numpy array rather than only Python ints).

:class:`MCMCCheckpoint` is the typed, JSON+``npz``-serializable contract; :func:`run_checkpointed`
starts a fresh chain and :func:`resume_checkpointed` continues a loaded one, both checkpointing at a
caller-specified raw-step cadence via ``on_checkpoint`` (paired with :func:`save_checkpoint`/
:func:`load_checkpoint` for the disk path). ``config_digest`` refuses to resume a checkpoint against a
different grid/observation/schedule than the one that produced it, rather than silently mixing state
from two different problems.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from mixle_pde.field_inversion import FieldGaussianPrior
from mixle_pde.field_mcmc import MCMCReport, metropolis_field_invert
from mixle_pde.latent import Field3D, PosteriorFieldSamples3D
from mixle_pde.observations import ForwardOperatorRegistry, Observation

CHECKPOINT_SCHEMA = "mixle_pde.mcmc_checkpoint/v1"

__all__ = [
    "CHECKPOINT_SCHEMA",
    "MCMCCheckpoint",
    "run_checkpointed",
    "resume_checkpointed",
    "save_checkpoint",
    "load_checkpoint",
]


def _json_safe(value: Any) -> Any:
    """Coerce an RNG ``bit_generator.state`` tree into a JSON-serialisable equivalent.

    Every built-in numpy bit generator's state is a dict of plain Python ints except MT19937's,
    whose ``state["state"]["key"]`` is a ``(624,)`` ``uint32`` array; that array is tagged so
    :func:`_state_from_json_safe` can restore the exact dtype rather than guessing from a list.
    """
    if isinstance(value, np.ndarray):
        return {"__ndarray__": True, "dtype": str(value.dtype), "data": value.tolist()}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _state_from_json_safe(value: Any) -> Any:
    """Inverse of :func:`_json_safe`."""
    if isinstance(value, dict):
        if value.get("__ndarray__"):
            return np.array(value["data"], dtype=value["dtype"])
        return {k: _state_from_json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_state_from_json_safe(v) for v in value]
    return value


def _capture_rng(rng: np.random.Generator) -> tuple[str, dict[str, Any]]:
    bit_generator = rng.bit_generator
    return type(bit_generator).__name__, _json_safe(bit_generator.state)


def _restore_rng(bit_generator_name: str, state: dict[str, Any]) -> np.random.Generator:
    try:
        bit_generator_cls = getattr(np.random, bit_generator_name)
    except AttributeError as exc:
        raise ValueError(f"unknown bit generator {bit_generator_name!r}; cannot restore rng state.") from exc
    bit_generator = bit_generator_cls()
    bit_generator.state = _state_from_json_safe(state)
    return np.random.Generator(bit_generator)


def _config_digest(
    grid: Field3D,
    observations: Sequence[Observation],
    step_scale: Any,
    burn_in: int,
    thin: int,
    n_samples: int,
) -> str:
    """A sha256 fingerprint of the problem/schedule a checkpoint was produced against.

    Not cryptographic provenance (see :mod:`mixle_pde.io.artifacts` for that) -- purely a guard so
    :func:`resume_checkpointed` fails loudly on a mismatched grid/observations/schedule instead of
    silently continuing one chain's state as though it belonged to a different problem.
    """
    payload = {
        "grid_n": int(grid.n),
        "property_name": grid.property_name,
        "n_observations": len(observations),
        "observation_kinds": sorted(obs.kind for obs in observations),
        "burn_in": int(burn_in),
        "thin": int(thin),
        "n_samples": int(n_samples),
        "step_scale": np.round(np.asarray(step_scale, dtype=float), 12).tolist(),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _stored_mask(iteration: int, length: int, burn_in: int, thin: int) -> np.ndarray:
    """Which of the raw steps ``iteration+1 .. iteration+length`` a ``(burn_in, thin)`` schedule
    stores, matching :func:`mixle_pde.field_mcmc.metropolis_field_invert`'s own
    ``step > burn_in and (step - burn_in) % thin == 0`` condition exactly."""
    steps = iteration + 1 + np.arange(length)
    return (steps > burn_in) & ((steps - burn_in) % thin == 0)


def _stack_samples(samples: list[np.ndarray], n: int) -> np.ndarray:
    """``list[(n,) array] -> (m, n) array``, keeping the second axis even for ``m == 0`` (still
    within burn-in, so nothing has been stored yet) -- ``np.asarray([])`` would otherwise collapse
    an empty list to shape ``(0,)`` instead of ``(0, n)``."""
    if not samples:
        return np.empty((0, n), dtype=float)
    return np.asarray(samples, dtype=float)


@dataclass(frozen=True)
class MCMCCheckpoint:
    """A serialisable snapshot of an in-progress :func:`~mixle_pde.field_mcmc.metropolis_field_invert`
    chain: enough state to continue it bit-for-bit from ``iteration`` raw steps onward, in this or a
    later process. Does not capture ``grid``/``observations``/``registry``/``prior`` -- the caller
    supplies those again on resume, exactly as it would re-run any script from source; ``config_digest``
    guards against resuming against a different problem than the one that produced the checkpoint.
    """

    schema: str
    iteration: int
    total_steps: int
    burn_in: int
    thin: int
    n_samples: int
    grid_n: int
    step_scale: np.ndarray
    current: np.ndarray
    current_logp: float
    best: np.ndarray
    best_logp: float
    accepted: int
    stored_samples: np.ndarray
    stored_log_posterior: np.ndarray
    rng_bit_generator: str
    rng_state: dict[str, Any]
    config_digest: str

    def __post_init__(self) -> None:
        if self.total_steps != self.burn_in + self.n_samples * self.thin:
            raise ValueError("total_steps must equal burn_in + n_samples * thin.")
        if not (0 <= self.iteration <= self.total_steps):
            raise ValueError(f"iteration must be in [0, {self.total_steps}], got {self.iteration}.")
        n = int(self.grid_n)
        current = np.asarray(self.current, dtype=float)
        if current.shape != (n,):
            raise ValueError(f"current must have shape ({n},), got {current.shape}.")
        object.__setattr__(self, "current", current)
        best = np.asarray(self.best, dtype=float)
        if best.shape != (n,):
            raise ValueError(f"best must have shape ({n},), got {best.shape}.")
        object.__setattr__(self, "best", best)
        stored_samples = np.asarray(self.stored_samples, dtype=float)
        if stored_samples.ndim != 2 or stored_samples.shape[1] != n:
            raise ValueError(f"stored_samples must have shape (m, {n}), got {stored_samples.shape}.")
        object.__setattr__(self, "stored_samples", stored_samples)
        stored_log_posterior = np.asarray(self.stored_log_posterior, dtype=float)
        if stored_log_posterior.shape != (stored_samples.shape[0],):
            raise ValueError("stored_log_posterior must have shape (m,) matching stored_samples.")
        object.__setattr__(self, "stored_log_posterior", stored_log_posterior)
        object.__setattr__(self, "step_scale", np.asarray(self.step_scale, dtype=float))


def save_checkpoint(checkpoint: MCMCCheckpoint, path: str | Path) -> None:
    """Write ``{path}.npz`` (arrays) + ``{path}.json`` (scalars/rng state), the same array/header
    split :func:`mixle_pde.io.artifacts.save_posterior` uses (not imported here -- this module never
    touches ``mixle_pde/io/artifacts.py``)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        f"{path}.npz",
        current=checkpoint.current,
        best=checkpoint.best,
        stored_samples=checkpoint.stored_samples,
        stored_log_posterior=checkpoint.stored_log_posterior,
        step_scale=checkpoint.step_scale,
    )
    header = {
        "schema": checkpoint.schema,
        "iteration": checkpoint.iteration,
        "total_steps": checkpoint.total_steps,
        "burn_in": checkpoint.burn_in,
        "thin": checkpoint.thin,
        "n_samples": checkpoint.n_samples,
        "grid_n": checkpoint.grid_n,
        "current_logp": checkpoint.current_logp,
        "best_logp": checkpoint.best_logp,
        "accepted": checkpoint.accepted,
        "rng_bit_generator": checkpoint.rng_bit_generator,
        "rng_state": checkpoint.rng_state,
        "config_digest": checkpoint.config_digest,
    }
    with open(f"{path}.json", "w") as fh:
        json.dump(header, fh, indent=2, sort_keys=True)


def load_checkpoint(path: str | Path) -> MCMCCheckpoint:
    """Inverse of :func:`save_checkpoint`."""
    path = Path(path)
    with open(f"{path}.json") as fh:
        header = json.load(fh)
    with np.load(f"{path}.npz") as npz:
        current = np.array(npz["current"], dtype=float)
        best = np.array(npz["best"], dtype=float)
        stored_samples = np.array(npz["stored_samples"], dtype=float)
        stored_log_posterior = np.array(npz["stored_log_posterior"], dtype=float)
        step_scale = np.array(npz["step_scale"], dtype=float)
    return MCMCCheckpoint(
        schema=header["schema"],
        iteration=header["iteration"],
        total_steps=header["total_steps"],
        burn_in=header["burn_in"],
        thin=header["thin"],
        n_samples=header["n_samples"],
        grid_n=header["grid_n"],
        step_scale=step_scale,
        current=current,
        current_logp=header["current_logp"],
        best=best,
        best_logp=header["best_logp"],
        accepted=header["accepted"],
        stored_samples=stored_samples,
        stored_log_posterior=stored_log_posterior,
        rng_bit_generator=header["rng_bit_generator"],
        rng_state=header["rng_state"],
        config_digest=header["config_digest"],
    )


def _finalize(
    grid: Field3D,
    *,
    burn_in: int,
    thin: int,
    stored_samples: list[np.ndarray],
    stored_log_posterior: list[float],
    best: np.ndarray,
    best_logp: float,
    accepted: int,
    current_logp: float,
    total_steps: int,
    step_scale: np.ndarray,
    checkpointed: bool,
) -> tuple[PosteriorFieldSamples3D, MCMCReport]:
    posterior = PosteriorFieldSamples3D(
        grid=grid,
        samples=_stack_samples(stored_samples, grid.n),
        log_posterior=np.asarray(stored_log_posterior, dtype=float),
        map=best,
        provenance={
            "method": "random_walk_metropolis",
            "small_reference": True,
            "burn_in": burn_in,
            "thin": thin,
            "step_scale": np.asarray(step_scale, dtype=float).tolist(),
            "checkpointed": checkpointed,
        },
    )
    report = MCMCReport(
        iterations=total_steps,
        burn_in=burn_in,
        thin=thin,
        proposed=total_steps,
        accepted=accepted,
        acceptance_rate=accepted / total_steps if total_steps else 0.0,
        stored_samples=len(stored_samples),
        final_log_posterior=current_logp,
        best_log_posterior=best_logp,
    )
    return posterior, report


def _run_segments(
    grid: Field3D,
    observations: list[Observation],
    registry: ForwardOperatorRegistry,
    prior: FieldGaussianPrior,
    *,
    burn_in: int,
    thin: int,
    n_samples: int,
    step_scale: Any,
    total_steps: int,
    iteration: int,
    current: np.ndarray | None,
    current_logp: float | None,
    best: np.ndarray | None,
    best_logp: float,
    accepted: int,
    stored_samples: list[np.ndarray],
    stored_log_posterior: list[float],
    rng: np.random.Generator,
    checkpoint_every: int,
    on_checkpoint: Callable[[MCMCCheckpoint], None] | None,
    config_digest: str,
) -> tuple[PosteriorFieldSamples3D, MCMCReport, MCMCCheckpoint]:
    if checkpoint_every <= 0:
        raise ValueError("checkpoint_every must be positive.")

    while iteration < total_steps:
        length = min(int(checkpoint_every), total_steps - iteration)
        seg_posterior, seg_report = metropolis_field_invert(
            grid,
            observations,
            registry,
            prior,
            n_samples=length,
            burn_in=0,
            thin=1,
            step_scale=step_scale,
            initial_unconstrained=current,
            rng=rng,
        )
        mask = _stored_mask(iteration, length, burn_in, thin)
        if np.any(mask):
            stored_samples.extend(seg_posterior.samples[mask])
            stored_log_posterior.extend(float(v) for v in seg_posterior.log_posterior[mask])

        accepted += seg_report.accepted
        current = seg_posterior.samples[-1]
        current_logp = float(seg_posterior.log_posterior[-1])
        seg_best_logp = float(seg_report.best_log_posterior)
        if best is None or seg_best_logp > best_logp:
            best = seg_posterior.map.copy()
            best_logp = seg_best_logp
        iteration += length

        bit_generator_name, rng_state = _capture_rng(rng)
        checkpoint = MCMCCheckpoint(
            schema=CHECKPOINT_SCHEMA,
            iteration=iteration,
            total_steps=total_steps,
            burn_in=burn_in,
            thin=thin,
            n_samples=n_samples,
            grid_n=grid.n,
            step_scale=np.asarray(step_scale, dtype=float),
            current=current,
            current_logp=current_logp,
            best=best,
            best_logp=best_logp,
            accepted=accepted,
            stored_samples=_stack_samples(stored_samples, grid.n),
            stored_log_posterior=np.asarray(stored_log_posterior, dtype=float),
            rng_bit_generator=bit_generator_name,
            rng_state=rng_state,
            config_digest=config_digest,
        )
        if on_checkpoint is not None:
            on_checkpoint(checkpoint)

    if current_logp is None or best is None or current is None:
        # Not reachable through the public entry points: run_checkpointed always starts at
        # iteration=0 < total_steps (total_steps >= 1 whenever n_samples >= 1), and
        # resume_checkpointed short-circuits an already-complete checkpoint (iteration ==
        # total_steps) before ever calling this function, using the checkpoint's own state
        # directly. Guards a direct/future caller of this private helper instead of finalizing
        # from unset state.
        raise ValueError("_run_segments produced no segments; total_steps was already reached at entry.")

    posterior, report = _finalize(
        grid,
        burn_in=burn_in,
        thin=thin,
        stored_samples=stored_samples,
        stored_log_posterior=stored_log_posterior,
        best=best,
        best_logp=best_logp,
        accepted=accepted,
        current_logp=current_logp,
        total_steps=total_steps,
        step_scale=step_scale,
        checkpointed=True,
    )
    return posterior, report, checkpoint


def run_checkpointed(
    grid: Field3D,
    observations: list[Observation],
    registry: ForwardOperatorRegistry,
    prior: FieldGaussianPrior,
    *,
    n_samples: int,
    burn_in: int = 1000,
    thin: int = 1,
    step_scale: float | np.ndarray = 1.0,
    initial_unconstrained: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
    checkpoint_every: int,
    on_checkpoint: Callable[[MCMCCheckpoint], None] | None = None,
) -> tuple[PosteriorFieldSamples3D, MCMCReport, MCMCCheckpoint]:
    """Run :func:`~mixle_pde.field_mcmc.metropolis_field_invert` to completion in segments of
    ``checkpoint_every`` raw steps, calling ``on_checkpoint(checkpoint)`` after each segment.

    Identical arguments to ``metropolis_field_invert`` plus ``checkpoint_every``/``on_checkpoint``.
    Returns ``(posterior, report, checkpoint)`` where ``posterior``/``report`` are bit-for-bit
    identical to calling ``metropolis_field_invert`` directly with the same ``rng`` seed and no
    checkpointing, for any ``checkpoint_every`` (including one that does not evenly divide
    ``burn_in``/``thin``/``n_samples``) -- see the module docstring for why. ``checkpoint`` is the
    final (completed) snapshot; pass it (or a value loaded via :func:`load_checkpoint` after
    :func:`save_checkpoint`) to :func:`resume_checkpointed` to continue an interrupted run.
    """
    rng = np.random.default_rng() if rng is None else rng
    total_steps = int(burn_in) + int(n_samples) * int(thin)
    config_digest = _config_digest(grid, observations, step_scale, burn_in, thin, n_samples)
    return _run_segments(
        grid,
        observations,
        registry,
        prior,
        burn_in=int(burn_in),
        thin=int(thin),
        n_samples=int(n_samples),
        step_scale=step_scale,
        total_steps=total_steps,
        iteration=0,
        current=initial_unconstrained,
        current_logp=None,
        best=None,
        best_logp=float("-inf"),
        accepted=0,
        stored_samples=[],
        stored_log_posterior=[],
        rng=rng,
        checkpoint_every=checkpoint_every,
        on_checkpoint=on_checkpoint,
        config_digest=config_digest,
    )


def resume_checkpointed(
    checkpoint: MCMCCheckpoint,
    grid: Field3D,
    observations: list[Observation],
    registry: ForwardOperatorRegistry,
    prior: FieldGaussianPrior,
    *,
    checkpoint_every: int | None = None,
    on_checkpoint: Callable[[MCMCCheckpoint], None] | None = None,
) -> tuple[PosteriorFieldSamples3D, MCMCReport, MCMCCheckpoint]:
    """Continue a chain from ``checkpoint`` to its recorded ``total_steps``, in a process that may
    share nothing with the one that produced it beyond ``checkpoint`` itself (typically loaded via
    :func:`load_checkpoint`) and the same ``grid``/``observations``/``registry``/``prior`` the
    original run used. Raises ``ValueError`` if ``grid``/``observations``/schedule do not match the
    checkpoint's own ``config_digest``. ``checkpoint_every`` defaults to finishing in one segment;
    pass a smaller value to keep checkpointing as the resumed run continues.
    """
    expected_digest = _config_digest(
        grid, observations, checkpoint.step_scale, checkpoint.burn_in, checkpoint.thin, checkpoint.n_samples
    )
    if expected_digest != checkpoint.config_digest:
        raise ValueError(
            "checkpoint config_digest does not match the supplied grid/observations/schedule; "
            "refusing to resume against a different problem."
        )
    if grid.n != checkpoint.grid_n:
        raise ValueError(f"checkpoint was captured for grid.n={checkpoint.grid_n}, got grid.n={grid.n}.")

    remaining = checkpoint.total_steps - checkpoint.iteration
    if checkpoint_every is None:
        checkpoint_every = max(remaining, 1)

    if remaining <= 0:
        # Already complete: no segment to run. _run_segments's while loop requires at least one
        # segment (see its own trailing guard), so this case is finalized directly here, from the
        # checkpoint's own already-consistent state, instead of calling it.
        posterior, report = _finalize(
            grid,
            burn_in=checkpoint.burn_in,
            thin=checkpoint.thin,
            stored_samples=list(checkpoint.stored_samples),
            stored_log_posterior=list(checkpoint.stored_log_posterior),
            best=checkpoint.best,
            best_logp=checkpoint.best_logp,
            accepted=checkpoint.accepted,
            current_logp=checkpoint.current_logp,
            total_steps=checkpoint.total_steps,
            step_scale=checkpoint.step_scale,
            checkpointed=True,
        )
        return posterior, report, checkpoint

    rng = _restore_rng(checkpoint.rng_bit_generator, checkpoint.rng_state)
    return _run_segments(
        grid,
        observations,
        registry,
        prior,
        burn_in=checkpoint.burn_in,
        thin=checkpoint.thin,
        n_samples=checkpoint.n_samples,
        step_scale=checkpoint.step_scale,
        total_steps=checkpoint.total_steps,
        iteration=checkpoint.iteration,
        current=checkpoint.current,
        current_logp=checkpoint.current_logp,
        best=checkpoint.best,
        best_logp=checkpoint.best_logp,
        accepted=checkpoint.accepted,
        stored_samples=list(checkpoint.stored_samples),
        stored_log_posterior=list(checkpoint.stored_log_posterior),
        rng=rng,
        checkpoint_every=checkpoint_every,
        on_checkpoint=on_checkpoint,
        config_digest=checkpoint.config_digest,
    )
