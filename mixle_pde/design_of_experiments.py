"""Design-of-experiments / adaptive sampling for :mod:`mixle_pde.surrogate`.

``mixle_pde.surrogate.distill_forward`` already accepts a pluggable ``sampler(n, rng) -> Sequence``
argument -- the draw that decides *where* the expensive ``teacher`` gets called to build the training
set -- but ships no concrete sampler. Callers were left writing their own ``rng.uniform(...)`` closure
inline (see the ``_uniform_sampler`` helper duplicated across ``tests/test_e6_surrogate.py`` and
``tests/test_p4_surrogate.py``). This module supplies that missing piece: a small family of
``sampler(n, rng) -> (n, d)`` array factories usable as-is with ``distill_forward``, plus a "run this
design and keep a receipt" wrapper around each one.

Three design kinds, in increasing sophistication:

1. **Plain random** (:func:`random_sampler` / :func:`random_design`) -- i.i.d. uniform draws over a
   bounding box. The baseline; no space-filling guarantee.
2. **Space-filling QMC** (:func:`latin_hypercube_sampler` / :func:`sobol_sampler` and their
   ``*_design`` counterparts) -- low-discrepancy designs from ``scipy.stats.qmc`` (already a hard
   dependency of this package; see ``pyproject.toml``). These cover a bounding box far more evenly
   than random draws for the same point count, which is exactly what a one-shot ``distill_forward``
   training sample wants.
3. **Adaptive / error-driven** (:func:`adaptive_design`) -- a small sequential-design loop: bootstrap
   a :class:`~mixle_pde.surrogate.Surrogate` via :func:`~mixle_pde.surrogate.distill_forward` on a
   small space-filling batch, then repeatedly draw a fresh round of candidates (part global
   exploration, part local jitter around the highest-error points found so far), score every one of
   them with the bootstrap surrogate's own :meth:`~mixle_pde.surrogate.Surrogate.evaluate` (true
   held-out error against ``teacher``) and :meth:`~mixle_pde.surrogate.Surrogate.is_ood` (density-gate
   novelty), and keep all of them -- scoring a candidate already teacher-labels it, so nothing spent
   is thrown away. The worst-scoring candidates from each round seed the next round's local jitter,
   concentrating subsequent draws where the surrogate is worst, until the declared budget of
   teacher-labeled points is spent.

Every design run (:func:`random_design`, :func:`latin_hypercube_design`, :func:`sobol_design`,
:func:`adaptive_design`) returns a :class:`TrainingDesignManifest`: design kind, bounds, seed,
requested/actual point count, and a ``hashlib.sha256``-over-canonical-JSON content hash of the
realized coordinates. The hashing idiom (``json.dumps(..., sort_keys=True, separators=(",", ":"))``
then ``hashlib.sha256(...).hexdigest()``) is copied verbatim from
:func:`mixle_pde.ownership.migration_inventory_digest`, so a design is a reproducible, auditable
receipt without needing the still-incomplete artifact store: replaying the same design kind, bounds,
and seed reproduces the same coordinates and therefore the same hash.

Scope is deliberately narrow (a first increment): no multifidelity/co-kriging layout, no distributed
execution, and the adaptive loop refits nothing after its one bootstrap fit. ``mixle_pde/surrogate.py``
is read-only from here -- only imported from, never modified.
"""

from __future__ import annotations

import hashlib
import json
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.stats import qmc

from mixle_pde.surrogate import Surrogate, distill_forward

__all__ = [
    "TrainingDesignManifest",
    "AdaptiveDesignResult",
    "random_sampler",
    "latin_hypercube_sampler",
    "sobol_sampler",
    "random_design",
    "latin_hypercube_design",
    "sobol_design",
    "adaptive_design",
]

# One ``(lo, hi)`` pair per input dimension -- the bounding box every design in this module draws from.
Bounds = Sequence[tuple[float, float]]
# The exact ``sampler`` contract ``distill_forward`` accepts: draw ``n`` candidate inputs given an rng.
Sampler = Callable[[int, np.random.Generator], np.ndarray]

_DESIGN_KINDS = frozenset({"random", "latin_hypercube", "sobol", "adaptive"})


@dataclass(frozen=True)
class TrainingDesignManifest:
    """A reproducibility receipt for one design-of-experiments run.

    Enough to recreate or audit the realized coordinate set without needing the (still-incomplete)
    artifact store: replaying the same ``kind``/``bounds``/``seed`` through this module reproduces the
    same coordinates and therefore the same ``content_hash``. The hash covers only the realized
    coordinates (not ``kind``/``bounds``/``seed`` themselves), mirroring
    :func:`mixle_pde.ownership.migration_inventory_digest`'s canonical-JSON idiom: ``json.dumps`` with
    sorted keys and no whitespace, then ``hashlib.sha256`` over the encoded bytes.
    """

    kind: str
    bounds: tuple[tuple[float, float], ...]
    seed: int
    requested_points: int
    actual_points: int
    content_hash: str

    def __post_init__(self) -> None:
        if self.kind not in _DESIGN_KINDS:
            raise ValueError(f"unsupported design kind {self.kind!r}; expected one of {sorted(_DESIGN_KINDS)}")
        if not self.bounds:
            raise ValueError("bounds must contain at least one (lo, hi) pair")
        for lo, hi in self.bounds:
            if not hi > lo:
                raise ValueError(f"bounds pair ({lo}, {hi}) is not a valid lo < hi interval")
        if self.requested_points <= 0 or self.actual_points <= 0:
            raise ValueError("requested_points and actual_points must be positive")
        if len(self.content_hash) != 64 or any(c not in "0123456789abcdef" for c in self.content_hash):
            raise ValueError("content_hash must be a lowercase sha256 hex digest")


@dataclass
class AdaptiveDesignResult:
    """The output of :func:`adaptive_design`.

    ``points`` holds the realized coordinates in draw order: the initial space-filling batch followed
    by every error-concentrated refinement round. ``points[:n_initial]`` is exactly that initial batch
    (same points, same order) -- the same array :func:`adaptive_design` fed to its own bootstrap
    :func:`~mixle_pde.surrogate.distill_forward` call -- so a caller can refit "initial batch only" vs.
    "full adaptive design" surrogates on directly comparable data. ``bootstrap_surrogate`` is that one
    bootstrap fit, the same surrogate used to score every refinement candidate.
    """

    points: np.ndarray
    manifest: TrainingDesignManifest
    n_initial: int
    bootstrap_surrogate: Surrogate


def _split_bounds(bounds: Bounds) -> tuple[np.ndarray, np.ndarray]:
    pairs = list(bounds)
    if not pairs:
        raise ValueError("bounds must contain at least one (lo, hi) pair")
    lo = np.array([float(pair[0]) for pair in pairs], dtype=np.float64)
    hi = np.array([float(pair[1]) for pair in pairs], dtype=np.float64)
    if np.any(hi <= lo):
        raise ValueError("every bounds pair must satisfy lo < hi")
    return lo, hi


def _canonical_bounds(bounds: Bounds) -> tuple[tuple[float, float], ...]:
    lo, hi = _split_bounds(bounds)
    return tuple((float(a), float(b)) for a, b in zip(lo.tolist(), hi.tolist()))


def _content_hash(points: np.ndarray) -> str:
    """``sha256`` over canonical JSON of ``points`` -- see :func:`mixle_pde.ownership.migration_inventory_digest`."""
    payload = np.asarray(points, dtype=np.float64).tolist()
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _build_manifest(
    *, kind: str, bounds: Bounds, seed: int, requested_points: int, points: np.ndarray
) -> TrainingDesignManifest:
    return TrainingDesignManifest(
        kind=kind,
        bounds=_canonical_bounds(bounds),
        seed=seed,
        requested_points=requested_points,
        actual_points=int(points.shape[0]),
        content_hash=_content_hash(points),
    )


def _fixed_point_sampler(points: np.ndarray) -> Sampler:
    """Replay an already-realized coordinate array as a ``sampler(n, rng)``, ignoring ``rng``.

    Used internally to feed :func:`adaptive_design`'s own initial batch into
    :func:`~mixle_pde.surrogate.distill_forward` without redrawing it (redrawing could diverge from
    the exact points already recorded in the manifest).
    """
    frozen = np.asarray(points, dtype=np.float64)

    def sampler(n: int, rng: np.random.Generator) -> np.ndarray:
        del rng  # the whole point of "fixed" is to ignore any rng state and replay verbatim
        if n != frozen.shape[0]:
            raise ValueError(f"fixed-point sampler holds {frozen.shape[0]} points, requested {n}")
        return frozen

    return sampler


def random_sampler(bounds: Bounds) -> Sampler:
    """A ``sampler(n, rng) -> (n, d)`` array of i.i.d. uniform draws over ``bounds``.

    The baseline design: no space-filling guarantee, just plain Monte Carlo. Usable directly as the
    ``sampler`` argument to :func:`mixle_pde.surrogate.distill_forward`.
    """
    lo, hi = _split_bounds(bounds)
    dim = lo.shape[0]

    def sampler(n: int, rng: np.random.Generator) -> np.ndarray:
        return rng.uniform(lo, hi, size=(int(n), dim))

    return sampler


def latin_hypercube_sampler(bounds: Bounds, *, scramble: bool = True) -> Sampler:
    """A ``sampler(n, rng) -> (n, d)`` array from a scrambled Latin-Hypercube QMC design over ``bounds``.

    One stratified sample per bin along every axis (``scipy.stats.qmc.LatinHypercube``) -- covers a
    bounding box far more evenly than i.i.d. random draws at the same point count. Usable directly as
    the ``sampler`` argument to :func:`mixle_pde.surrogate.distill_forward`.
    """
    lo, hi = _split_bounds(bounds)
    dim = lo.shape[0]

    def sampler(n: int, rng: np.random.Generator) -> np.ndarray:
        engine = qmc.LatinHypercube(d=dim, scramble=scramble, rng=rng)
        unit = engine.random(int(n))
        return qmc.scale(unit, lo, hi)

    return sampler


def sobol_sampler(bounds: Bounds, *, scramble: bool = True) -> Sampler:
    """A ``sampler(n, rng) -> (n, d)`` array from a scrambled Sobol'-sequence QMC design over ``bounds``.

    Usable directly as the ``sampler`` argument to :func:`mixle_pde.surrogate.distill_forward`. Sobol'
    points are only maximally balanced when ``n`` is a power of two; ``distill_forward`` budgets rarely
    are, so scipy's harmless "balance properties" warning for other ``n`` is deliberately suppressed --
    the sequence is still valid, just not guaranteed maximally balanced.
    """
    lo, hi = _split_bounds(bounds)
    dim = lo.shape[0]

    def sampler(n: int, rng: np.random.Generator) -> np.ndarray:
        engine = qmc.Sobol(d=dim, scramble=scramble, rng=rng)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=r"The balance properties of Sobol.*", category=UserWarning)
            unit = engine.random(int(n))
        return qmc.scale(unit, lo, hi)

    return sampler


def _run_design(
    *, kind: str, sampler: Sampler, bounds: Bounds, n_points: int, seed: int
) -> tuple[np.ndarray, TrainingDesignManifest]:
    if n_points <= 0:
        raise ValueError("n_points must be positive")
    rng = np.random.default_rng(seed)
    points = np.asarray(sampler(int(n_points), rng), dtype=np.float64)
    if points.shape[0] != n_points:
        raise ValueError(f"{kind} sampler returned {points.shape[0]} points, requested {n_points}")
    manifest = _build_manifest(kind=kind, bounds=bounds, seed=seed, requested_points=n_points, points=points)
    return points, manifest


def random_design(bounds: Bounds, n_points: int, *, seed: int = 0) -> tuple[np.ndarray, TrainingDesignManifest]:
    """Draw a plain uniform-random design over ``bounds`` and return ``(points, manifest)``."""
    return _run_design(kind="random", sampler=random_sampler(bounds), bounds=bounds, n_points=n_points, seed=seed)


def latin_hypercube_design(
    bounds: Bounds, n_points: int, *, seed: int = 0, scramble: bool = True
) -> tuple[np.ndarray, TrainingDesignManifest]:
    """Draw a Latin-Hypercube QMC design over ``bounds`` and return ``(points, manifest)``."""
    sampler = latin_hypercube_sampler(bounds, scramble=scramble)
    return _run_design(kind="latin_hypercube", sampler=sampler, bounds=bounds, n_points=n_points, seed=seed)


def sobol_design(
    bounds: Bounds, n_points: int, *, seed: int = 0, scramble: bool = True
) -> tuple[np.ndarray, TrainingDesignManifest]:
    """Draw a Sobol'-sequence QMC design over ``bounds`` and return ``(points, manifest)``."""
    sampler = sobol_sampler(bounds, scramble=scramble)
    return _run_design(kind="sobol", sampler=sampler, bounds=bounds, n_points=n_points, seed=seed)


def _score_candidate(surrogate: Surrogate, x: np.ndarray) -> float:
    """Held-out error against the teacher, boosted when ``x`` is also out-of-distribution.

    Uses exactly the two public :class:`~mixle_pde.surrogate.Surrogate` methods meant for this:
    :meth:`~mixle_pde.surrogate.Surrogate.evaluate` (single-point, so its aggregate ``max_abs_error``
    equals that point's own error) for held-out error, and
    :meth:`~mixle_pde.surrogate.Surrogate.is_ood` for density-gate novelty. A point-prediction alone
    gives no useful error estimate once ``x`` is out-of-distribution, so an OOD candidate is bumped up
    by the surrogate's own calibrated precision floor (``tol``) -- enough to outrank any in-distribution
    candidate whose error is already inside that floor (the common, "well-fit" case), while still
    letting a severely mispredicted in-distribution candidate outrank a marginally-OOD one.
    """
    report = surrogate.evaluate([x])
    err = float(np.max(report["max_abs_error"]))
    if surrogate.is_ood(x):
        err += float(np.max(surrogate.tol))
    return err


def adaptive_design(
    teacher: Callable[[Any], Any],
    bounds: Bounds,
    budget: int,
    *,
    seed: int = 0,
    init_fraction: float = 0.4,
    points_per_round: int = 16,
    explore_fraction: float = 0.5,
    elite_fraction: float = 0.25,
    local_scale: float = 0.15,
    **distill_kwargs: Any,
) -> AdaptiveDesignResult:
    """Error-driven sequential design: request more ``teacher``-labeled points where a bootstrap
    :class:`~mixle_pde.surrogate.Surrogate` is worst, until ``budget`` points have been collected.

    Algorithm: draw a small space-filling ``init_fraction`` of ``budget`` (Latin-Hypercube) and
    distill a bootstrap :class:`~mixle_pde.surrogate.Surrogate` from it via
    :func:`~mixle_pde.surrogate.distill_forward`. Then, until ``budget`` is spent, repeat: draw
    ``points_per_round`` candidates for this round (an ``explore_fraction`` share drawn uniformly over
    the whole domain, the rest jittered -- Gaussian, std ``local_scale * (hi - lo)`` per axis -- around
    the highest-error points found in the previous round), score every candidate with
    :func:`_score_candidate` (which calls ``teacher`` exactly once per candidate, via
    :meth:`~mixle_pde.surrogate.Surrogate.evaluate`), and keep all of them -- scoring already
    teacher-labeled them, so nothing paid for is discarded. The ``elite_fraction`` worst-scoring
    candidates from the round become next round's jitter centers, concentrating subsequent draws where
    the surrogate is worst.

    Args:
        teacher: the expensive forward being designed for, ``teacher(x) -> float | array-like``.
        bounds: one ``(lo, hi)`` pair per input dimension.
        budget: total teacher-labeled points to collect, initial batch included
            (:func:`~mixle_pde.surrogate.distill_forward`'s own floor of 16 applies to that batch).
        seed: determinism for every draw in every round, and for the bootstrap fit.
        init_fraction: fraction of ``budget`` spent on the initial space-filling batch.
        points_per_round: candidates drawn -- and, since scoring labels them, kept -- per round.
        explore_fraction: fraction of each round's candidates drawn uniformly over the whole domain
            rather than jittered around the current worst-error frontier.
        elite_fraction: fraction of each round's candidates (by worst score) that seed the next
            round's local jitter centers.
        local_scale: jitter standard deviation for exploitation candidates, as a fraction of each
            dimension's ``(hi - lo)`` extent.
        **distill_kwargs: forwarded to the bootstrap :func:`~mixle_pde.surrogate.distill_forward` call
            (e.g. ``alpha``, ``holdout``, ``ood_alpha``, ``hidden``, ``epochs``, ``lr``).

    Returns:
        An :class:`AdaptiveDesignResult`.
    """
    if budget < 16:
        raise ValueError("adaptive_design needs a budget of at least 16 teacher calls (distill_forward's own floor)")
    lo, hi = _split_bounds(bounds)
    dim = lo.shape[0]
    extent = hi - lo

    n_initial = min(int(budget), max(16, int(round(budget * init_fraction))))
    rng = np.random.default_rng(seed)

    x_init = np.asarray(latin_hypercube_sampler(bounds)(n_initial, rng), dtype=np.float64)
    bootstrap = distill_forward(teacher, _fixed_point_sampler(x_init), budget=n_initial, seed=seed, **distill_kwargs)

    collected = [x_init]
    n_collected = n_initial
    frontier = x_init

    while n_collected < budget:
        round_n = min(points_per_round, budget - n_collected)
        n_explore = min(round_n, max(0, int(round(round_n * explore_fraction))))
        n_exploit = round_n - n_explore

        pieces = []
        if n_explore:
            pieces.append(rng.uniform(lo, hi, size=(n_explore, dim)))
        if n_exploit:
            centers = frontier[rng.integers(0, len(frontier), size=n_exploit)]
            jitter = rng.normal(scale=local_scale * extent, size=(n_exploit, dim))
            pieces.append(np.clip(centers + jitter, lo, hi))
        candidates = np.vstack(pieces)

        scores = np.array([_score_candidate(bootstrap, x) for x in candidates])
        collected.append(candidates)
        n_collected += candidates.shape[0]

        elite_k = max(1, int(round(candidates.shape[0] * elite_fraction)))
        worst_first = np.argsort(scores)[::-1]
        frontier = candidates[worst_first[:elite_k]]

    points = np.vstack(collected)
    manifest = _build_manifest(kind="adaptive", bounds=bounds, seed=seed, requested_points=budget, points=points)
    return AdaptiveDesignResult(points=points, manifest=manifest, n_initial=n_initial, bootstrap_surrogate=bootstrap)
