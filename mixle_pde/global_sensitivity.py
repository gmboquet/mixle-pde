"""Variance-based global sensitivity analysis: Saltelli pick-freeze Sobol' indices and Morris screening.

Global sensitivity analysis (GSA) asks how much of a quantity-of-interest (QoI)'s output variance -- across
the *entire* input parameter range, not a single point -- is attributable to each input parameter, alone or
in combination with the others. This is a different question from the local, single-point derivative
:func:`mixle_pde.dynamics.integrate_sensitivity` answers (a Jacobian ``dY/dp`` at one parameter value); GSA
instead characterizes importance as an expectation over a parameter *distribution*, which is the right
question when deciding which of a model's parameters are worth calibrating, worth spending compute to nail
down, or safe to fix at a nominal value.

Two complementary methods are implemented, both variance/screening classics (Saltelli et al. 2008, "Global
Sensitivity Analysis: The Primer"):

* **Sobol' indices** (:func:`saltelli_design` / :func:`saltelli_sample` + :func:`sobol_indices`) --
  quantitative variance decomposition via Saltelli's (2010) "pick-freeze" estimator. The first-order index
  ``S_i = V[E[Y|X_i]] / V(Y)`` is the fraction of output variance explained by ``X_i`` alone (averaged over
  the other parameters); the total-order index ``S_Ti = E[V[Y|X_~i]] / V(Y)`` is the fraction explained by
  every term involving ``X_i``, alone or interacting. ``S_Ti > S_i`` signals ``X_i`` interacts with other
  parameters; ``S_Ti approx 0`` is the standard criterion for "safe to fix this parameter."
* **Morris elementary effects** (:func:`morris_design` + :func:`morris_indices`) -- a cheap qualitative
  *screening* method (Morris 1991, refined by Campolongo et al. 2007) for when a full Sobol' study (whose
  cost scales with ``n_base * (n_params + 2)`` evaluations) is too expensive to even decide which parameters
  matter. Each of ``r`` random trajectories takes ``n_params + 1`` QoI evaluations, one coordinate change at
  a time, so screening costs ``r * (n_params + 1)`` evaluations -- typically far fewer than a full Sobol'
  study, at the cost of a ranking rather than a variance decomposition. ``mu_star`` (mean *absolute*
  elementary effect) is the primary importance ranking; ``mu`` (signed mean) and ``sigma`` (std of the
  effects) additionally distinguish monotonic-linear (``sigma`` small relative to ``|mu|``) from
  non-monotonic-or-interacting (``sigma`` large relative to ``|mu|``) influence.

Both methods follow this repo's solver-agnostic "caller supplies the data, module does the numerics" pattern
(see :mod:`mixle_pde.reduced_basis`): this module never imports or calls a PDE/ODE solver. A design function
(:func:`saltelli_design`/:func:`saltelli_sample`, :func:`morris_design`) builds the specific parameter
*ensemble* -- an ``(n_evaluations, n_params)`` array -- that the caller must evaluate their own QoI at (a PDE
forward solve, an ODE integration, an analytic function, anything returning one scalar per row), in row
order; the resulting ``(n_evaluations,)`` QoI array is then passed back in to :func:`sobol_indices` /
:func:`morris_indices`, which do the pure index-arithmetic numerics and never touch a solver.

Scope note: this module covers first-order and total-order Sobol' indices (not second-order/interaction
indices individually -- ``S_Ti - S_i`` already signals *that* ``X_i`` interacts, just not *with which* other
parameter) and classic random-trajectory Morris sampling (not the Campolongo et al. "optimal trajectories"
oversample-and-select refinement, which trades more up-front sampling cost for better trajectory
space-filling). Neither estimator here returns a bootstrap confidence interval; both are point estimates
whose accuracy is controlled by the sample size the caller chooses (``n_base`` for Sobol', ``n_trajectories``
for Morris). Both methods assume independent parameters and a single scalar QoI (for a vector-valued QoI,
call the analysis function once per component); for dependent or non-uniform-marginal inputs, transform to
independent uniforms (e.g. via each marginal's inverse CDF) before calling the design functions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import qmc

__all__ = [
    "SaltelliDesign",
    "SobolIndices",
    "saltelli_design",
    "saltelli_sample",
    "sobol_indices",
    "MorrisDesign",
    "MorrisResult",
    "morris_design",
    "morris_indices",
]


def _rng_or_default(rng: np.random.Generator | None) -> np.random.Generator:
    return rng if rng is not None else np.random.default_rng()


def _validate_bounds(bounds) -> tuple[np.ndarray, np.ndarray]:
    """Validate a ``(n_params, 2)`` sequence of ``(lo, hi)`` pairs and split it into ``lo``, ``hi`` arrays."""
    bounds = np.asarray(bounds, dtype=float)
    if bounds.ndim != 2 or bounds.shape[1] != 2:
        raise ValueError(f"bounds must be a sequence of (lo, hi) pairs, shape (n_params, 2); got shape {bounds.shape}.")
    if bounds.shape[0] < 1:
        raise ValueError("bounds must specify at least one parameter.")
    lo, hi = bounds[:, 0].copy(), bounds[:, 1].copy()
    if not np.all(hi > lo):
        raise ValueError(f"every bound must have hi > lo; got lo={lo.tolist()}, hi={hi.tolist()}.")
    return lo, hi


# ---------------------------------------------------------------------------
# Sobol' indices via Saltelli's pick-freeze estimator
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SaltelliDesign:
    """A Saltelli pick-freeze parameter ensemble, ready for the caller to evaluate a QoI at.

    ``points`` (shape ``(n_base * (n_params + 2), n_params)``) stacks, in row-block order: the ``A`` base
    sample (``n_base`` rows), the ``B`` base sample (``n_base`` rows), then for each parameter ``i`` in
    ``range(n_params)``, the "``AB_i``" recombination -- ``A`` with column ``i`` replaced by ``B``'s column
    ``i`` (``n_base`` rows each). This is the classic Saltelli (2002/2010) design: ``n_base * (n_params + 2)``
    total evaluations buys both first-order and total-order indices for every parameter at once (no extra
    evaluations needed for total-order beyond what first-order already requires).

    Build one with :func:`saltelli_design` (from your own ``A``/``B`` base samples) or :func:`saltelli_sample`
    (drawing ``A``/``B`` via scipy QMC from parameter bounds). Evaluate your QoI at every row of ``points``
    in order, then pass the resulting ``(n_evaluations,)`` array to :func:`sobol_indices` together with this
    design.
    """

    points: np.ndarray
    n_base: int
    n_params: int

    def __post_init__(self) -> None:
        if self.points.ndim != 2:
            raise ValueError(
                f"SaltelliDesign.points must be 2-D (n_evaluations, n_params); got shape {self.points.shape}."
            )
        expected = (self.n_base * (self.n_params + 2), self.n_params)
        if self.points.shape != expected:
            raise ValueError(
                f"SaltelliDesign.points must have shape (n_base*(n_params+2), n_params) = {expected}; got "
                f"{self.points.shape}."
            )

    @property
    def n_evaluations(self) -> int:
        """Number of QoI evaluations this design requires: ``n_base * (n_params + 2)``."""
        return self.n_base * (self.n_params + 2)


@dataclass(frozen=True)
class SobolIndices:
    """First-order and total-order Sobol' indices estimated by :func:`sobol_indices`.

    ``first_order[i]`` is ``S_i = V[E[Y|X_i]] / V(Y)`` (variance explained by ``X_i`` alone); ``total_order[i]``
    is ``S_Ti = E[V[Y|X_~i]] / V(Y)`` (variance explained by every term involving ``X_i``, alone or
    interacting). Both are Monte Carlo point estimates from finitely many samples, so for a parameter with a
    truly-zero index, the estimate can land slightly negative -- that is expected estimator noise, not a bug,
    and shrinks as ``n_base`` grows. ``variance`` is the estimated total QoI variance ``V(Y)`` (from the
    pooled ``A``/``B`` samples) both indices are normalized by.
    """

    first_order: np.ndarray
    total_order: np.ndarray
    variance: float
    n_base: int

    def __post_init__(self) -> None:
        if self.first_order.shape != self.total_order.shape:
            raise ValueError(
                f"first_order and total_order must have matching shape; got {self.first_order.shape} vs "
                f"{self.total_order.shape}."
            )


def saltelli_design(a, b) -> SaltelliDesign:
    """Build a :class:`SaltelliDesign` pick-freeze ensemble from two independent base parameter samples.

    ``a`` and ``b`` are ``(n_base, n_params)`` arrays of independent draws from the same parameter
    distribution (e.g. two disjoint halves of a QMC sequence, as :func:`saltelli_sample` builds, or any other
    independent-sampling scheme the caller prefers -- rejection sampling, Latin hypercube, MCMC draws, ...).
    This function only does the pick-freeze recombination (building the ``AB_i`` matrices); it never draws
    samples itself. Use :func:`saltelli_sample` for a self-contained convenience that also draws ``a``/``b``
    from parameter bounds.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError(f"a and b must be 2-D (n_base, n_params) arrays; got shapes {a.shape} and {b.shape}.")
    if a.shape != b.shape:
        raise ValueError(f"a and b must have identical shape; got {a.shape} vs {b.shape}.")
    n_base, n_params = a.shape
    if n_base < 1:
        raise ValueError("a/b must contain at least one base-sample row.")
    if n_params < 1:
        raise ValueError("a/b must have at least one parameter column.")

    blocks = [a, b]
    for i in range(n_params):
        ab_i = a.copy()
        ab_i[:, i] = b[:, i]
        blocks.append(ab_i)

    return SaltelliDesign(points=np.concatenate(blocks, axis=0), n_base=n_base, n_params=n_params)


def saltelli_sample(bounds, n_base: int, *, rng: np.random.Generator | None = None) -> SaltelliDesign:
    """Draw a :class:`SaltelliDesign` from independent-uniform ``bounds`` via a scrambled Sobol' QMC sequence.

    ``bounds`` is a ``(n_params, 2)`` sequence of ``(lo, hi)`` pairs (one per parameter; every parameter is
    treated as an independent Uniform(lo, hi) -- see the module scope note for non-uniform marginals).
    ``n_base`` must be a power of two: the ``A``/``B`` base samples are the two halves of one
    ``2 * n_params``-dimensional scrambled Sobol' sequence (drawing both halves from one higher-dimensional
    sequence, rather than two independent sequences, is the standard construction -- Saltelli et al. 2010 --
    that keeps ``A`` and ``B`` low-discrepancy *and* mutually independent); scipy's Sobol' sampler only
    guarantees its low-discrepancy balance properties when the requested count is a power of two, so that
    requirement is enforced here rather than silently degrading QMC quality for an arbitrary ``n_base``.
    """
    lo, hi = _validate_bounds(bounds)
    n_params = lo.shape[0]
    n_base = int(n_base)
    if n_base < 1 or (n_base & (n_base - 1)) != 0:
        raise ValueError(f"n_base must be a power of two (Sobol' QMC balance property); got {n_base!r}.")
    rng = _rng_or_default(rng)

    sampler = qmc.Sobol(d=2 * n_params, scramble=True, rng=rng)
    unit = sampler.random_base2(m=n_base.bit_length() - 1)
    a = qmc.scale(unit[:, :n_params], lo, hi)
    b = qmc.scale(unit[:, n_params:], lo, hi)
    return saltelli_design(a, b)


def sobol_indices(design: SaltelliDesign, qoi) -> SobolIndices:
    """Estimate first- and total-order Sobol' indices from QoI evaluations at a :class:`SaltelliDesign`.

    ``qoi`` must be a 1-D array of ``design.n_evaluations`` scalar QoI values, one per row of
    ``design.points`` **in that exact order** (``A``, then ``B``, then ``AB_0, AB_1, ..., AB_{n_params-1}``
    -- the order :func:`saltelli_design`/:func:`saltelli_sample` produced). Uses the Saltelli (2010) / Jansen
    (1999) estimator pair (the combination the reference literature -- and SALib, the field's reference
    implementation -- treat as the accurate, numerically stable default)::

        S_i  = mean(Y_B * (Y_ABi - Y_A)) / Var([Y_A, Y_B])
        S_Ti = 0.5 * mean((Y_A - Y_ABi) ** 2) / Var([Y_A, Y_B])

    Raises ``ValueError`` if the QoI is (numerically) constant across the base samples: the indices are a
    *fraction of output variance*, undefined when that variance is zero, and silently returning ``nan``/``inf``
    would be a wrong-number bug, not a graceful degradation.
    """
    qoi = np.asarray(qoi, dtype=float)
    if qoi.ndim != 1:
        raise ValueError(f"qoi must be a 1-D array of scalar QoI evaluations; got shape {qoi.shape}.")
    expected = design.n_evaluations
    if qoi.shape[0] != expected:
        raise ValueError(
            f"qoi must have exactly design.n_evaluations={expected} entries (n_base={design.n_base} * "
            f"(n_params={design.n_params} + 2)), one per row of design.points in order; got {qoi.shape[0]}."
        )

    n, d = design.n_base, design.n_params
    y_a = qoi[:n]
    y_b = qoi[n : 2 * n]
    y_ab = qoi[2 * n :].reshape(d, n).T  # (n, d); column i is f(AB_i)

    combined = np.concatenate([y_a, y_b])
    variance = float(np.var(combined))
    if not np.isfinite(variance) or variance <= 0.0:
        raise ValueError(
            "qoi has zero (or non-finite) variance across the A/B base samples -- Sobol' indices are a "
            f"fraction of output variance and are undefined when that variance is {variance!r}; this usually "
            "means the QoI is constant over the sampled parameter ranges."
        )

    first_order = np.mean(y_b[:, None] * (y_ab - y_a[:, None]), axis=0) / variance
    total_order = 0.5 * np.mean((y_a[:, None] - y_ab) ** 2, axis=0) / variance

    return SobolIndices(first_order=first_order, total_order=total_order, variance=variance, n_base=n)


# ---------------------------------------------------------------------------
# Morris elementary-effects screening
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MorrisDesign:
    """A Morris elementary-effects trajectory ensemble, ready for the caller to evaluate a QoI at.

    ``points`` (shape ``(n_trajectories * (n_params + 1), n_params)``) stacks ``n_trajectories`` trajectories
    of ``n_params + 1`` points each; within a trajectory, consecutive points differ in exactly one parameter,
    offset by one grid step (Morris 1991's discretized OAT -- "one at a time" -- design). ``changed_param``
    (shape ``(n_trajectories, n_params)``, int) records, for trajectory ``t`` and step ``k``, which parameter
    column changed between ``points`` rows ``k`` and ``k + 1`` of that trajectory (``design.points`` reshaped
    to ``(n_trajectories, n_params + 1, n_params)``).

    Build one with :func:`morris_design`. Evaluate your QoI at every row of ``points`` in order, then pass the
    resulting ``(n_evaluations,)`` array to :func:`morris_indices` together with this design.
    """

    points: np.ndarray
    changed_param: np.ndarray
    n_trajectories: int
    n_params: int

    def __post_init__(self) -> None:
        if self.points.ndim != 2:
            raise ValueError(
                f"MorrisDesign.points must be 2-D (n_evaluations, n_params); got shape {self.points.shape}."
            )
        expected_points = (self.n_trajectories * (self.n_params + 1), self.n_params)
        if self.points.shape != expected_points:
            raise ValueError(
                f"MorrisDesign.points must have shape (n_trajectories*(n_params+1), n_params) = "
                f"{expected_points}; got {self.points.shape}."
            )
        expected_changed = (self.n_trajectories, self.n_params)
        if self.changed_param.shape != expected_changed:
            raise ValueError(
                f"MorrisDesign.changed_param must have shape (n_trajectories, n_params) = {expected_changed}; "
                f"got {self.changed_param.shape}."
            )

    @property
    def n_evaluations(self) -> int:
        """Number of QoI evaluations this design requires: ``n_trajectories * (n_params + 1)``."""
        return self.n_trajectories * (self.n_params + 1)


@dataclass(frozen=True)
class MorrisResult:
    """Elementary-effect screening statistics estimated by :func:`morris_indices`.

    ``mu_star[i]`` (mean *absolute* elementary effect) is the primary importance ranking: large ``mu_star``
    means ``X_i`` moves the QoI a lot, small means it barely matters, and -- because it is an absolute value
    -- it cannot be fooled by a non-monotonic effect whose positive and negative swings cancel in a signed
    average (Campolongo et al. 2007). ``mu[i]`` is that signed mean (sign shows the average direction of
    effect, when meaningful); ``sigma[i]`` is the elementary effects' standard deviation, which is large when
    ``X_i``'s effect is strongly nonlinear or interacts with other parameters, and near zero when its effect
    is an (approximately) constant slope.
    """

    mu: np.ndarray
    mu_star: np.ndarray
    sigma: np.ndarray
    n_trajectories: int

    def __post_init__(self) -> None:
        if not (self.mu.shape == self.mu_star.shape == self.sigma.shape):
            raise ValueError(
                f"mu, mu_star, and sigma must have matching shape; got {self.mu.shape}, {self.mu_star.shape}, "
                f"{self.sigma.shape}."
            )


def morris_design(
    bounds,
    n_trajectories: int,
    *,
    n_levels: int = 4,
    rng: np.random.Generator | None = None,
) -> MorrisDesign:
    """Draw ``n_trajectories`` random Morris (1991) elementary-effects trajectories over ``bounds``.

    ``bounds`` is a ``(n_params, 2)`` sequence of ``(lo, hi)`` pairs, one per parameter. Each parameter is
    discretized onto an ``n_levels``-point grid over ``[lo, hi]``; ``n_levels`` must be an even integer
    (Morris's standard construction restricts each trajectory's random base point to the half of the grid
    that leaves room for a same-sized step in the chosen direction, which requires an even number of levels
    to split cleanly -- ``n_levels=4`` is the classic default). A trajectory takes a random base point, then --
    in a random order, one parameter at a time -- steps each parameter by one grid unit (``delta = n_levels /
    (2 * (n_levels - 1))`` of the ``[lo, hi]`` range, in whichever of ``+``/``-`` keeps every step inside
    ``[lo, hi]``), producing ``n_params + 1`` points per trajectory that visit every parameter exactly once
    each.
    """
    lo, hi = _validate_bounds(bounds)
    n_params = lo.shape[0]
    n_trajectories = int(n_trajectories)
    if n_trajectories < 1:
        raise ValueError(f"n_trajectories must be a positive integer; got {n_trajectories!r}.")
    n_levels = int(n_levels)
    if n_levels < 2 or n_levels % 2 != 0:
        raise ValueError(f"n_levels must be an even integer >= 2 (Morris 1991 grid construction); got {n_levels!r}.")
    rng = _rng_or_default(rng)

    p = n_levels
    delta = p / (2.0 * (p - 1))
    grid = np.arange(p) / (p - 1)  # unit-cube grid levels, [0, 1]
    low_half = grid[grid <= 1.0 - delta + 1.0e-9]  # valid base levels for a "+delta" step
    high_half = grid[grid >= delta - 1.0e-9]  # valid base levels for a "-delta" step

    unit_points = np.empty((n_trajectories, n_params + 1, n_params))
    changed_param = np.empty((n_trajectories, n_params), dtype=int)
    for t in range(n_trajectories):
        directions = rng.choice(np.array([-1.0, 1.0]), size=n_params)
        base = np.array([rng.choice(low_half) if directions[j] > 0 else rng.choice(high_half) for j in range(n_params)])
        order = rng.permutation(n_params)

        current = base.copy()
        unit_points[t, 0] = current
        for k, j in enumerate(order):
            current = current.copy()
            current[j] += directions[j] * delta
            unit_points[t, k + 1] = current
            changed_param[t, k] = j

    points = lo + unit_points.reshape(-1, n_params) * (hi - lo)
    return MorrisDesign(points=points, changed_param=changed_param, n_trajectories=n_trajectories, n_params=n_params)


def morris_indices(design: MorrisDesign, qoi) -> MorrisResult:
    """Compute elementary-effect screening statistics from QoI evaluations at a :class:`MorrisDesign`.

    ``qoi`` must be a 1-D array of ``design.n_evaluations`` scalar QoI values, one per row of
    ``design.points`` in that exact order. Within each trajectory, the elementary effect of the parameter
    that changes at step ``k`` is ``(qoi[k+1] - qoi[k]) / (points[k+1, j] - points[k, j])`` -- the QoI change
    divided by the *signed* parameter change, which automatically reduces to the standard Morris (1991)
    definition regardless of whether that trajectory happened to step ``+delta`` or ``-delta`` in physical
    (bounds-scaled) units. ``mu``/``mu_star``/``sigma`` are then the across-trajectory mean, mean-absolute,
    and standard deviation (``ddof=1``, ``0`` when only one trajectory is given) of each parameter's
    elementary effects.
    """
    qoi = np.asarray(qoi, dtype=float)
    if qoi.ndim != 1:
        raise ValueError(f"qoi must be a 1-D array of scalar QoI evaluations; got shape {qoi.shape}.")
    expected = design.n_evaluations
    if qoi.shape[0] != expected:
        raise ValueError(
            f"qoi must have exactly design.n_evaluations={expected} entries (n_trajectories="
            f"{design.n_trajectories} * (n_params={design.n_params} + 1)), one per row of design.points in "
            f"order; got {qoi.shape[0]}."
        )

    r, d = design.n_trajectories, design.n_params
    y = qoi.reshape(r, d + 1)
    points = design.points.reshape(r, d + 1, d)

    effects = np.full((r, d), np.nan)
    for t in range(r):
        for k in range(d):
            j = design.changed_param[t, k]
            step = points[t, k + 1, j] - points[t, k, j]
            effects[t, j] = (y[t, k + 1] - y[t, k]) / step

    mu = np.mean(effects, axis=0)
    mu_star = np.mean(np.abs(effects), axis=0)
    sigma = np.std(effects, axis=0, ddof=1) if r > 1 else np.zeros(d)

    return MorrisResult(mu=mu, mu_star=mu_star, sigma=sigma, n_trajectories=r)
