"""Structural/probabilistic reliability analysis: FORM (Hasofer-Lind-Rackwitz-Fiessler) with an
importance-sampling honesty check.

Reliability analysis asks a different question from :mod:`mixle_pde.decision_quantities` and
:mod:`mixle_pde.global_sensitivity`, the two sibling members of work-plan MP-I11 ("Forward UQ, global
sensitivity, reliability"). ``decision_quantities.prob_exceed`` gives the crude-Monte-Carlo *fraction of a
spatial region* exceeding a threshold under a posterior; ``global_sensitivity`` decomposes a QoI's output
*variance* by input parameter. Neither is built to answer "what is the probability that this scalar
limit-state condition is violated?" efficiently when that probability is small (a rare event) -- crude
Monte Carlo needs ``O(1/p)`` samples to pin down a probability ``p`` to a fixed relative precision, which
is exactly why classical structural reliability analysis (Hasofer & Lind 1974; Rackwitz & Fiessler 1978)
exists as its own family of methods.

Given a scalar *limit-state function* ``g`` of ``dim`` independent standard-normal inputs, with the
convention that the **failure domain is ``{u : g(u) <= 0}``** and the origin is safe (``g(0) > 0``), two
complementary estimators are implemented:

* :func:`form` -- the First-Order Reliability Method via Hasofer-Lind-Rackwitz-Fiessler (HLRF) iteration.
  It finds the *design point* (a.k.a. most-probable point, MPP): the point on the limit-state surface
  closest to the origin in standard-normal space. Its distance from the origin, ``beta`` (the
  "reliability index"), converts to a failure-probability approximation ``Phi(-beta)`` by treating the
  limit-state surface as if it were the *tangent hyperplane* at the design point. This costs only
  ``O(dim)`` limit-state evaluations per iteration (via finite-difference or supplied gradients) --
  orders of magnitude cheaper than Monte Carlo for a small probability -- but the linearization is only
  exact when ``g`` truly is affine; for a curved limit-state surface ``Phi(-beta)`` carries a genuine,
  sometimes large, systematic bias (see the module's test suite for a worked example where FORM is
  off by roughly 8x on a purely quadratic, radially symmetric limit state -- not a corner case, a textbook
  one).
* :func:`importance_sampling_probability` -- a variance-reduced Monte Carlo estimator that shifts the
  sampling density's mean to a supplied point (canonically :attr:`FORMResult.design_point`, where failure
  probability mass concentrates) and re-weights. Unlike FORM it has **no linearization bias**: it is a
  consistent estimator of the true probability regardless of how curved ``g`` is, at the cost of actual
  sampling (its error is purely statistical and shrinks as the sample count grows, reported honestly via
  :attr:`ImportanceSamplingResult.standard_error` and ``coefficient_of_variation`` -- never claim a
  probability is trustworthy without checking these).

The intended workflow composes both: run :func:`form` first (cheap, gives a design point even when the
true probability is far too small for crude Monte Carlo to see in a reasonable sample budget), then run
:func:`importance_sampling_probability` centered at that design point to get an honest, near-bias-free
probability estimate with a reported uncertainty -- close agreement between the two is real evidence the
limit-state surface is nearly linear near the design point; a large gap is real evidence it is not, and
only the importance-sampling number should then be trusted.

Both estimators are solver-agnostic in the same sense as :mod:`mixle_pde.global_sensitivity` and
:mod:`mixle_pde.reduced_basis`: this module never imports or calls a PDE/ODE solver or
:mod:`mixle_pde.decision_quantities` itself. The caller's ``g`` can wrap anything that reduces to one
scalar limit-state value per standard-normal input point -- an analytic function, a PDE forward solve, or
a ``decision_quantities`` derived quantity evaluated at a single posterior draw (e.g.
``g(u) = threshold - quantity_of_interest(physical_point(u))``, using :func:`standard_normal_to_physical`
for the ``physical_point`` map).

:func:`standard_normal_to_physical` / :func:`physical_to_standard_normal` provide the isoprobabilistic
transform (per-marginal inverse-CDF / CDF, i.e. a Rosenblatt transform under independence) between physical
parameter space and the standard-normal space :func:`form` and :func:`importance_sampling_probability`
both operate in -- the same "supply each marginal's inverse CDF" idiom
:mod:`mixle_pde.global_sensitivity`'s module docstring already documents for its own (uniform-space)
design functions.

Scope note: FORM here is first-order only (no SORM curvature correction); the design-point search assumes
``g`` is differentiable and uses finite differences by default (an analytic ``gradient`` callable may be
supplied when available, e.g. from :mod:`mixle_pde.adjoint`, but wiring that is left to the caller -- this
module accepts any callable of the right shape). Both estimators assume independent standard-normal (or,
via the transform helpers, independent-marginal) inputs; correlated inputs need a decorrelating transform
(e.g. Nataf) applied by the caller before calling in, exactly analogous to the independence assumption
:mod:`mixle_pde.global_sensitivity` already documents for Sobol'/Morris.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import stats

__all__ = [
    "FORMResult",
    "ImportanceSamplingResult",
    "form",
    "importance_sampling_probability",
    "standard_normal_to_physical",
    "physical_to_standard_normal",
]


# ---------------------------------------------------------------------------
# Isoprobabilistic transform: standard-normal space <-> physical space
# ---------------------------------------------------------------------------


def standard_normal_to_physical(u: np.ndarray, marginals: Sequence[Any]) -> np.ndarray:
    """Map standard-normal-space points to physical space via each parameter's marginal ``.ppf``.

    ``u`` has shape ``(..., dim)``; ``marginals`` is a length-``dim`` sequence of independent marginal
    distributions, each exposing ``.ppf`` (inverse CDF), e.g. frozen ``scipy.stats`` distributions. This is
    the standard isoprobabilistic (Rosenblatt-under-independence) transform: ``x_i = F_i^{-1}(Phi(u_i))``.
    Use this to wrap a physical-space limit-state function for :func:`form` /
    :func:`importance_sampling_probability`, both of which operate in standard-normal space. Correlated
    marginals need a decorrelating transform (e.g. Nataf) applied first -- out of scope here.
    """
    u = np.asarray(u, dtype=float)
    dim = len(marginals)
    if dim < 1:
        raise ValueError("marginals must contain at least one distribution.")
    if u.shape[-1] != dim:
        raise ValueError(f"u's last axis must have length len(marginals)={dim}; got shape {u.shape}.")
    p = stats.norm.cdf(u)
    columns = [marginals[i].ppf(p[..., i]) for i in range(dim)]
    return np.stack(columns, axis=-1)


def physical_to_standard_normal(x: np.ndarray, marginals: Sequence[Any]) -> np.ndarray:
    """Inverse of :func:`standard_normal_to_physical`: physical-space points to standard-normal space.

    ``x`` has shape ``(..., dim)``; ``marginals`` is a length-``dim`` sequence of independent marginal
    distributions, each exposing ``.cdf``. Computes ``u_i = Phi^{-1}(F_i(x_i))``.
    """
    x = np.asarray(x, dtype=float)
    dim = len(marginals)
    if dim < 1:
        raise ValueError("marginals must contain at least one distribution.")
    if x.shape[-1] != dim:
        raise ValueError(f"x's last axis must have length len(marginals)={dim}; got shape {x.shape}.")
    p = np.stack([marginals[i].cdf(x[..., i]) for i in range(dim)], axis=-1)
    return stats.norm.ppf(p)


# ---------------------------------------------------------------------------
# FORM: Hasofer-Lind-Rackwitz-Fiessler design-point search
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FORMResult:
    """First-Order Reliability Method result from :func:`form`.

    ``beta`` is the reliability index (Euclidean distance from the origin to the design point in
    standard-normal space); ``probability`` is FORM's tangent-hyperplane failure-probability approximation
    ``Phi(-beta)`` -- a point estimate that is exact only when the limit-state surface is affine, and can
    carry substantial bias otherwise (see the module docstring). ``design_point`` (shape ``(dim,)``) is the
    most-probable point (MPP) the HLRF iteration converged to. ``alpha`` (shape ``(dim,)``) is the unit
    vector from the origin to the design point, defined directly as ``design_point / beta`` (so
    ``design_point == beta * alpha`` by construction, avoiding any ambiguity about the *sign* of a
    gradient-derived direction); ``alpha[i] ** 2`` is conventionally read as parameter ``i``'s "importance
    factor" in the FORM approximation, and the entries sum to 1. When ``beta`` is (numerically) zero --
    the origin itself lies on the limit-state surface, a degenerate zero-margin case -- ``alpha`` is
    reported as all-``nan`` rather than a fabricated direction. ``converged`` is only ``True`` when the
    HLRF step size actually fell below ``tol`` within ``max_iter`` iterations; a non-converged result
    still reports its last iterate honestly rather than silently claiming success.
    """

    beta: float
    probability: float
    design_point: np.ndarray
    alpha: np.ndarray
    converged: bool
    n_iterations: int
    n_evaluations: int

    def __post_init__(self) -> None:
        if self.design_point.shape != self.alpha.shape:
            raise ValueError(
                f"design_point and alpha must have matching shape; got {self.design_point.shape} vs {self.alpha.shape}."
            )
        if self.beta < 0.0:
            raise ValueError(f"beta must be non-negative; got {self.beta!r}.")
        if not (0.0 <= self.probability <= 1.0):
            raise ValueError(f"probability must be in [0, 1]; got {self.probability!r}.")


def form(
    g: Callable[[np.ndarray], float],
    dim: int,
    *,
    gradient: Callable[[np.ndarray], np.ndarray] | None = None,
    u0: np.ndarray | None = None,
    max_iter: int = 100,
    tol: float = 1.0e-6,
    finite_diff_step: float = 1.0e-6,
) -> FORMResult:
    """Find the FORM design point and reliability index via Hasofer-Lind-Rackwitz-Fiessler (HLRF) iteration.

    ``g`` maps one standard-normal point (shape ``(dim,)``) to a scalar limit-state value; the failure
    domain is ``{u : g(u) <= 0}`` and ``g(u0)`` (the starting point, the origin by default) must be
    strictly positive -- FORM searches *from* the safe domain *for* the nearest boundary crossing, so a
    caller whose limit state is phrased the other way round should pass ``-g`` instead. Unlike
    :mod:`mixle_pde.global_sensitivity`'s design functions, ``g`` is evaluated one point at a time rather
    than at a precomputed ensemble: HLRF is inherently sequential (each iterate depends on the gradient at
    the previous one), so there is no static batch of points to hand the caller up front.

    Each iteration takes a Newton-like step to the point closest to the origin on the *linearization* of
    ``g`` at the current iterate (the classic Rackwitz-Fiessler recursion
    ``u_{k+1} = alpha_k * (alpha_k . u_k - g(u_k) / ||grad g(u_k)||)`` with ``alpha_k = grad
    g(u_k)/||grad g(u_k)||``), and stops when the step size falls below ``tol``. If ``gradient`` is not
    supplied, a central finite difference with step ``finite_diff_step`` is used (costing ``2 * dim``
    extra evaluations per iteration); supply an analytic ``gradient`` (e.g. from
    :mod:`mixle_pde.adjoint`, wrapped to standard-normal-space via the chain rule through
    :func:`standard_normal_to_physical`) when ``g`` is expensive.

    Raises ``ValueError`` if ``g(u0) <= 0`` (the starting point is not on the safe side) or if the gradient
    vanishes or is non-finite at any iterate (FORM cannot take a step from a stationary point).
    """
    dim = int(dim)
    if dim < 1:
        raise ValueError(f"dim must be a positive integer; got {dim!r}.")
    if u0 is None:
        u_k = np.zeros(dim)
    else:
        u_k = np.array(u0, dtype=float)
        if u_k.shape != (dim,):
            raise ValueError(f"u0 must have shape (dim,) = ({dim},); got {u_k.shape}.")
    max_iter = int(max_iter)
    if max_iter < 1:
        raise ValueError(f"max_iter must be a positive integer; got {max_iter!r}.")
    if not tol > 0.0:
        raise ValueError(f"tol must be positive; got {tol!r}.")
    finite_diff_step = float(finite_diff_step)
    if not finite_diff_step > 0.0:
        raise ValueError(f"finite_diff_step must be positive; got {finite_diff_step!r}.")

    def _gradient_at(u: np.ndarray) -> tuple[np.ndarray, int]:
        if gradient is not None:
            grad = np.asarray(gradient(u), dtype=float)
            if grad.shape != (dim,):
                raise ValueError(f"gradient(u) must return shape (dim,) = ({dim},); got {grad.shape}.")
            return grad, 0
        grad = np.empty(dim)
        for i in range(dim):
            step = np.zeros(dim)
            step[i] = finite_diff_step
            grad[i] = (float(g(u + step)) - float(g(u - step))) / (2.0 * finite_diff_step)
        return grad, 2 * dim

    n_evaluations = 0
    g_k = float(g(u_k))
    n_evaluations += 1
    if g_k <= 0.0:
        raise ValueError(
            "form() requires g(u0) > 0 (the starting point must be in the safe domain, where the failure "
            f"domain is g(u) <= 0); got g(u0)={g_k!r}. Negate g if your convention is the other way round."
        )

    converged = False
    n_iterations = 0
    for n_iterations in range(1, max_iter + 1):
        grad_k, n_grad_evals = _gradient_at(u_k)
        n_evaluations += n_grad_evals
        norm_grad = float(np.linalg.norm(grad_k))
        if not np.isfinite(norm_grad) or norm_grad == 0.0:
            raise ValueError(
                f"gradient vanished or is non-finite at iteration {n_iterations} (u={u_k.tolist()}, "
                f"g(u)={g_k!r}); FORM cannot proceed from a stationary point."
            )
        direction = grad_k / norm_grad
        u_next = direction * (float(np.dot(direction, u_k)) - g_k / norm_grad)
        step_norm = float(np.linalg.norm(u_next - u_k))
        u_k = u_next
        g_k = float(g(u_k))
        n_evaluations += 1
        if step_norm < tol:
            converged = True
            break

    beta = float(np.linalg.norm(u_k))
    alpha = u_k / beta if beta > 1.0e-12 else np.full(dim, np.nan)
    probability = float(stats.norm.cdf(-beta))

    return FORMResult(
        beta=beta,
        probability=probability,
        design_point=u_k,
        alpha=alpha,
        converged=converged,
        n_iterations=n_iterations,
        n_evaluations=n_evaluations,
    )


# ---------------------------------------------------------------------------
# Importance-sampling probability estimate (the honesty cross-check)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ImportanceSamplingResult:
    """Importance-sampling failure-probability estimate from :func:`importance_sampling_probability`.

    ``probability`` is the (unbiased) importance-sampling estimate of ``P(g(U) <= 0)``. ``standard_error``
    is that estimate's Monte Carlo standard error (shrinks as ``1/sqrt(n_samples)``); never treat
    ``probability`` as trustworthy without checking it. ``coefficient_of_variation`` is
    ``standard_error / probability`` (``inf`` when the estimate is exactly zero -- no failure samples were
    drawn at all) -- the field-standard "is this probability estimate trustworthy" diagnostic in
    reliability engineering; values above roughly 0.1-0.3 conventionally mean "increase n_samples or
    reconsider the shift point" rather than "report this number". ``n_effective`` is Kish's effective
    sample size (``(sum w)^2 / sum(w^2)`` over the raw importance weights, before the failure indicator is
    applied) -- a general diagnostic of how degenerate the importance weights are, independent of the
    specific limit state; ``n_effective`` much smaller than ``n_samples`` means a handful of samples are
    dominating the estimate.
    """

    probability: float
    standard_error: float
    coefficient_of_variation: float
    n_samples: int
    n_effective: float

    def __post_init__(self) -> None:
        if not (0.0 <= self.probability <= 1.0):
            raise ValueError(f"probability must be in [0, 1]; got {self.probability!r}.")
        if self.standard_error < 0.0:
            raise ValueError(f"standard_error must be non-negative; got {self.standard_error!r}.")
        if self.n_samples < 1:
            raise ValueError(f"n_samples must be a positive integer; got {self.n_samples!r}.")
        if self.n_effective < 0.0:
            raise ValueError(f"n_effective must be non-negative; got {self.n_effective!r}.")


def importance_sampling_probability(
    g: Callable[[np.ndarray], np.ndarray],
    dim: int,
    *,
    shift: np.ndarray,
    n_samples: int,
    rng: np.random.Generator | None = None,
) -> ImportanceSamplingResult:
    """Variance-reduced Monte Carlo estimate of ``P(g(U) <= 0)`` via a mean-shifted normal sampler.

    Draws ``n_samples`` points from ``N(shift, I)`` (standard-normal space, mean shifted to ``shift`` --
    canonically :attr:`FORMResult.design_point`, where failure-region probability mass concentrates) and
    re-weights by the exact likelihood ratio to the nominal ``N(0, I)`` density,
    ``w(u) = exp(-u . shift + 0.5 * ||shift||^2)``, so the weighted estimate is unbiased for the *nominal*
    probability regardless of how well ``shift`` was chosen -- a poor shift only costs precision (a large
    ``standard_error`` / ``coefficient_of_variation``, both reported honestly), never a biased answer.

    Unlike :func:`form`, ``g`` is called once on the whole ``(n_samples, dim)`` batch and must return
    shape ``(n_samples,)`` -- the same batched-QoI convention :mod:`mixle_pde.global_sensitivity` uses,
    appropriate here because (unlike HLRF) the sample points do not depend on each other.
    """
    dim = int(dim)
    if dim < 1:
        raise ValueError(f"dim must be a positive integer; got {dim!r}.")
    shift = np.asarray(shift, dtype=float)
    if shift.shape != (dim,):
        raise ValueError(f"shift must have shape (dim,) = ({dim},); got {shift.shape}.")
    n_samples = int(n_samples)
    if n_samples < 2:
        raise ValueError(f"n_samples must be at least 2 (a variance estimate needs >= 2 draws); got {n_samples!r}.")
    rng = rng if rng is not None else np.random.default_rng()

    samples = rng.normal(size=(n_samples, dim)) + shift[None, :]
    g_vals = np.asarray(g(samples), dtype=float)
    if g_vals.shape != (n_samples,):
        raise ValueError(f"g(samples) must return shape (n_samples,) = ({n_samples},); got {g_vals.shape}.")
    indicator = (g_vals <= 0.0).astype(float)

    log_weight = -(samples @ shift) + 0.5 * float(np.dot(shift, shift))
    weight = np.exp(log_weight)

    contributions = indicator * weight
    probability = float(np.mean(contributions))
    variance = float(np.var(contributions, ddof=1))
    standard_error = float(np.sqrt(max(variance, 0.0) / n_samples))
    coefficient_of_variation = standard_error / probability if probability > 0.0 else float("inf")

    weight_sq_sum = float(np.sum(weight**2))
    n_effective = float(np.sum(weight) ** 2 / weight_sq_sum) if weight_sq_sum > 0.0 else 0.0

    return ImportanceSamplingResult(
        probability=probability,
        standard_error=standard_error,
        coefficient_of_variation=coefficient_of_variation,
        n_samples=n_samples,
        n_effective=n_effective,
    )
