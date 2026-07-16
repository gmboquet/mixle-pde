"""MP-N7 -- surrogate-accelerated inversion: a hybrid full-order/surrogate wrapper around Gauss-Newton.

:mod:`mixle_pde.surrogate` distills a cheap, calibrated stand-in for an expensive forward and exposes
two questions per input: ``predict(x)`` (the fast estimate) and ``defer(x)`` (the honest gate saying
whether that estimate should be trusted or the real, expensive model should run instead). Every
existing wiring of that gate in this package (``mixle_pde.simulation_service.register_surrogate``, the
composite-heat reference study's ``evaluate_with_surrogate``) makes a SINGLE cheap-vs-real decision per
forward call. Neither wraps an ITERATIVE inversion, and neither returns a per-iteration record --
``simulation_service`` tracks nothing across calls, and the reference study only tallies grid-point
counts for a closed-form quadrature posterior, not Newton iterations.

This module closes that gap for :func:`mixle_pde.field_gauss_newton.gauss_newton_invert`: an outer loop
over that same MAP objective (fixed prior mean/precision, fixed observations -- only the evaluator
changes) where each iteration either spends one cheap :meth:`~mixle_pde.surrogate.Surrogate.predict`
call or runs the full-order kernel. The gate itself is never rebuilt here -- :meth:`Surrogate.defer`
(the calibrated OOD/precision-floor check from ``distill_forward``) decides every fallback, exactly as
elsewhere in the package. Two rules force the expensive path regardless of what ``defer`` says: no
surrogate at all, and the reserved final iterations before the loop can report convergence, so a caller
can never receive a "converged" posterior that was never checked against the real forward. The returned
:class:`HybridInversionReport` names, iteration by iteration, which evaluator produced it -- the hybrid
claim is a fact a caller can inspect, not an assertion in a docstring.

``gauss_newton_invert`` itself gained one small, backward-compatible addition to make this possible:
an optional ``u_init`` warm start (default ``None`` reproduces its exact prior behaviour). Without it,
every call restarts Newton from the prior mean, so repeated calls could never make combined progress --
``u_init`` is what lets this module thread ONE continuous trajectory through however many outer
iterations it takes, full-order calls included, without touching the stated Bayesian problem (the prior
term is still evaluated against the true, fixed prior mean; only the search's starting point moves).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from mixle_pde.field_gauss_newton import GaussNewtonReport, gauss_newton_invert
from mixle_pde.field_inversion import FieldGaussianPrior
from mixle_pde.latent import Field3D, PosteriorField3D
from mixle_pde.observations import ForwardOperatorRegistry, Observation
from mixle_pde.surrogate import Surrogate

__all__ = ["HybridInversionReport", "HybridIterationRecord", "hybrid_gauss_newton_invert"]


@dataclass(frozen=True)
class HybridIterationRecord:
    """One outer iteration of :func:`hybrid_gauss_newton_invert`: what evaluated it, and why.

    ``used_surrogate=True`` means this iteration spent exactly one cheap
    :meth:`~mixle_pde.surrogate.Surrogate.predict` call and never touched ``registry`` or the real
    forward/Jacobian. ``used_surrogate=False`` means it ran the full-order kernel; ``inner_report`` is
    that call's own :class:`~mixle_pde.field_gauss_newton.GaussNewtonReport`.

    ``reason`` names exactly which rule selected this iteration's evaluator:

    - ``"surrogate_trusted"`` -- a surrogate was available, ``defer`` returned ``False``, and this
      iteration was not in the reserved full-order tail.
    - ``"surrogate_deferred"`` -- the calibrated OOD/precision-floor gate fired.
    - ``"final_iterations_reserved"`` -- inside the last ``n_final_full_order`` scheduled iterations,
      which are always full-order regardless of what ``defer`` says.
    - ``"surrogate_unavailable"`` -- no surrogate was given at all.
    - ``"mandatory_verification"`` -- a safety-net call appended after the scheduled loop, only reached
      if every scheduled iteration somehow used the surrogate (should not happen when
      ``n_final_full_order >= 1``, but the return value never depends on that being bug-free).
    """

    index: int
    used_surrogate: bool
    reason: str
    step_norm: float
    inner_report: GaussNewtonReport | None


@dataclass(frozen=True)
class HybridInversionReport:
    """Auditable summary of a surrogate-accelerated inversion: exactly which iterations were cheap.

    ``verified_against_full_order`` is always ``True`` on a normal return: every code path in
    :func:`hybrid_gauss_newton_invert` ends with the last entry of ``iterations`` being a full-order
    one, so the returned posterior was, by construction, produced or confirmed by the real kernel.
    ``final_report`` is that last full-order call's own report, so ``converged`` here carries exactly
    the same meaning it does for a single :func:`~mixle_pde.field_gauss_newton.gauss_newton_invert`
    call -- it is not a fabricated summary flag.
    """

    iterations: tuple[HybridIterationRecord, ...]
    converged: bool
    verified_against_full_order: bool
    n_surrogate_iterations: int
    n_full_order_iterations: int
    final_report: GaussNewtonReport


def hybrid_gauss_newton_invert(
    grid: Field3D,
    observations: list[Observation],
    registry: ForwardOperatorRegistry,
    prior: FieldGaussianPrior,
    surrogate: Surrogate | None = None,
    *,
    max_iterations: int = 6,
    n_final_full_order: int = 2,
    inner_max_iter: int = 25,
    tol: float = 1e-5,
    jitter: float = 1.0e-10,
    factor_cache: dict | None = None,
    invert: Callable[..., tuple[PosteriorField3D, GaussNewtonReport]] = gauss_newton_invert,
) -> tuple[PosteriorField3D, HybridInversionReport]:
    """Surrogate-accelerated Gauss-Newton: cheap ``surrogate.predict`` steps, verified by the real kernel.

    Runs up to ``max_iterations`` outer iterations of the SAME MAP objective ``invert`` solves (fixed
    ``prior``/``observations`` throughout -- only the evaluator per iteration changes). Each iteration
    is either:

    - a cheap step: ``u_next = surrogate.predict(u)``, taken exactly when ``surrogate`` is available,
      not inside the reserved full-order tail, and ``surrogate.defer(u)`` is ``False`` (the surrogate's
      own calibrated gate says this input is trustworthy); or
    - a full-order step: one warm-started call to ``invert(grid, observations, registry, prior,
      max_iter=inner_max_iter, ..., u_init=u)``, taken whenever the surrogate is unavailable, defers, or
      this iteration falls in the last ``n_final_full_order`` scheduled iterations.

    The loop exits early once a full-order call reports convergence (``inner_report.converged``); it is
    never allowed to exit on a surrogate iteration, and a mandatory safety-net full-order call is
    appended if, somehow, none of the scheduled iterations ran full-order -- so the returned posterior
    is always the literal output of ``invert``, never the surrogate's own guess.

    ``surrogate``, if given, must have been distilled (e.g. via
    :func:`mixle_pde.surrogate.distill_forward`) against a teacher returning a ``(grid.n,)`` vector in
    the SAME unconstrained parameter space ``u`` that ``prior``/``invert`` operate in -- a mismatched
    shape raises ``ValueError`` rather than silently truncating or broadcasting. ``invert`` defaults to
    :func:`~mixle_pde.field_gauss_newton.gauss_newton_invert` and may be swapped for any routine sharing
    its ``(grid, observations, registry, prior, *, max_iter, tol, jitter, factor_cache, u_init) ->
    (PosteriorField3D, GaussNewtonReport)`` contract.

    Returns the final posterior and a :class:`HybridInversionReport` naming which iterations were
    surrogate vs. full-order, so the hybrid claim is auditable from the return value, not asserted.
    """
    if max_iterations < 1:
        raise ValueError("max_iterations must be >= 1.")
    if n_final_full_order < 1:
        raise ValueError("n_final_full_order must be >= 1 -- at least the last iteration must be full-order.")

    u = prior.mean_vector(grid)
    records: list[HybridIterationRecord] = []
    final_posterior: PosteriorField3D | None = None
    final_report: GaussNewtonReport | None = None

    for index in range(max_iterations):
        in_final_stretch = index >= max_iterations - n_final_full_order
        if surrogate is None:
            use_surrogate, reason = False, "surrogate_unavailable"
        elif in_final_stretch:
            use_surrogate, reason = False, "final_iterations_reserved"
        elif surrogate.defer(u):
            use_surrogate, reason = False, "surrogate_deferred"
        else:
            use_surrogate, reason = True, "surrogate_trusted"

        if use_surrogate:
            proposed = np.asarray(surrogate.predict(u), dtype=float)
            if proposed.shape != u.shape:
                raise ValueError(
                    f"surrogate.predict(u) returned shape {proposed.shape}, expected {u.shape} (one "
                    "value per grid cell) -- distill the surrogate against a teacher matching this "
                    "field's unconstrained-parameter dimensionality."
                )
            step_norm = float(np.linalg.norm(proposed - u))
            u = proposed
            records.append(
                HybridIterationRecord(
                    index=index, used_surrogate=True, reason=reason, step_norm=step_norm, inner_report=None
                )
            )
            continue

        posterior, report = invert(
            grid,
            observations,
            registry,
            prior,
            max_iter=inner_max_iter,
            tol=tol,
            jitter=jitter,
            factor_cache=factor_cache,
            u_init=u,
        )
        step_norm = float(np.linalg.norm(posterior.mean - u))
        u = posterior.mean
        final_posterior, final_report = posterior, report
        records.append(
            HybridIterationRecord(
                index=index, used_surrogate=False, reason=reason, step_norm=step_norm, inner_report=report
            )
        )
        if report.converged:
            break

    if final_posterior is None or final_report is None:
        # Safety net: every reachable configuration above already reserves the last scheduled
        # iteration as full-order, so this should not trigger -- but the contract ("never return a
        # result that was not checked against the full-order model") must hold even if that reservation
        # logic ever regresses, so it is enforced here unconditionally rather than merely documented.
        posterior, report = invert(
            grid,
            observations,
            registry,
            prior,
            max_iter=inner_max_iter,
            tol=tol,
            jitter=jitter,
            factor_cache=factor_cache,
            u_init=u,
        )
        step_norm = float(np.linalg.norm(posterior.mean - u))
        final_posterior, final_report = posterior, report
        records.append(
            HybridIterationRecord(
                index=len(records),
                used_surrogate=False,
                reason="mandatory_verification",
                step_norm=step_norm,
                inner_report=report,
            )
        )

    n_surrogate_iterations = sum(1 for record in records if record.used_surrogate)
    n_full_order_iterations = len(records) - n_surrogate_iterations
    return final_posterior, HybridInversionReport(
        iterations=tuple(records),
        converged=final_report.converged,
        verified_against_full_order=True,
        n_surrogate_iterations=n_surrogate_iterations,
        n_full_order_iterations=n_full_order_iterations,
        final_report=final_report,
    )
