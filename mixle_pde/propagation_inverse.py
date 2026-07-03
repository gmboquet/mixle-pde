"""Convenience builders that pose the sonar/radar propagation INVERSE problems on the PE forward.

The hard machinery already exists: :class:`mixle_pde.parabolic_equation.ParabolicEquation2D` is a
differentiable one-way propagator, and :func:`mixle_pde.inverse.Differential` is an observation whose
forward model is any ``forward(p, ops) -> solution``. The two classic remote-sensing inverse problems in
this stack are therefore ordinary ``Differential`` instances with ZERO new inverse machinery:

* **Refractivity-from-clutter (RFC).** A trapping modified-refractivity duct guides a radar field along the
  surface far past the horizon; the duct's geometry is written into the range/height structure of the
  received (clutter) field. Parameterize ``M(z)`` by a latent DUCT HEIGHT ``h_d`` (and optionally the
  trapping strength), propagate with the PE, observe the field, and fit to recover the duct.

* **Ocean/geoacoustic inversion.** A sound-speed perturbation (a warm anomaly, a mixed-layer offset) or a
  seabed sound speed reshapes the received acoustic field; parameterize the SSP by a latent scalar offset,
  propagate with the PE (index ``n = c0/c``), observe the field, and fit to recover it.

Both builders are thin wrappers: they close a differentiable M/SSP profile over the free driver, build the
PE forward, and return the ``(field, proxy)`` pair consumed by :func:`mixle_pde.inverse.joint`::

    from mixle.ppl import free, joint
    h_d = free(1, name="h_d", support="real")  # metre-scale steps; a positive/log reparam can overshoot
    obs = refractivity_from_clutter(y_obs, h_d, pe=pe, source_depth=20.0, m0=350.0, base_gradient=0.118)
    post = joint([obs]).fit(how="gauss_newton")   # not 'laplace': complex/sparse forward -> first order
    h_mean, h_sd = post.posterior("h_d")

The received field is complex (the ``Differential`` proxy scores a complex Gaussian misfit on it), so fit
with ``how='map'``, ``'gauss_newton'``, or ``'vi'``; ``'laplace'`` is unavailable for this forward.
"""

from __future__ import annotations

from typing import Any

from mixle_pde.inverse import Differential
from mixle_pde.parabolic_equation import ParabolicEquation2D, modified_refractivity_index

__all__ = ["refractivity_from_clutter", "ocean_sound_speed_inversion"]


def _soft_surface_duct(z, h_d, m0, base_gradient, strength, ops):
    """A differentiable surface-duct modified-refractivity profile ``M(z)`` from a duct height ``h_d``.

    Below the duct height the profile falls with a trapping (negative) gradient; above it, it rises at the
    standard-atmosphere gradient. The corner at ``z = h_d`` is smoothed with ``ops.heaviside`` so the whole
    map is differentiable in ``h_d`` (a hard ``where`` would give a zero/undefined gradient at the corner):

        M(z) = m0 + s(z) * [ (m0 - strength * z) - m0 ]   below,   rising branch above,

    written as a smooth blend ``M = m0 - strength * z * (1 - H) + [M_top + base_gradient (z - h_d)] * H``
    with ``H = heaviside(z - h_d)`` and ``M_top = m0 - strength * h_d`` the M-value at the duct top. The
    trapping slope ``-strength`` (M-units/m) is negative-gradient (ducting); ``base_gradient`` (~0.118) is the
    positive standard slope above the duct. ``ops`` supplies the differentiable ``heaviside``.
    """
    m_top = m0 - strength * h_d
    below = m0 - strength * z
    above = m_top + base_gradient * (z - h_d)
    gate = ops.heaviside(z - h_d, eps=max(1.0, float(strength)))
    return below * (1.0 - gate) + above * gate


def refractivity_from_clutter(
    y: Any,
    duct_height,
    *,
    pe: ParabolicEquation2D,
    source_depth: float,
    m0: float = 350.0,
    base_gradient: float = 0.118,
    strength: float = 1.0,
    starter_width: float | None = None,
    n_range: int | None = None,
    observe=None,
    scale: Any = 1.0,
    family: str = "gaussian",
) -> tuple:
    """Refractivity-from-clutter: recover a radar surface-duct height from the propagated (clutter) field.

    Builds a :func:`mixle_pde.inverse.Differential` whose PE forward propagates the field through a
    modified-refractivity profile parameterized by the latent ``duct_height`` free driver, then observes the
    received field so that fitting recovers the duct height. The profile is the smooth surface duct of
    :func:`_soft_surface_duct` (trapping below ``h_d``, standard gradient above), differentiable in ``h_d``.

    Parameters
    ----------
    y : array
        Observed (complex) received field. Either the full ``(n_range, nz)`` field or, with a custom
        ``observe``, whatever that returns (e.g. the last-range depth column, a surface strip).
    duct_height : free handle
        ``free(1, name="h_d", support="real")`` -- the latent duct height (m) to recover. Use ``"real"``
        (metre-scale Gauss-Newton steps) rather than a positive/log reparam, which can overshoot.
    pe : ParabolicEquation2D
        A configured propagator (radar: ``surface="free"``, ``k0`` from the carrier). Reused every solve.
    source_depth : float
        Antenna height (m) for the PE self-starter.
    m0 : float
        Surface modified-refractivity level (M-units) the profile starts from.
    base_gradient : float
        Standard-atmosphere M-gradient above the duct (~0.118 M-units/m).
    strength : float
        Trapping slope magnitude below the duct (M-units/m); the sub-duct gradient is ``-strength`` (< 0).
    starter_width, n_range :
        PE starter width and number of range steps (defaults: ``pe`` starter default, full ``y`` length).
    observe : callable, optional
        ``observe(field, p, ops) -> predicted`` mapping the marched ``(n_range, nz)`` complex field to the
        observed quantity. Default: the whole field.
    scale, family :
        Passed through to :func:`Differential` (noise level, ``'gaussian'``).

    Returns the ``(field, proxy)`` pair for :func:`joint`. Multi-parameter extension: add a second free
    driver for ``strength`` (or ``m0``) and reference ``p.strength`` in a wrapping forward; the machinery is
    unchanged.
    """
    z = pe.depths()
    psi0 = pe.starter(float(source_depth), width=starter_width)
    nr = int(n_range) if n_range is not None else None

    def forward(p, ops):
        m_profile = _soft_surface_duct(z, p.h_d, float(m0), float(base_gradient), float(strength), ops)
        n_col = modified_refractivity_index(m_profile)
        return pe.march(psi0, n_col, nr)

    return Differential(
        y,
        drivers=[duct_height],
        forward=forward,
        observe=observe,
        scale=scale,
        family=family,
    )


def ocean_sound_speed_inversion(
    y: Any,
    sound_speed_offset,
    *,
    pe: ParabolicEquation2D,
    source_depth: float,
    c_profile,
    anomaly_depth: float | None = None,
    anomaly_width: float = 30.0,
    anomaly_shape=None,
    starter_width: float | None = None,
    n_range: int | None = None,
    observe=None,
    scale: Any = 1.0,
    family: str = "gaussian",
) -> tuple:
    """Ocean tomography: recover a scalar sound-speed anomaly amplitude (m/s) from the received field.

    Builds a :func:`mixle_pde.inverse.Differential` whose PE forward propagates through the sound-speed
    profile ``c(z) = c_profile + dc * shape(z)`` (index ``n = c0 / c``), where ``dc`` is the latent
    ``sound_speed_offset`` free driver and ``shape(z)`` is a depth-localized anomaly (a warm-water lens, a
    mixed-layer perturbation). Observing the received field, fitting recovers the anomaly amplitude ``dc``.
    The anomaly is depth-localized on purpose: a spatially UNIFORM sound-speed offset changes the index by a
    constant, which the split-step env phase applies as a global (depth-independent) phase that cancels in
    the field magnitude, so a uniform offset is unobservable in ``|field|``. A localized anomaly reshapes the
    field and is recoverable. The same pattern with a seabed-keyed shape recovers a geoacoustic sound speed.

    Parameters
    ----------
    y : array
        Observed (complex) received field (full field or whatever ``observe`` returns).
    sound_speed_offset : free handle
        ``free(1, name="dc", support="real")`` -- the latent anomaly amplitude (m/s) to recover.
    pe : ParabolicEquation2D
        A configured propagator (index reference ``pe.c0``). Reused every solve.
    source_depth : float
        Source depth (m) for the PE self-starter.
    c_profile : array or tensor
        Background sound-speed column ``c0(z)`` (length ``nz``, m/s).
    anomaly_depth : float, optional
        Centre depth (m) of the default Gaussian anomaly (default: mid water column).
    anomaly_width : float
        Half-width (m) of the default Gaussian anomaly ``exp(-((z - z0)/w)^2)``.
    anomaly_shape : array or tensor, optional
        A custom (length ``nz``) anomaly profile; overrides the Gaussian. Use a seabed mask for geoacoustic
        inversion, a mixed-layer step for a thermocline, etc.
    starter_width, n_range, observe, scale, family :
        As in :func:`refractivity_from_clutter`.

    Returns the ``(field, proxy)`` pair for :func:`joint`. Multi-parameter extension: add a second free
    driver for the anomaly depth (or a seabed sound speed on a depth mask) and reference it in the forward;
    the inverse machinery is unchanged.
    """
    import torch

    c0 = float(pe.c0)
    c_bg = torch.as_tensor(c_profile, dtype=torch.float64)
    if anomaly_shape is not None:
        shape = torch.as_tensor(anomaly_shape, dtype=torch.float64)
    else:
        z = pe.depths()
        z0 = float(anomaly_depth) if anomaly_depth is not None else float((pe.nz - 1) * pe.dz / 2.0)
        shape = torch.exp(-(((z - z0) / float(anomaly_width)) ** 2))
    psi0 = pe.starter(float(source_depth), width=starter_width)
    nr = int(n_range) if n_range is not None else None

    def forward(p, ops):
        c = c_bg + p.dc * shape
        n_col = c0 / c
        return pe.march(psi0, n_col, nr)

    return Differential(
        y,
        drivers=[sound_speed_offset],
        forward=forward,
        observe=observe,
        scale=scale,
        family=family,
    )
