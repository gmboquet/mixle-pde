"""Reverse-time migration (RTM) imaging on the differentiable acoustic wave stepper.

RTM forms a subsurface reflectivity image by propagating the source wavefield ``S(x, t)`` forward through a
smooth background velocity, propagating the recorded (scattered) receiver data ``R(x, t)`` backward in time
through the same background, and cross-correlating the two at zero time lag,

    I(x) = sum_t S(x, t) R(x, t).

A reflector lights up where the down-going source wavefield and the up-going back-propagated receiver
wavefield coincide in space and time, which is exactly the reflection point. Everything runs on the
checkpointed :class:`~mixle_pde.wave.WaveEquation2D` stepper via ``ops.integrate_record``, so both
propagations are adjoint-checkpointed and differentiable: the RTM image is the adjoint (gradient) of the
Born modeling operator applied to the data, which is why the same machinery gives least-squares RTM.

The image is a grid ``(nz, nx)`` with axis 0 = depth ``z`` (down) and axis 1 = horizontal ``x``. An optional
Laplacian post-filter removes the low-wavenumber source/receiver backscatter artifact that otherwise smears
the image along the illumination path (standard RTM practice).

``model_data_3d``/``rtm_image_3d`` lift the same recipe onto :class:`~mixle_pde.wave3d.WaveEquation3D`,
giving a volumetric ``(n, n, n)`` reflectivity from a single 3-D shot -- axis 0 is still depth, axes 1-2 the
two horizontal directions. ``elastic_rtm_2d`` runs the analogous imaging condition on a 2-D isotropic
elastic (P-SV) stepper, cross-correlating both particle-velocity components (an energy/P-S imaging
condition) so mode-converted arrivals contribute too; :mod:`mixle_pde.elastic`/``elastic_aniso`` are 3-D
only today, so the 2-D stepper is a small local reduction of that same velocity-stress physics to the
``(z, x)`` plane.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

Array = Any  # a torch tensor or numpy array, kept backend-agnostic through ``ops``


def _as_np(x) -> np.ndarray:
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


def _laplacian_2d(image: np.ndarray) -> np.ndarray:
    """Five-point Laplacian of a 2-D image (zero at the border). The standard RTM de-noising filter: it
    kills the smooth low-wavenumber backscatter artifact while keeping the high-wavenumber reflector."""
    out = np.zeros_like(image)
    out[1:-1, 1:-1] = image[2:, 1:-1] + image[:-2, 1:-1] + image[1:-1, 2:] + image[1:-1, :-2] - 4.0 * image[1:-1, 1:-1]
    return out


def _source_step(wave, ops, c2, src_node: int, ampl):
    """One-step closure that injects a time-dependent point source ``ampl[i]`` at ``src_node``."""
    nn = wave.n * wave.n

    def step(state, i):
        s = ops.zeros(nn).clone()
        s[src_node] = float(ampl[i])
        return wave.step(state, c2, ops, source=s)

    return step


def _long(x):
    """A long-dtype index tensor for scatter assignment."""
    import torch

    return torch.as_tensor(np.asarray(x), dtype=torch.long)


def _data_step(wave, ops, c2, recv_nodes, data):
    """One-step closure that injects the recorded trace ``data[i]`` at every receiver node (adjoint source)."""
    nn = wave.n * wave.n
    recv = _long(recv_nodes)

    def step(state, i):
        s = ops.zeros(nn).clone()
        s[recv] = ops.tensor(np.asarray(data[i]))
        return wave.step(state, c2, ops, source=s)

    return step


def model_data(
    wave,
    c2,
    src_node: int,
    source_wavelet: Sequence[float],
    recv_nodes,
    n_steps: int,
    ops,
    *,
    checkpoint: int | None = None,
) -> Array:
    """Forward-model a shot: propagate ``source_wavelet`` from ``src_node`` through velocity ``c2`` and record
    the displacement at ``recv_nodes`` for ``n_steps`` steps. Returns the ``(n_steps+1, n_recv)`` seismogram
    (differentiable in ``c2``)."""
    nn = wave.n * wave.n
    state0 = wave.pack(ops.zeros(nn), ops.zeros(nn))
    recv = _long(recv_nodes)

    def record(state, i):
        return wave.displacement(state)[recv]

    step = _source_step(wave, ops, c2, src_node, source_wavelet)
    return ops.integrate_record(step, state0, n_steps, record, checkpoint=checkpoint)


def _forward_wavefield(wave, c2, step, n_steps, ops, checkpoint):
    """Propagate ``step`` and record the FULL displacement field at every step: the ``(n_steps+1, nn)`` movie."""
    nn = wave.n * wave.n
    state0 = wave.pack(ops.zeros(nn), ops.zeros(nn))

    def record(state, i):
        return wave.displacement(state)

    return ops.integrate_record(step, state0, n_steps, record, checkpoint=checkpoint)


def rtm_image(
    wave,
    c2_background,
    src_node: int,
    source_wavelet: Sequence[float],
    recv_nodes,
    residual,
    n_steps: int,
    ops,
    *,
    checkpoint: int | None = None,
    laplacian_filter: bool = True,
) -> Array:
    """Reverse-time-migration image from one shot's residual (scattered) data.

    Cross-correlates the forward source wavefield with the back-propagated receiver wavefield at zero lag,
    ``I(x) = sum_t S(x, t) R(x, t)``, both propagated through the smooth background ``c2_background``.

    Parameters
    ----------
    wave : WaveEquation2D
        The acoustic stepper defining the ``n x n`` grid (axis 0 = depth, axis 1 = horizontal).
    c2_background : array
        Squared-velocity field of the smooth migration model (length ``n*n``), no reflector.
    src_node : int
        Flat index of the point source.
    source_wavelet : sequence
        Source time function, length ``n_steps+1``.
    recv_nodes : array of int
        Flat indices of the receivers.
    residual : array, shape (n_steps+1, n_recv)
        The scattered data to migrate: observed minus background (direct) data, i.e. the reflection only.
    n_steps : int
        Number of time steps.
    ops :
        The backend-agnostic math namespace (``make_ops()``).
    checkpoint : int, optional
        Adjoint-checkpoint segment length passed to ``integrate_record`` (memory for time).
    laplacian_filter : bool, default True
        Apply a Laplacian post-filter to remove the low-wavenumber backscatter artifact.

    Returns
    -------
    image : numpy.ndarray, shape (n, n)
        The migrated reflectivity image (depth by horizontal).
    """
    n = wave.n
    # forward source wavefield S(x, t) in the background model
    s_step = _source_step(wave, ops, c2_background, src_node, source_wavelet)
    s_field = _forward_wavefield(wave, c2_background, s_step, n_steps, ops, checkpoint)

    # back-propagate the residual: inject time-reversed traces at the receivers, then flip back to forward time
    res_np = _as_np(residual)
    data_rev = res_np[::-1].copy()
    r_step = _data_step(wave, ops, c2_background, recv_nodes, data_rev)
    r_field_rev = _forward_wavefield(wave, c2_background, r_step, n_steps, ops, checkpoint)
    r_field = _flip_time(r_field_rev)

    # zero-lag cross-correlation imaging condition
    image = ops.sum(s_field * r_field, axis=0).reshape(n, n)
    image_np = _as_np(image)
    return _laplacian_2d(image_np) if laplacian_filter else image_np


def _flip_time(field):
    """Reverse a ``(n_steps+1, nn)`` wavefield movie along the time axis (torch or numpy)."""
    import torch

    if torch.is_tensor(field):
        return torch.flip(field, dims=[0])
    return field[::-1].copy()


def born_modeling(
    wave,
    c2_background,
    dm: Array,
    src_node: int,
    source_wavelet: Sequence[float],
    recv_nodes,
    n_steps: int,
    ops,
    *,
    checkpoint: int | None = None,
) -> Array:
    """Born (single-scattering) forward operator: predict the scattered data for a reflectivity ``dm``.

    Models with the perturbed velocity ``c2_background + dm`` and subtracts the background (direct) data,
    giving the linearized reflection response. This is the operator whose adjoint is :func:`rtm_image`;
    least-squares RTM inverts it. ``dm`` is a per-node squared-velocity perturbation (length ``n*n``,
    differentiable). Returns ``(n_steps+1, n_recv)``.
    """
    dm_t = dm if hasattr(dm, "detach") else ops.tensor(np.asarray(dm))
    c2_pert = c2_background + dm_t
    d_pert = model_data(wave, c2_pert, src_node, source_wavelet, recv_nodes, n_steps, ops, checkpoint=checkpoint)
    d_bg = model_data(wave, c2_background, src_node, source_wavelet, recv_nodes, n_steps, ops, checkpoint=checkpoint)
    return d_pert - d_bg


def lsrtm_step(
    wave,
    c2_background,
    dm: Array,
    src_node: int,
    source_wavelet: Sequence[float],
    recv_nodes,
    observed_scattered: Array,
    n_steps: int,
    ops,
    *,
    step_size: float,
    checkpoint: int | None = None,
    grad_fn: Callable[..., Array] | None = None,
) -> np.ndarray:
    """One gradient-descent step of least-squares RTM on ``0.5 || Born(dm) - d_scat ||^2``.

    The gradient of the least-squares misfit is the RTM image of the current data residual
    ``Born(dm) - d_scat``, so one LSRTM step is ``dm <- dm - step_size * rtm_image(residual)``. Returns the
    updated reflectivity ``dm`` as a flat numpy array of length ``n*n``. Pass ``grad_fn`` to override the
    gradient (defaults to the RTM image, unfiltered, which is the true adjoint).
    """
    n = wave.n
    d_pred = born_modeling(
        wave, c2_background, dm, src_node, source_wavelet, recv_nodes, n_steps, ops, checkpoint=checkpoint
    )
    residual = _as_np(d_pred) - _as_np(observed_scattered)
    if grad_fn is None:
        grad = rtm_image(
            wave,
            c2_background,
            src_node,
            source_wavelet,
            recv_nodes,
            residual,
            n_steps,
            ops,
            checkpoint=checkpoint,
            laplacian_filter=False,
        )
    else:
        grad = grad_fn(residual)
    dm_np = _as_np(dm).reshape(-1)
    return dm_np - float(step_size) * np.asarray(grad).reshape(n * n)


# ======================================================================================================
# 3-D acoustic RTM (:class:`~mixle_pde.wave3d.WaveEquation3D`)
# ======================================================================================================


def _source_step_3d(wave3d, ops, c2, src_node: int, ampl):
    """One-step closure that injects a time-dependent point source ``ampl[i]`` at ``src_node`` (3-D)."""
    nnn = wave3d.n**3

    def step(state, i):
        s = ops.zeros(nnn).clone()
        s[src_node] = float(ampl[i])
        return wave3d.step(state, c2, ops, source=s)

    return step


def _data_step_3d(wave3d, ops, c2, recv_nodes, data):
    """One-step closure that injects the recorded trace ``data[i]`` at every receiver node (3-D adjoint
    source)."""
    nnn = wave3d.n**3
    recv = _long(recv_nodes)

    def step(state, i):
        s = ops.zeros(nnn).clone()
        s[recv] = ops.tensor(np.asarray(data[i]))
        return wave3d.step(state, c2, ops, source=s)

    return step


def model_data_3d(
    wave3d,
    c2,
    src_node: int,
    source_wavelet: Sequence[float],
    recv_nodes,
    n_steps: int,
    ops,
    *,
    checkpoint: int | None = None,
) -> Array:
    """The 3-D counterpart of :func:`model_data`: forward-model a shot on a
    :class:`~mixle_pde.wave3d.WaveEquation3D` grid.

    Propagates ``source_wavelet`` from ``src_node`` through the squared-velocity volume ``c2`` (length
    ``n**3``) and records the displacement at ``recv_nodes`` for ``n_steps`` steps. Returns the
    ``(n_steps+1, n_recv)`` seismogram (differentiable in ``c2``).
    """
    nnn = wave3d.n**3
    state0 = wave3d.pack(ops.zeros(nnn), ops.zeros(nnn))
    recv = _long(recv_nodes)

    def record(state, i):
        return wave3d.displacement(state)[recv]

    step = _source_step_3d(wave3d, ops, c2, src_node, source_wavelet)
    return ops.integrate_record(step, state0, n_steps, record, checkpoint=checkpoint)


def _forward_wavefield_3d(wave3d, step, n_steps, ops, checkpoint):
    """Propagate ``step`` and record the FULL displacement volume at every step: the
    ``(n_steps+1, n**3)`` movie."""
    nnn = wave3d.n**3
    state0 = wave3d.pack(ops.zeros(nnn), ops.zeros(nnn))

    def record(state, i):
        return wave3d.displacement(state)

    return ops.integrate_record(step, state0, n_steps, record, checkpoint=checkpoint)


def _laplacian_3d(image: np.ndarray) -> np.ndarray:
    """Seven-point Laplacian of a 3-D volume (zero at the border); the 3-D analogue of
    :func:`_laplacian_2d`, removing the same low-wavenumber backscatter artifact."""
    out = np.zeros_like(image)
    out[1:-1, 1:-1, 1:-1] = (
        image[2:, 1:-1, 1:-1]
        + image[:-2, 1:-1, 1:-1]
        + image[1:-1, 2:, 1:-1]
        + image[1:-1, :-2, 1:-1]
        + image[1:-1, 1:-1, 2:]
        + image[1:-1, 1:-1, :-2]
        - 6.0 * image[1:-1, 1:-1, 1:-1]
    )
    return out


def rtm_image_3d(
    wave3d,
    c2_background,
    src_node: int,
    source_wavelet: Sequence[float],
    recv_nodes,
    residual,
    n_steps: int,
    ops,
    *,
    checkpoint: int | None = None,
    laplacian_filter: bool = True,
) -> np.ndarray:
    """Volumetric reverse-time-migration image from one 3-D shot's residual (scattered) data.

    The exact 3-D lift of :func:`rtm_image`: cross-correlates the forward source wavefield with the
    back-propagated receiver wavefield at zero lag, ``I(x) = sum_t S(x, t) R(x, t)``, both propagated
    through the smooth background ``c2_background`` on :class:`~mixle_pde.wave3d.WaveEquation3D`. Returns
    the ``(n, n, n)`` migrated reflectivity volume (depth by the two horizontal axes).
    """
    n = wave3d.n
    s_step = _source_step_3d(wave3d, ops, c2_background, src_node, source_wavelet)
    s_field = _forward_wavefield_3d(wave3d, s_step, n_steps, ops, checkpoint)

    res_np = _as_np(residual)
    data_rev = res_np[::-1].copy()
    r_step = _data_step_3d(wave3d, ops, c2_background, recv_nodes, data_rev)
    r_field_rev = _forward_wavefield_3d(wave3d, r_step, n_steps, ops, checkpoint)
    r_field = _flip_time(r_field_rev)

    image = ops.sum(s_field * r_field, axis=0).reshape(n, n, n)
    image_np = _as_np(image)
    return _laplacian_3d(image_np) if laplacian_filter else image_np


# ======================================================================================================
# 2-D elastic (P-SV) RTM
# ======================================================================================================


class _ElasticWave2D:
    """A minimal 2-D isotropic elastic (P-SV) velocity-stress stepper (Virieux 1986 staggered leapfrog).

    :mod:`mixle_pde.elastic` and :mod:`mixle_pde.elastic_aniso` only expose the 3-D ``(n, n, n)`` stepper
    today; this reduces the same staggered velocity-stress physics (same sponge, same one-sided differences
    centring every derivative at its staggered location, same leapfrog ordering) to the ``(z, x)`` plane, so
    :func:`elastic_rtm_2d` gets a real 2-D elastic operator without a 3-D-only dependency. State order is
    ``(vx, vz, sxx, szz, sxz)``, each a flat length-``n*n`` component.
    """

    def __init__(
        self,
        n: int,
        *,
        dt: float,
        spacing: float | None = None,
        lam,
        mu,
        rho,
        absorb_width: int = 0,
        absorb_strength: float = 2.0,
    ):
        self.n = int(n)
        self.dt = float(dt)
        self.h = float(spacing) if spacing is not None else 1.0 / (n - 1)
        self._nc = self.n * self.n
        self.lam = self._as_field(lam)
        self.mu = self._as_field(mu)
        self.rho = self._as_field(rho)
        self._inv_rho = 1.0 / self.rho
        self.vp_max = float(np.sqrt((np.max(self.lam) + 2.0 * np.max(self.mu)) / np.min(self.rho)))
        self._gamma = self._build_sponge(absorb_width, absorb_strength)

    def _as_field(self, x):
        a = np.asarray(x, dtype=float)
        if a.ndim == 0:
            return np.full((self.n, self.n), float(a))
        return a.reshape(self.n, self.n).astype(float)

    def _build_sponge(self, width, strength):
        n = self.n
        gamma = np.zeros((n, n))
        if width > 0:
            idx = np.arange(n)
            d = np.minimum(idx, n - 1 - idx)
            ramp = np.where(d < width, (1.0 - d / width) ** 2, 0.0)
            gz = np.broadcast_to(ramp[:, None], (n, n))
            gx = np.broadcast_to(ramp[None, :], (n, n))
            gamma = float(strength) * np.maximum(gz, gx)
        return gamma.ravel()

    def zeros(self, ops):
        return ops.zeros(5 * self._nc)

    @staticmethod
    def _dp(f, axis, h):
        lo, hi = [slice(None)] * 2, [slice(None)] * 2
        lo[axis], hi[axis] = slice(0, -1), slice(1, None)
        out = f * 0.0
        out[tuple(lo)] = (f[tuple(hi)] - f[tuple(lo)]) / h
        return out

    @staticmethod
    def _dm(f, axis, h):
        lo, hi = [slice(None)] * 2, [slice(None)] * 2
        lo[axis], hi[axis] = slice(1, None), slice(0, -1)
        out = f * 0.0
        out[tuple(lo)] = (f[tuple(lo)] - f[tuple(hi)]) / h
        return out

    @staticmethod
    def _roll_avg(f, axes):
        out = f
        for a in axes:
            lo, hi = [slice(None)] * 2, [slice(None)] * 2
            lo[a], hi[a] = slice(0, -1), slice(1, None)
            shifted = out * 0.0
            shifted[tuple(lo)] = out[tuple(hi)]
            out = shifted
        return out

    def step(self, state, ops, *, source=None):
        """Advance one full velocity-stress leapfrog step (stress a half step, then velocity)."""
        n, h, dt, nc = self.n, self.h, self.dt, self._nc
        vx, vz, sxx, szz, sxz = (state[i * nc : (i + 1) * nc].reshape(n, n) for i in range(5))
        lam, mu, inv_rho = ops.tensor(self.lam), ops.tensor(self.mu), ops.tensor(self._inv_rho)
        src = source or {}

        def add(name, field):
            s = src.get(name)
            return field if s is None else field + dt * ops.tensor(np.asarray(s).reshape(n, n))

        dvx_dx, dvz_dz = self._dm(vx, 1, h), self._dm(vz, 0, h)
        lam2mu = lam + 2.0 * mu
        sxx = add("sxx", sxx + dt * (lam2mu * dvx_dx + lam * dvz_dz))
        szz = add("szz", szz + dt * (lam2mu * dvz_dz + lam * dvx_dx))

        dvx_dz, dvz_dx = self._dp(vx, 0, h), self._dp(vz, 1, h)
        mu_xz = 0.5 * (mu + self._roll_avg(mu, (0, 1)))
        sxz = add("sxz", sxz + dt * mu_xz * (dvx_dz + dvz_dx))

        fx = self._dp(sxx, 1, h) + self._dm(sxz, 0, h)
        fz = self._dm(sxz, 1, h) + self._dp(szz, 0, h)
        vx = add("vx", vx + dt * inv_rho * fx)
        vz = add("vz", vz + dt * inv_rho * fz)

        gamma = ops.tensor(self._gamma).reshape(n, n)
        damp = 1.0 / (1.0 + dt * gamma)
        vx, vz = vx * damp, vz * damp

        return ops.cat([vx.reshape(-1), vz.reshape(-1), sxx.reshape(-1), szz.reshape(-1), sxz.reshape(-1)])


def _elastic_point_source_step_2d(stepper, ops, src_node: int, ampl, component: str):
    n = stepper.n
    iz, ix = divmod(int(src_node), n)

    def step(state, i):
        s = np.zeros((n, n))
        s[iz, ix] = float(ampl[i])
        return stepper.step(state, ops, source={component: s})

    return step


def _elastic_data_step_2d(stepper, ops, recv_nodes, data, component: str):
    n = stepper.n
    recv = np.asarray(recv_nodes)
    iz, ix = np.unravel_index(recv, (n, n))

    def step(state, i):
        s = np.zeros((n, n))
        s[iz, ix] = np.asarray(data[i])
        return stepper.step(state, ops, source={component: s})

    return step


def _elastic_forward_wavefield_2d(stepper, step, n_steps, ops, checkpoint):
    """Propagate ``step`` and record the ``(vx, vz)`` particle-velocity movie: ``(n_steps+1, 2*n*n)``."""
    nc = stepper._nc
    state0 = stepper.zeros(ops)

    def record(state, i):
        return ops.cat([state[0:nc], state[nc : 2 * nc]])

    return ops.integrate_record(step, state0, n_steps, record, checkpoint=checkpoint)


def elastic_rtm_2d(
    elastic,
    lam,
    mu,
    rho,
    src_node: int,
    source_wavelet: Sequence[float],
    recv_nodes,
    residual,
    n_steps: int,
    ops,
    **kw,
) -> np.ndarray:
    """2-D elastic (P-SV) RTM image from one shot's residual data.

    Builds a local :class:`_ElasticWave2D` stepper on ``elastic``'s grid (its ``n``, ``dt``, and
    ``h``/``spacing``) with the given ``lam, mu, rho`` background -- ``elastic`` may be an existing 3-D
    stepper such as :class:`mixle_pde.elastic.ElasticWave3D`, used purely for its grid metadata, exactly as
    :func:`rtm_image` takes a ``wave`` template and a separate velocity field. Propagates a single-component
    point source (``kw["component"]``, default the vertical force ``"vz"``) forward, back-propagates
    ``residual`` (data at ``recv_nodes`` for the same component) in reversed time, and applies the elastic
    energy imaging condition ``I(x) = sum_t [vx_s(x,t) vx_r(x,t) + vz_s(x,t) vz_r(x,t)]`` -- the P-SV
    analogue of the acoustic zero-lag cross-correlation, using both particle-velocity components so
    mode-converted (P-to-S) energy also contributes. Returns the ``(n, n)`` image (depth by horizontal).

    Keyword args: ``checkpoint`` (adjoint-checkpoint segment length), ``laplacian_filter`` (default
    ``True``), ``component`` (default ``"vz"``), ``absorb_width``/``absorb_strength`` (sponge, default off).
    """
    checkpoint = kw.get("checkpoint")
    laplacian_filter = kw.get("laplacian_filter", True)
    component = kw.get("component", "vz")
    absorb_width = kw.get("absorb_width", 0)
    absorb_strength = kw.get("absorb_strength", 2.0)

    n = int(elastic.n)
    dt = float(elastic.dt)
    h = float(getattr(elastic, "h", 1.0 / (n - 1)))
    stepper = _ElasticWave2D(
        n, dt=dt, spacing=h, lam=lam, mu=mu, rho=rho, absorb_width=absorb_width, absorb_strength=absorb_strength
    )
    nc = stepper._nc

    s_step = _elastic_point_source_step_2d(stepper, ops, src_node, source_wavelet, component)
    s_field = _elastic_forward_wavefield_2d(stepper, s_step, n_steps, ops, checkpoint)

    res_np = _as_np(residual)
    data_rev = res_np[::-1].copy()
    r_step = _elastic_data_step_2d(stepper, ops, recv_nodes, data_rev, component)
    r_field_rev = _elastic_forward_wavefield_2d(stepper, r_step, n_steps, ops, checkpoint)
    r_field = _flip_time(r_field_rev)

    s_vx, s_vz = s_field[:, :nc], s_field[:, nc : 2 * nc]
    r_vx, r_vz = r_field[:, :nc], r_field[:, nc : 2 * nc]
    image = (ops.sum(s_vx * r_vx, axis=0) + ops.sum(s_vz * r_vz, axis=0)).reshape(n, n)
    image_np = _as_np(image)
    return _laplacian_2d(image_np) if laplacian_filter else image_np
