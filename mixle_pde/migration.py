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
