"""C9 acceptance tests: 3-D acoustic RTM, 2-D elastic RTM, and single-gather processing.

The headline check is the 3-D lift of RTM: forward-model a shot with :func:`model_data_3d` on a synthetic
layered volume with one flat reflector, image the residual with :func:`rtm_image_3d`, and confirm the image
peaks at the true reflector depth (within +/- 1 cell). Alongside it: the elastic 2-D RTM energy imaging
condition runs end to end on a small P-SV problem; :func:`nmo_correction` flattens a known hyperbolic
moveout; :func:`stack` collapses a corrected gather; and :func:`well_tie` recovers a known checkshot
(sample-lag) mis-tie between a synthetic-from-log seismogram and a recorded trace.
"""

from __future__ import annotations

import numpy as np
import pytest

from mixle_pde.migration import elastic_rtm_2d, model_data_3d, rtm_image_3d
from mixle_pde.ops import make_ops
from mixle_pde.seismic_processing import _shift_zero_fill, nmo_correction, stack, well_tie
from mixle_pde.wave3d import WaveEquation3D

torch = pytest.importorskip("torch")


# ==========================================================================================================
# 3-D acoustic RTM -- the Definition-of-Done check
# ==========================================================================================================


def _layered_3d_setup():
    """A small 3-D two-layer model with one flat, horizontal interface at ``z_ref``.

    Axis 0 = depth ``z`` (down); axes 1, 2 = the two horizontal directions. A single source with a small
    areal patch of receivers just below the top of the model, matching the tuning of the working 2-D RTM
    reference test (``tests/migration_test.py``) scaled to 3-D.
    """
    torch.set_default_dtype(torch.float64)
    n = 30
    h = 1.0 / (n - 1)
    c0 = 1.0
    dt = 0.2 * h / c0
    aw = 5  # absorbing sponge width
    wave = WaveEquation3D(n, dt=dt, spacing=h, absorb_width=aw, absorb_strength=0.8 / dt)
    ops = make_ops()

    z_top = aw + 2
    z_ref = 18  # reflector depth (grid index), well inside the sponge-free interior
    c_bg = np.full((n, n, n), c0)
    c_true = c_bg.copy()
    c_true[z_ref:, :, :] = 1.6 * c0  # faster lower layer -> one reflection
    c2_bg = torch.as_tensor((c_bg**2).ravel())
    c2_true = torch.as_tensor((c_true**2).ravel())

    mid = n // 2
    src_node = (z_top * n + mid) * n + mid

    f0 = 6.0
    dist_round_trip = 2.0 * (z_ref - z_top) * h
    nt = int((dist_round_trip / c0 + 6.0 / f0) / dt) + 10
    tg = np.arange(nt + 1) * dt
    a = (np.pi * f0 * (tg - 3.0 / f0)) ** 2
    ricker = (1 - 2 * a) * np.exp(-a)

    lo, hi = aw + 2, n - aw - 2
    xs = np.arange(lo, hi, 2)
    ys = np.arange(lo, hi, 2)
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    recv_nodes = (z_top * n + yy.ravel()) * n + xx.ravel()

    return {
        "n": n,
        "nt": nt,
        "wave": wave,
        "ops": ops,
        "z_ref": z_ref,
        "z_top": z_top,
        "aw": aw,
        "c2_bg": c2_bg,
        "c2_true": c2_true,
        "src_node": src_node,
        "ricker": ricker,
        "recv_nodes": recv_nodes,
    }


def _scattered_data_3d(s):
    """Observed (true model) minus background (direct) data = the isolated 3-D reflection."""
    ops, wave, nt = s["ops"], s["wave"], s["nt"]
    d_true = model_data_3d(
        wave, s["c2_true"], s["src_node"], s["ricker"], s["recv_nodes"], nt, ops, checkpoint=20
    ).detach()
    d_bg = model_data_3d(wave, s["c2_bg"], s["src_node"], s["ricker"], s["recv_nodes"], nt, ops, checkpoint=20).detach()
    return d_true - d_bg


def test_rtm_image_3d_peaks_at_reflector_depth():
    s = _layered_3d_setup()
    residual = _scattered_data_3d(s)
    # the scattered field must be a real reflection, not numerical dust
    assert float(residual.abs().max()) > 1e-8

    image = rtm_image_3d(
        s["wave"],
        s["c2_bg"],
        s["src_node"],
        s["ricker"],
        s["recv_nodes"],
        residual,
        s["nt"],
        s["ops"],
        checkpoint=20,
    )
    assert image.shape == (s["n"], s["n"], s["n"])
    assert np.isfinite(image).all()

    z_top, aw, n = s["z_top"], s["aw"], s["n"]
    zlo, zhi = z_top + 2, n - aw - 2
    profile = np.abs(image[zlo:zhi]).sum(axis=(1, 2))
    zz = np.arange(zlo, zhi)
    peak_z = int(zz[np.argmax(profile)])
    # the RTM image peaks AT the reflector depth, within one grid cell
    assert abs(peak_z - s["z_ref"]) <= 1


def test_model_data_3d_is_differentiable_in_velocity():
    """The 3-D forward keeps gradients flowing to ``c2`` (the FWI/LSRTM sensitivity), like the 2-D one."""
    s = _layered_3d_setup()
    c2 = s["c2_bg"].clone().requires_grad_(True)
    d = model_data_3d(s["wave"], c2, s["src_node"], s["ricker"], s["recv_nodes"], s["nt"], s["ops"], checkpoint=20)
    d.sum().backward()
    assert c2.grad is not None
    assert torch.isfinite(c2.grad).all()
    assert float(c2.grad.abs().sum()) > 0.0


# ==========================================================================================================
# 2-D elastic RTM (P-SV energy imaging condition) -- sanity/plumbing check
# ==========================================================================================================


def test_elastic_rtm_2d_produces_a_finite_image_with_energy_below_the_source():
    n = 36
    h = 1.0 / (n - 1)
    rho0 = 1.0
    vp_bg, vs_bg = 1.5, 0.9
    vp_true, vs_true = 2.0, 1.2
    aw = 6
    dt = 0.2 * h / vp_bg
    z_ref = 22

    lam_bg = np.full((n, n), rho0 * (vp_bg**2 - 2.0 * vs_bg**2))
    mu_bg = np.full((n, n), rho0 * vs_bg**2)
    rho_bg = np.full((n, n), rho0)
    lam_true, mu_true, rho_true = lam_bg.copy(), mu_bg.copy(), rho_bg.copy()
    lam_true[z_ref:, :] = rho0 * (vp_true**2 - 2.0 * vs_true**2)
    mu_true[z_ref:, :] = rho0 * vs_true**2

    ops = make_ops()

    class _GridTemplate:
        """Grid-only stand-in for ``elastic`` (mirrors how ``rtm_image`` takes a ``wave`` template)."""

        def __init__(self, n, dt, h):
            self.n, self.dt, self.h = n, dt, h

    template = _GridTemplate(n, dt, h)

    z_top = aw + 2
    mid = n // 2
    src_node = z_top * n + mid

    f0 = 6.0
    dist_round_trip = 2.0 * (z_ref - z_top) * h
    nt = int((dist_round_trip / vp_bg + 6.0 / f0) / dt) + 10
    tg = np.arange(nt + 1) * dt
    a = (np.pi * f0 * (tg - 3.0 / f0)) ** 2
    ricker = (1 - 2 * a) * np.exp(-a)

    lo, hi = aw + 2, n - aw - 2
    recv_nodes = z_top * n + np.arange(lo, hi, 2)

    # residual data: any nonzero perturbation stands in for a real true-minus-background seismogram --
    # this test exercises the imaging-condition plumbing (shapes, dtypes, finiteness, depth localisation
    # inside the sponge-free interior), not a full elastic Born-linearization cross-check.
    rng = np.random.default_rng(0)
    residual = 1e-3 * rng.standard_normal((nt + 1, recv_nodes.shape[0]))

    image = elastic_rtm_2d(
        template,
        lam_bg,
        mu_bg,
        rho_bg,
        src_node,
        ricker,
        recv_nodes,
        residual,
        nt,
        ops,
        checkpoint=20,
        absorb_width=aw,
        absorb_strength=0.8 / dt,
    )
    assert image.shape == (n, n)
    assert np.isfinite(image).all()

    # energy is concentrated in the subsurface interior, not smeared onto the source node / sponge
    zlo, zhi = z_top + 1, n - aw - 1
    interior = np.abs(image[zlo:zhi, :])
    assert interior.sum() > 0.0
    weights = interior.sum(axis=1)
    centroid_z = float(np.average(np.arange(zlo, zhi), weights=weights))
    assert z_top < centroid_z < (n - aw)


# ==========================================================================================================
# nmo_correction / stack
# ==========================================================================================================


def _ricker_at(t_axis, t_center, f0):
    a = (np.pi * f0 * (t_axis - t_center)) ** 2
    return (1.0 - 2.0 * a) * np.exp(-a)


def test_nmo_correction_flattens_a_known_hyperbola():
    n_samples = 400
    dt = 0.002
    t0_axis = np.arange(n_samples) * dt
    offsets = np.array([0.0, 200.0, 400.0, 600.0, 800.0])
    vnmo = 2000.0
    t0_true = 0.3
    f0 = 20.0

    gather = np.zeros((n_samples, offsets.size))
    for j, x in enumerate(offsets):
        t_obs = np.sqrt(t0_true**2 + x**2 / vnmo**2)
        gather[:, j] = _ricker_at(t0_axis, t_obs, f0)

    # before correction the event is NOT aligned: far-offset arrival is measurably later
    raw_peak_times = t0_axis[np.argmax(np.abs(gather), axis=0)]
    assert raw_peak_times[-1] - raw_peak_times[0] > 5 * dt

    corrected = nmo_correction(gather, offsets, t0_axis, vnmo)
    assert corrected.shape == gather.shape
    peak_times = t0_axis[np.argmax(np.abs(corrected), axis=0)]
    assert np.allclose(peak_times, t0_true, atol=2 * dt)


def test_stack_is_the_offset_axis_mean():
    gathers = np.array([[1.0, 3.0, 5.0], [2.0, 4.0, 6.0]])
    stacked = stack(gathers)
    assert np.allclose(stacked, [3.0, 4.0])
    # a 1-D trace passes through unchanged
    assert np.allclose(stack(np.array([7.0, 8.0])), [7.0, 8.0])


# ==========================================================================================================
# well_tie -- the Definition-of-Done checkshot-shift check
# ==========================================================================================================


def test_well_tie_recovers_known_checkshot_shift():
    n = 300
    dt = 0.002
    log_twt = np.arange(n) * dt
    reflectivity = np.zeros(n)
    spike_idx = [40, 90, 150, 210, 260]
    spike_val = [0.12, -0.08, 0.15, -0.05, 0.09]
    reflectivity[spike_idx] = spike_val

    wavelet = _ricker_at(np.arange(41) * dt - 41 // 2 * dt, 0.0, 25.0)
    true_synthetic = np.convolve(reflectivity, wavelet, mode="same")

    injected_shift = 15
    rng = np.random.default_rng(1)
    seismic_trace = _shift_zero_fill(true_synthetic, injected_shift) + 1e-3 * rng.standard_normal(n)

    result = well_tie(reflectivity, log_twt, seismic_trace, wavelet=wavelet, max_lag=60)
    assert set(result) >= {"synthetic", "shift", "correlation", "time_depth"}
    assert result["synthetic"].shape == (n,)
    assert result["shift"] == injected_shift
    assert result["correlation"] > 0.9
    assert np.allclose(result["time_depth"], log_twt)


def test_well_tie_zero_shift_when_perfectly_tied():
    n = 200
    dt = 0.002
    log_twt = np.arange(n) * dt
    reflectivity = np.zeros(n)
    reflectivity[[50, 100, 150]] = [0.1, -0.1, 0.08]

    wavelet = _ricker_at(np.arange(31) * dt - 31 // 2 * dt, 0.0, 30.0)
    synthetic = np.convolve(reflectivity, wavelet, mode="same")

    result = well_tie(reflectivity, log_twt, synthetic, wavelet=wavelet, max_lag=30)
    assert result["shift"] == 0
    assert result["correlation"] > 0.99
