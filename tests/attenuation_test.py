"""Validation of the absorption/attenuation models against their published reference values.

Reference values asserted:
  * Thorp seawater: alpha(1 kHz) ~ 0.06 dB/km, alpha(10 kHz) ~ 1.0-1.2 dB/km (classic 1967 form).
  * Francois-Garrison: order-of-magnitude agreement with Thorp at 10 kHz, 4 C, 35 ppt.
  * ITU-R P.676 oxygen: ~15 dB/km in the 60 GHz band, ~0.01 dB/km at 10 GHz (within a factor ~2).
  * ITU-R P.838 rain: gamma_R ~ 1-2 dB/km at 10 GHz, 50 mm/h, horizontal.
  * Differentiability of each model w.r.t. frequency / rain rate (torch autograd).
  * The Q bridge reproduces the solver's exp(-omega r / (2 Q c)) decay.
"""

import numpy as np
import pytest

from mixle_pde.attenuation import (
    complex_modulus_fraction,
    db_per_length_to_nepers,
    francois_garrison_seawater,
    itu_gaseous,
    itu_gaseous_oxygen,
    itu_rain_specific,
    quality_factor,
    thorp_seawater,
)

torch = pytest.importorskip("torch")


# --- seawater ------------------------------------------------------------------------------------------------
def test_thorp_reference_values():
    # alpha(1 kHz) ~ 0.06 dB/km and alpha(10 kHz) in 1.0-1.2 dB/km
    a1 = thorp_seawater(1.0)
    a10 = thorp_seawater(10.0)
    assert a1 == pytest.approx(0.063, abs=0.01)
    assert 1.0 <= a10 <= 1.2
    # the classic value at 10 kHz
    assert a10 == pytest.approx(1.082, abs=0.02)
    # monotone increasing across the band and positive floor at DC
    fs = np.array([0.1, 1.0, 3.0, 10.0, 30.0, 100.0])
    vals = np.array([thorp_seawater(f) for f in fs])
    assert np.all(np.diff(vals) > 0)
    assert thorp_seawater(0.0) == pytest.approx(0.003, abs=1e-9)


def test_francois_garrison_matches_thorp_order_of_magnitude():
    for f in (1.0, 10.0, 25.0):
        fg = francois_garrison_seawater(f, temperature_c=4.0, salinity_ppt=35.0)
        th = thorp_seawater(f)
        assert 0.5 < fg / th < 2.0  # same order of magnitude across the sonar band
    # at 10 kHz they agree to a few percent
    fg10 = francois_garrison_seawater(10.0, temperature_c=4.0, salinity_ppt=35.0)
    assert fg10 == pytest.approx(thorp_seawater(10.0), rel=0.15)
    # warmer/fresher water attenuates differently but stays physical (positive, finite)
    warm = francois_garrison_seawater(10.0, temperature_c=20.0, salinity_ppt=30.0, depth_m=1000.0)
    assert warm > 0.0 and np.isfinite(warm)


# --- atmosphere: gaseous -------------------------------------------------------------------------------------
def test_itu_oxygen_reference_values():
    # ~15 dB/km in the 60 GHz oxygen complex (within a factor ~2)
    g60 = itu_gaseous_oxygen(60.0)
    assert 7.5 <= g60 <= 30.0
    assert g60 == pytest.approx(15.0, rel=0.5)
    # ~0.01 dB/km at 10 GHz (within a factor ~2)
    g10 = itu_gaseous_oxygen(10.0)
    assert 0.005 <= g10 <= 0.02
    # the 60 GHz band is >1000x more absorbing than the 10 GHz window
    assert g60 / g10 > 1000.0


def test_itu_total_gaseous_has_water_line():
    # the 22.235 GHz water-vapour line lifts total absorption above the oxygen-only baseline
    total = itu_gaseous(22.235, vapour_hpa=7.5)
    o2 = itu_gaseous_oxygen(22.235, vapour_hpa=7.5)
    assert total > o2
    assert total == pytest.approx(0.18, rel=0.6)  # ~0.1-0.2 dB/km at the resonance, standard humidity


# --- atmosphere: rain --------------------------------------------------------------------------------------
def test_itu_rain_reference_value():
    # 10 GHz, 50 mm/h, horizontal -> ~1-2 dB/km
    g = itu_rain_specific(10.0, 50.0, polarisation="h")
    assert 1.0 <= g <= 2.0
    assert g == pytest.approx(1.66, rel=0.15)
    # horizontal polarisation attenuates more than vertical (oblate drops)
    gv = itu_rain_specific(10.0, 50.0, polarisation="v")
    assert g > gv
    # k R^alpha superlinear in rain rate at 10 GHz (alpha > 1)
    assert itu_rain_specific(10.0, 100.0) / itu_rain_specific(10.0, 50.0) > 2.0


def test_itu_rain_at_table_knot_is_exact():
    # at a tabulated frequency the interpolation returns the tabulated coefficient exactly
    g = itu_rain_specific(10.0, 1.0, polarisation="h")  # R=1 -> gamma = kH
    assert g == pytest.approx(0.01217, rel=1e-6)


# --- differentiability -------------------------------------------------------------------------------------
def test_thorp_differentiable_in_frequency():
    f = torch.tensor(10.0, dtype=torch.float64, requires_grad=True)
    thorp_seawater(f).backward()
    assert f.grad is not None and torch.isfinite(f.grad) and f.grad > 0  # rising with frequency


def test_francois_garrison_differentiable():
    f = torch.tensor(10.0, dtype=torch.float64, requires_grad=True)
    T = torch.tensor(4.0, dtype=torch.float64, requires_grad=True)
    from mixle_pde.ops import make_ops

    a = francois_garrison_seawater(f, temperature_c=T, salinity_ppt=35.0, ops=make_ops())
    a.backward()
    assert torch.isfinite(f.grad) and f.grad > 0
    assert torch.isfinite(T.grad)  # temperature sensitivity is finite


def test_itu_oxygen_differentiable_in_frequency():
    from mixle_pde.ops import make_ops

    f = torch.tensor(55.0, dtype=torch.float64, requires_grad=True)
    itu_gaseous_oxygen(f, ops=make_ops()).backward()
    assert torch.isfinite(f.grad) and f.grad != 0.0  # steep flank of the 60 GHz band


def test_itu_rain_differentiable_in_rate_and_frequency():
    from mixle_pde.ops import make_ops

    ops = make_ops()
    r = torch.tensor(50.0, dtype=torch.float64, requires_grad=True)
    itu_rain_specific(10.0, r, polarisation="h", ops=ops).backward()
    assert torch.isfinite(r.grad) and r.grad > 0
    f = torch.tensor(13.0, dtype=torch.float64, requires_grad=True)
    itu_rain_specific(f, 50.0, polarisation="h", ops=ops).backward()
    assert torch.isfinite(f.grad) and f.grad > 0  # more rain loss at higher frequency here


# --- Q bridge --------------------------------------------------------------------------------------------
def test_db_to_nepers_conversion():
    # 8.686 dB = 1 neper (amplitude), so 8.686 dB/km -> 1 neper/km
    assert db_per_length_to_nepers(8.685889638) == pytest.approx(1.0, rel=1e-6)


def test_quality_factor_reproduces_amplitude_decay():
    # a 10 kHz sonar ping in seawater: Thorp alpha, c=1500 m/s. Check exp(-omega r/(2 Q c)) == 10^(-alpha_dB r /20).
    alpha_db = float(thorp_seawater(10.0))  # dB/km
    f_hz = 10_000.0
    c = 1500.0
    Q = quality_factor(alpha_db, f_hz, c)  # length_scale defaults to 1000 (dB/km)
    assert Q > 0.0 and np.isfinite(Q)
    r = 500.0  # metres
    omega = 2.0 * np.pi * f_hz
    solver_amp = np.exp(-omega * r / (2.0 * Q * c))
    physical_amp = 10.0 ** (-(alpha_db * (r / 1000.0)) / 20.0)
    assert solver_amp == pytest.approx(physical_amp, rel=1e-9)
    # loss tangent is exactly 1/Q
    assert complex_modulus_fraction(alpha_db, f_hz, c) == pytest.approx(1.0 / Q, rel=1e-12)


def test_quality_factor_differentiable():
    f = torch.tensor(10.0, dtype=torch.float64, requires_grad=True)  # kHz for the model
    alpha_db = thorp_seawater(f)
    Q = quality_factor(alpha_db, f * 1000.0, 1500.0)  # f in Hz for the bridge
    Q.backward()
    assert torch.isfinite(f.grad)
