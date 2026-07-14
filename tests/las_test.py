"""Conformance test for workstream B3 -- LAS well-log ingest + petrophysics (Definition of Done)."""

from __future__ import annotations

import os

import numpy as np

from mixle_pde.io.las import WellLog, load_las
from mixle_pde.petrophysics import archie_sw, gardner_density, vp_from_dt
from mixle_pde.rock_physics import moduli_from_velocity

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "tiny.las")


def _load() -> WellLog:
    return load_las(FIXTURE)


def test_load_las_curves_match_depth_length():
    log = _load()
    assert log.depth.ndim == 1 and log.depth.size == 20
    assert np.all(np.isfinite(log.depth))
    for name, curve in log.curves.items():
        assert curve.shape == log.depth.shape, f"curve {name!r} length mismatches depth"


def test_petrophysics_pipeline_produces_finite_moduli():
    log = _load()
    vp = vp_from_dt(log.curves["DT"])
    rho = log.curves["RHOB"]

    # density-porosity estimate (quartz matrix / fresh-water fluid) to drive Archie's equation --
    # RHOB alone does not carry porosity, so this is the standard log-derived phi proxy.
    phi = (2.65 - rho) / (2.65 - 1.0)
    assert np.all(phi > 0.0) and np.all(phi < 1.0)

    sw = archie_sw(log.curves["RT"], phi)
    assert sw.shape == log.depth.shape
    assert np.all(sw >= 0.0) and np.all(sw <= 1.0)

    K, mu = moduli_from_velocity(vp, 0.5 * vp, rho)
    assert np.all(np.isfinite(K)) and np.all(np.isfinite(mu))


def test_vp_from_dt_matches_closed_form():
    dt = np.array([50.0, 100.0])
    assert np.allclose(vp_from_dt(dt), 304.8 / dt)


def test_gardner_density_matches_closed_form():
    vp = np.array([2.0, 4.0])
    assert np.allclose(gardner_density(vp), 1.741 * vp**0.25)


def test_archie_sw_clips_to_unit_interval():
    rt = np.array([0.001, 1000.0])
    phi = np.array([0.3, 0.3])
    sw = archie_sw(rt, phi)
    assert np.all(sw >= 0.0) and np.all(sw <= 1.0)
