"""L2 DoD: a dry climate scenario flags a water shortfall; the baseline scenario does not.

Loads the synthetic ``catchment_stub`` mesh fixture and drives :func:`water_balance` with two climate
scenarios that differ only in precipitation: a low-precipitation ("dry", e.g. a downscaled drought
projection) scenario and a baseline scenario. Both climate drivers carry a ``content_hash`` (the shape an
L4 ``ProvenancedResult`` and a plain dict share), so both resulting budgets must carry a non-null
``climate_hash``. The dry scenario should be unable to keep pace with evaporation + demand and accumulate
a positive ``shortfall_m3``; the baseline scenario should comfortably cover both and keep ``shortfall_m3``
at exactly zero.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from mixle_pde.mesh import SimplexMesh
from mixle_pde.water_balance import WaterBudget, water_balance

FIXTURE = Path(__file__).parent / "fixtures" / "catchment_stub.npz"


def _catchment() -> SimplexMesh:
    data = np.load(FIXTURE)
    return SimplexMesh(data["nodes"], data["simplices"])


def _stub_climate(*, precip_mm: float, evap_mm: float, tag: str) -> dict:
    """A minimal stand-in for an L4 ``ClimateProjectionStub`` ``ProvenancedResult``: same shape
    (``value``/``content_hash``/``model_id``/``version``), no ``mixle_mlops`` import required."""
    return {
        "value": {"precip_mm": precip_mm, "evap_mm": evap_mm},
        "content_hash": tag * 64,
        "model_id": "climate-stub",
        "version": "v1",
    }


def test_dry_scenario_flags_shortfall():
    catchment = _catchment()
    baseline = _stub_climate(precip_mm=80.0, evap_mm=50.0, tag="b")
    dry = _stub_climate(precip_mm=10.0, evap_mm=50.0, tag="d")

    wb_baseline = water_balance(catchment, climate=baseline, demand_m3=20_000.0, storage0_m3=50_000.0)
    wb_dry = water_balance(catchment, climate=dry, demand_m3=20_000.0, storage0_m3=50_000.0)

    assert isinstance(wb_baseline, WaterBudget)
    assert isinstance(wb_dry, WaterBudget)

    assert wb_dry.shortfall_m3 > 0.0
    assert wb_baseline.shortfall_m3 == 0.0

    # every water number traces back to the climate driver that produced it
    assert wb_baseline.climate_hash is not None and len(wb_baseline.climate_hash) == 64
    assert wb_dry.climate_hash is not None and len(wb_dry.climate_hash) == 64
    assert wb_baseline.climate_hash == baseline["content_hash"]
    assert wb_dry.climate_hash == dry["content_hash"]

    # the dry scenario should shift the budget: less routed inflow, lower/zero terminal storage
    assert wb_dry.inflow.sum() < wb_baseline.inflow.sum()
    assert wb_dry.storage[-1] <= wb_baseline.storage[-1]

    for wb in (wb_baseline, wb_dry):
        assert wb.inflow.shape == (12,)
        assert wb.outflow.shape == (12,)
        assert wb.storage.shape == (12,)
        assert np.all(wb.storage >= 0.0)
        assert wb.provenance["climate_hash"] == wb.climate_hash


def test_provenanced_result_like_object_is_accepted():
    """A duck-typed ProvenancedResult (``.value``/``.content_hash``, no dict) works the same way."""

    class _Stub:
        def __init__(self, value, content_hash, model_id="cmip6-stub", version="v2"):
            self.value = value
            self.content_hash = content_hash
            self.model_id = model_id
            self.version = version

    catchment = _catchment()
    climate = _Stub({"precip_mm": 80.0, "evap_mm": 50.0}, "e" * 64)
    wb = water_balance(catchment, climate=climate, demand_m3=20_000.0, storage0_m3=50_000.0, steps=6)

    assert wb.climate_hash == "e" * 64
    assert wb.provenance["climate_model_id"] == "cmip6-stub"
    assert wb.inflow.shape == (6,)


def test_missing_climate_hash_reports_none():
    catchment = _catchment()
    climate = {"precip_mm": 80.0, "evap_mm": 50.0}  # no content_hash supplied
    wb = water_balance(catchment, climate=climate, demand_m3=1_000.0, storage0_m3=10_000.0, steps=2)
    assert wb.climate_hash is None
