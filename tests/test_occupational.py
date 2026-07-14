"""K1 DoD: occupational-exposure transport (face-dust source + workplace ventilation).

``face_dust_ref.npy`` is an independent analytic reference: the closed-form steady solution of
the same governing equation (``D*C'' - u*C' - k*C = 0`` with Danckwerts flux-inlet/zero-gradient-
outlet boundary conditions) that ``occupational_exposure``'s finite-difference solve discretizes,
evaluated at the same grid nodes and physical parameters. It is not the numerical solver's own
output frozen as a golden file -- the two are independently derived, so a match is a real physics
check, not a tautology.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mixle_pde.dispersion import (
    ConcentrationField,
    Mesh,
    SourceTerm,
    VentilationBC,
    occupational_exposure,
)

FIXTURES = Path(__file__).parent / "fixtures"

# --- shared reference scenario: a face-dust source with once-through-plus-recirculated ventilation
LENGTH = 10.0
AIRFLOW_MPS = 0.5
RECIRC = 0.1
AREA = 2.0
EMISSION_RATE = 3.0e-3
N_NODES = 41


def _reference_scenario() -> tuple[SourceTerm, Mesh, VentilationBC]:
    mesh = Mesh(n_nodes=N_NODES, length=LENGTH, cross_section_area=AREA)
    ventilation = VentilationBC(inflow_faces=np.array([0]), airflow_mps=AIRFLOW_MPS, recirc=RECIRC)
    source = SourceTerm(rate=EMISSION_RATE, species="silica_pm4", location=0)
    return source, mesh, ventilation


def test_face_dust_matches_reference():
    source, mesh, ventilation = _reference_scenario()

    field = occupational_exposure(source, mesh, ventilation=ventilation, species="silica_pm4")

    reference = np.load(FIXTURES / "face_dust_ref.npy")
    assert field.mean.shape == reference.shape

    nrmse = np.sqrt(np.mean((field.mean - reference) ** 2)) / (reference.max() - reference.min())
    assert nrmse < 0.05


def test_result_is_a_concentration_field():
    source, mesh, ventilation = _reference_scenario()
    field = occupational_exposure(source, mesh, ventilation=ventilation)
    assert isinstance(field, ConcentrationField)
    assert field.mean.shape == (mesh.n_nodes,)
    assert np.all(field.mean >= 0.0)


def test_concentration_decays_away_from_the_working_face():
    source, mesh, ventilation = _reference_scenario()
    field = occupational_exposure(source, mesh, ventilation=ventilation)
    # concentration should be highest at (or very near) the working face and fall off downstream
    assert field.mean[0] >= field.mean[-1]


def test_higher_airflow_dilutes_the_concentration():
    source, mesh, _ = _reference_scenario()
    low = occupational_exposure(source, mesh, ventilation=VentilationBC(inflow_faces=np.array([0]), airflow_mps=0.2))
    high = occupational_exposure(source, mesh, ventilation=VentilationBC(inflow_faces=np.array([0]), airflow_mps=1.0))
    assert high.mean.max() < low.mean.max()


def test_deposition_rate_differs_by_species():
    mesh = Mesh(n_nodes=N_NODES, length=LENGTH, cross_section_area=AREA)
    ventilation = VentilationBC(inflow_faces=np.array([0]), airflow_mps=AIRFLOW_MPS)
    silica = occupational_exposure(
        SourceTerm(rate=EMISSION_RATE, species="silica_pm4"), mesh, ventilation=ventilation, species="silica_pm4"
    )
    coal = occupational_exposure(
        SourceTerm(rate=EMISSION_RATE, species="coal_dust_pm10"),
        mesh,
        ventilation=ventilation,
        species="coal_dust_pm10",
    )
    assert not np.allclose(silica.mean, coal.mean)


def test_transient_is_a_non_goal():
    source, mesh, ventilation = _reference_scenario()
    with pytest.raises(NotImplementedError):
        occupational_exposure(source, mesh, ventilation=ventilation, steady=False)


def test_ventilation_bc_validates_inputs():
    with pytest.raises(ValueError):
        VentilationBC(inflow_faces=np.array([0]), airflow_mps=-1.0)
    with pytest.raises(ValueError):
        VentilationBC(inflow_faces=np.array([0]), airflow_mps=1.0, recirc=1.5)


def test_source_rate_uncertainty_propagates_to_variance():
    class _ScalarPosterior:
        """A minimal IC-1-shaped posterior over a single emission rate."""

        def __init__(self, mean: float, var: float) -> None:
            self._mean = mean
            self._var = var

        @property
        def mean(self) -> np.ndarray:
            return np.array([self._mean])

        @property
        def cov(self) -> np.ndarray:
            return np.array([[self._var]])

    mesh = Mesh(n_nodes=N_NODES, length=LENGTH, cross_section_area=AREA)
    ventilation = VentilationBC(inflow_faces=np.array([0]), airflow_mps=AIRFLOW_MPS)

    deterministic = occupational_exposure(SourceTerm(rate=EMISSION_RATE), mesh, ventilation=ventilation)
    assert deterministic.variance is None

    uncertain = occupational_exposure(
        SourceTerm(rate=_ScalarPosterior(EMISSION_RATE, (0.2 * EMISSION_RATE) ** 2)),
        mesh,
        ventilation=ventilation,
    )
    assert uncertain.variance is not None
    assert np.all(uncertain.variance >= 0.0)
    np.testing.assert_allclose(uncertain.mean, deterministic.mean)

    lo, hi = uncertain.credible_interval(0.9)
    assert np.all(lo <= uncertain.mean) and np.all(uncertain.mean <= hi)

    rng = np.random.default_rng(0)
    draws = uncertain.samples(256, rng)
    assert draws.shape == (256, mesh.n_nodes)

    dq = uncertain.derived_quantity(lambda c: c.sum(axis=-1), 256, np.random.default_rng(1))
    assert dq.prior_dominated is False
    dq_lo, dq_hi = dq.credible_interval(0.9)
    assert np.all(dq_lo <= dq_hi)
