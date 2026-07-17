"""DoD test for workstream G4 -- InSAR subsidence ingest + poroelastic deformation forward/inverse.

A synthetic aquifer-dewatering ``dV`` field (drawn from the SAME Gaussian smoothness prior
:func:`~mixle_pde.poroelastic.invert_deformation` regularizes with -- the standard simulation-based-
calibration recipe for checking a Bayesian linear-Gaussian inversion is honestly calibrated, matching
the existing `field_inversion_test.py` acceptance test's own "coverage over repeated noise draws"
philosophy) is forward-modelled through the Mogi poroelastic Green's function
(:func:`~mixle_pde.poroelastic.poroelastic_subsidence`) into a synthetic LOS-displacement GeoTIFF,
written to disk and read back with :func:`~mixle_pde.io.insar.load_insar` -- exercising the full
raster -> Observation -> posterior pipeline, not a shortcut through hand-built Observations. The
recovered posterior (:func:`~mixle_pde.poroelastic.invert_deformation`) must bracket the true recovered
dewatering source's magnitude (total volume change) and centroid (its recovered location) in their 90%
credible regions.
"""

import os

import numpy as np
import pytest

from mixle_pde.field_inversion import FieldGaussianPrior
from mixle_pde.io.insar import load_insar
from mixle_pde.latent import Field3D
from mixle_pde.observations import Observation
from mixle_pde.poroelastic import DeformationPosterior, gassmann_moduli, invert_deformation, poroelastic_subsidence

rasterio = pytest.importorskip("rasterio")

MODULI = gassmann_moduli(k_solid=36.0e9, k_fluid=2.2e9, k_dry=9.0e9, mu=7.0e9, phi=0.25)
CRS = "EPSG:32611"


def _aquifer_cells():
    """A 7x7 grid of aquifer cells at 400 m depth, 100 m spacing, spanning +/-300 m."""
    xs = np.linspace(-300.0, 300.0, 7)
    ys = np.linspace(-300.0, 300.0, 7)
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    return np.column_stack([xx.ravel(), yy.ravel(), np.full(xx.size, -400.0)])


def _true_dewatering_field(cells, *, seed, prior):
    """A dewatering (predominantly-contracting) dV field drawn from ``prior`` -- guarantees the
    resulting posterior is calibrated (a draw from the SAME prior the inversion regularizes with)."""
    grid = Field3D(coordinates=cells, spacing=1.0, units="m^3", property_name="volume_change", bounds=None)
    cov = np.linalg.inv(prior.precision(grid))
    chol = np.linalg.cholesky(cov)
    z = np.random.default_rng(seed).standard_normal(cells.shape[0])
    dv = chol @ z
    return -dv if dv.sum() > 0.0 else dv  # flip to the "dewatering" (net-negative) branch; N(0, cov) is symmetric


def _write_los_raster(path, disp_grid, *, transform, crs=CRS):
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=disp_grid.shape[0],
        width=disp_grid.shape[1],
        count=1,
        dtype="float64",
        crs=crs,
        transform=transform,
    ) as ds:
        ds.write(disp_grid, 1)


def _pixel_grid(n_pix, half_extent):
    """(n_pix, n_pix) surface observation points (east, north, 0) plus the matching rasterio transform."""
    from rasterio.transform import Affine

    px = 2.0 * half_extent / n_pix
    transform = Affine(px, 0.0, -half_extent, 0.0, -px, half_extent)
    x = -half_extent + px * (np.arange(n_pix) + 0.5)
    y = half_extent - px * (np.arange(n_pix) + 0.5)
    xx, yy = np.meshgrid(x, y)
    obs_xyz = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(xx.size)])
    return obs_xyz, transform


def _centroid(m, cells):
    """Volume-weighted (x, y) centroid of the contracting (dV < 0) part of a dV field."""
    w = np.clip(-m, 0.0, None)
    denom = max(float(w.sum()), 1.0e-6)
    return np.array([np.sum(w * cells[:, 0]) / denom, np.sum(w * cells[:, 1]) / denom])


def test_load_insar_reads_valid_pixels_drops_nan_and_carries_los_vector(tmp_path):
    n_pix, half_extent = 10, 200.0
    obs_xyz, transform = _pixel_grid(n_pix, half_extent)
    disp = 0.01 * np.ones((n_pix, n_pix))
    disp[0, 0] = np.nan  # a low-coherence gap, the usual real-unwrapped-interferogram case

    path = os.path.join(tmp_path, "los.tif")
    _write_los_raster(path, disp, transform=transform)

    los_vector = (0.05, -0.02, 0.998)
    obs = load_insar(path, crs=CRS, los_vector=los_vector, noise_std=0.005)

    assert isinstance(obs, list) and len(obs) == 1
    o = obs[0]
    assert o.kind == "insar_los"
    assert o.crs == CRS
    assert o.modality == "insar"
    assert o.n == n_pix * n_pix - 1  # the NaN pixel is dropped
    assert np.all(np.isfinite(o.value))
    assert np.allclose(o.value, 0.01)
    normalized = np.asarray(los_vector) / np.linalg.norm(los_vector)
    assert np.allclose(o.provenance["los_vector"], normalized, atol=1e-9)


def test_poroelastic_subsidence_is_linear_and_requires_positive_depth():
    cells = np.array([[0.0, 0.0, -100.0]])
    obs_xy = np.array([[0.0, 0.0, 0.0], [50.0, 0.0, 0.0]])

    d1 = poroelastic_subsidence(np.array([-1000.0]), cells, obs_xy, moduli=MODULI)
    d2 = poroelastic_subsidence(np.array([-2000.0]), cells, obs_xy, moduli=MODULI)
    assert np.allclose(d2, 2.0 * d1)  # linear in volume_change
    assert d1[0] < 0.0  # dewatering (negative dV) subsides the straight-up point (los=(0,0,1))
    assert abs(d1[1]) < abs(d1[0])  # displacement decays away from the epicentre

    with pytest.raises(ValueError):
        # an "observation" at or below the source depth breaks the (east, north, up) convention
        poroelastic_subsidence(np.array([-1000.0]), cells, np.array([[0.0, 0.0, -150.0]]), moduli=MODULI)


def test_invert_deformation_recovers_dewatering_source_in_90pct_credible_region(tmp_path):
    cells = _aquifer_cells()
    prior_kwargs = dict(prior_volume_scale=1.0e4, smoothness_precision=2.0e-4, length_scale=100.0)
    prior = FieldGaussianPrior(
        mean=0.0,
        smoothness_precision=prior_kwargs["smoothness_precision"],
        marginal_precision=1.0 / prior_kwargs["prior_volume_scale"] ** 2,
        length_scale=prior_kwargs["length_scale"],
    )
    true_dv = _true_dewatering_field(cells, seed=9, prior=prior)
    true_total = float(true_dv.sum())
    true_centroid = _centroid(true_dv, cells)

    los_vector = np.array([0.1, -0.05, 0.993])
    obs_xyz, transform = _pixel_grid(25, 400.0)
    clean_disp = poroelastic_subsidence(true_dv, cells, obs_xyz, moduli=MODULI, los_vector=los_vector)

    noise_std = 0.003
    noisy = clean_disp + np.random.default_rng(3).normal(0.0, noise_std, clean_disp.shape)
    path = os.path.join(tmp_path, "subsidence.tif")
    _write_los_raster(path, noisy.reshape(25, 25), transform=transform)

    obs = load_insar(path, crs=CRS, los_vector=los_vector, noise_std=noise_std)
    assert obs[0].n == 25 * 25

    posterior = invert_deformation(obs, cells, moduli=MODULI, **prior_kwargs)
    assert isinstance(posterior, DeformationPosterior)
    assert posterior.mean.shape == (cells.shape[0],)
    assert posterior.cov.shape == (cells.shape[0], cells.shape[0])

    rng = np.random.default_rng(123)
    samples = posterior.samples(2000, rng)
    assert samples.shape == (2000, cells.shape[0])

    total_dq = posterior.derived_quantity(lambda m: np.array([m.sum()]), n=4000, rng=np.random.default_rng(1))
    lo, hi = total_dq.credible_interval(0.9)
    assert lo[0] <= true_total <= hi[0], f"true total {true_total} not in 90% CI [{lo[0]}, {hi[0]}]"

    centroid_dq = posterior.derived_quantity(lambda m: _centroid(m, cells), n=4000, rng=np.random.default_rng(2))
    clo, chi = centroid_dq.credible_interval(0.9)
    assert clo[0] <= true_centroid[0] <= chi[0], f"true centroid x {true_centroid[0]} not in [{clo[0]}, {chi[0]}]"
    assert clo[1] <= true_centroid[1] <= chi[1], f"true centroid y {true_centroid[1]} not in [{clo[1]}, {chi[1]}]"


def test_credible_region_coverage_is_near_nominal_over_repeated_noise_draws():
    # calibration is a repeated-sampling property (mirrors field_inversion_test.py's own convention):
    # average coverage over independent noise realizations rather than trusting a single draw.
    cells = _aquifer_cells()
    prior_kwargs = dict(prior_volume_scale=1.0e4, smoothness_precision=2.0e-4, length_scale=100.0)
    prior = FieldGaussianPrior(
        mean=0.0,
        smoothness_precision=prior_kwargs["smoothness_precision"],
        marginal_precision=1.0 / prior_kwargs["prior_volume_scale"] ** 2,
        length_scale=prior_kwargs["length_scale"],
    )
    true_dv = _true_dewatering_field(cells, seed=9, prior=prior)
    true_total = float(true_dv.sum())
    true_centroid = _centroid(true_dv, cells)

    los_vector = np.array([0.1, -0.05, 0.993])
    obs_xyz, _ = _pixel_grid(25, 400.0)
    clean_disp = poroelastic_subsidence(true_dv, cells, obs_xyz, moduli=MODULI, los_vector=los_vector)
    noise_std = 0.003

    covered_total = covered_cx = covered_cy = 0
    n_trials = 12
    for seed in range(n_trials):
        noisy = clean_disp + np.random.default_rng(seed).normal(0.0, noise_std, clean_disp.shape)
        obs = Observation(
            kind="insar_los",
            location=obs_xyz,
            value=noisy,
            noise_cov=np.full(noisy.shape, noise_std**2),
            provenance={"los_vector": (los_vector / np.linalg.norm(los_vector)).tolist()},
        )
        posterior = invert_deformation([obs], cells, moduli=MODULI, **prior_kwargs)

        total_dq = posterior.derived_quantity(
            lambda m: np.array([m.sum()]), n=2000, rng=np.random.default_rng(1000 + seed)
        )
        lo, hi = total_dq.credible_interval(0.9)
        covered_total += lo[0] <= true_total <= hi[0]

        centroid_dq = posterior.derived_quantity(
            lambda m: _centroid(m, cells), n=2000, rng=np.random.default_rng(2000 + seed)
        )
        clo, chi = centroid_dq.credible_interval(0.9)
        covered_cx += clo[0] <= true_centroid[0] <= chi[0]
        covered_cy += clo[1] <= true_centroid[1] <= chi[1]

    # 90% nominal; require the empirical rate land in a sensible calibrated band (not overconfident).
    assert covered_total / n_trials >= 0.65
    assert covered_cx / n_trials >= 0.65
    assert covered_cy / n_trials >= 0.65
