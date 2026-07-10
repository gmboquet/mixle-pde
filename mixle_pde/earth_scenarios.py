"""Reusable synthetic 3D/4D Earth inversion harnesses.

These compact scenarios combine geophysical, geochemical, and biostratigraphic
evidence with field priors and sampled posterior updates. They are designed as
deterministic release fixtures and demo inputs, not as real exploration
recommendations. Public reports should preserve their synthetic-data status,
package revisions, metrics, and uncertainty limitations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from mixle_pde.field_assimilation import PosteriorFieldSamples4D, assimilate_4d_joint_linear_dynamics
from mixle_pde.field_inversion import FieldGaussianPrior, linear_gaussian_invert
from mixle_pde.geo_observations import BiostratConstraint, GeochemAssay
from mixle_pde.latent import Field3D, PosteriorField3D, PosteriorFieldSamples3D
from mixle_pde.observations import ForwardOperatorRegistry, Observation, borehole_forward_operator, gravity_forward_operator
from mixle_pde.sample_update import (
    SampleUpdateReport,
    biostrat_constraint_likelihood,
    geochem_assay_likelihood,
    timed_likelihood,
    update_sampled_field_posterior_with_observations,
    update_sampled_field_posterior_4d,
)


@dataclass(frozen=True)
class Synthetic3DInversionResult:
    """Result bundle for a synthetic 3D geophysics + geochemistry inversion."""

    grid: Field3D
    truth: np.ndarray
    geophysical_posterior: PosteriorField3D
    sampled_posterior: PosteriorFieldSamples3D
    geochem_updated_posterior: PosteriorFieldSamples3D
    update_report: SampleUpdateReport
    metrics: dict[str, float]


@dataclass(frozen=True)
class Synthetic4DAssimilationResult:
    """Result bundle for a synthetic 4D age posterior with biostratigraphic evidence."""

    grid: Field3D
    times: np.ndarray
    truth: np.ndarray
    dynamics_posterior: Any
    sampled_posterior: PosteriorFieldSamples4D
    biostrat_updated_posterior: PosteriorFieldSamples4D
    update_report: SampleUpdateReport
    metrics: dict[str, float]


def run_synthetic_3d_geochem_geophysics_inversion(
    *, n_samples: int = 1024, rng: np.random.Generator | None = None
) -> Synthetic3DInversionResult:
    """Run a compact 3D inversion that fuses gravity with a geochemical assay update."""
    rng = np.random.default_rng(123) if rng is None else rng
    grid = Field3D(
        coordinates=np.array(
            [
                [0.0, 0.0, -50.0],
                [100.0, 0.0, -50.0],
                [0.0, 100.0, -50.0],
                [100.0, 100.0, -50.0],
            ]
        ),
        spacing=100.0,
        units="ore_intensity",
        property_name="ore_intensity",
    )
    truth = np.array([0.8, 1.0, 1.2, 3.0])
    registry = ForwardOperatorRegistry()
    volumes = np.full(grid.n, 100.0**3)
    registry.register(gravity_forward_operator(grid.coordinates, volumes))
    registry.register(borehole_forward_operator())
    surface = np.array([[0.0, 0.0, 10.0], [100.0, 0.0, 10.0], [0.0, 100.0, 10.0], [100.0, 100.0, 10.0]])
    gravity_op = registry.get("gravity")
    gravity_obs = Observation(
        "gravity",
        surface,
        gravity_op.jacobian(grid, surface) @ truth,
        np.full(surface.shape[0], 1.0e-4**2),
        units="mGal",
    )
    prior = FieldGaussianPrior(mean=1.0, smoothness_precision=0.02, marginal_precision=0.2, length_scale=120.0)
    geophysical = linear_gaussian_invert(grid, [gravity_obs], registry, prior)
    samples = geophysical.sample(int(n_samples), rng)
    sampled = PosteriorFieldSamples3D(
        grid=grid,
        samples=samples,
        provenance={"method": "synthetic_geophysical_posterior_samples"},
    )
    assay = GeochemAssay(
        element="Cu",
        location=grid.coordinates[[3]],
        value=np.array([truth[3]]),
        noise_std=np.array([0.12]),
        units="relative_grade",
        provenance={"scenario": "synthetic_3d"},
    )
    updated, report = update_sampled_field_posterior_with_observations(
        sampled,
        [geochem_assay_likelihood(assay, grid)],
        n_samples=int(n_samples),
        rng=rng,
    )
    prior_rmse = _rmse(prior.mean_vector(grid), truth)
    geophysics_rmse = _rmse(geophysical.mean, truth)
    updated_rmse = _rmse(updated.mean, truth)
    assay_index = 3
    metrics = {
        "prior_rmse": prior_rmse,
        "geophysical_rmse": geophysics_rmse,
        "geochem_updated_rmse": updated_rmse,
        "assay_cell_geophysical_error": abs(float(geophysical.mean[assay_index] - truth[assay_index])),
        "assay_cell_updated_error": abs(float(updated.mean[assay_index] - truth[assay_index])),
        "geochem_effective_sample_size": report.effective_sample_size,
    }
    return Synthetic3DInversionResult(
        grid=grid,
        truth=truth,
        geophysical_posterior=geophysical,
        sampled_posterior=sampled,
        geochem_updated_posterior=updated,
        update_report=report,
        metrics=metrics,
    )


def run_synthetic_4d_biostrat_assimilation(
    *, n_samples: int = 1024, rng: np.random.Generator | None = None
) -> Synthetic4DAssimilationResult:
    """Run a compact 4D age assimilation and update it with biostratigraphic range evidence."""
    rng = np.random.default_rng(321) if rng is None else rng
    grid = Field3D(
        coordinates=np.array([[0.0, 0.0, -100.0]]),
        spacing=1.0,
        units="Ma",
        property_name="age",
    )
    times = np.array([0.0, 1.0])
    truth = np.array([[35.0], [55.0]])
    registry = ForwardOperatorRegistry()
    registry.register(borehole_forward_operator())
    observations = [
        [Observation("borehole", grid.coordinates, truth[0], np.array([1.0]), time=0.0)],
        [],
    ]
    prior = FieldGaussianPrior(mean=30.0, smoothness_precision=0.0, marginal_precision=0.1)
    dynamics = assimilate_4d_joint_linear_dynamics(
        grid,
        times,
        observations,
        registry,
        prior,
        transitions=np.array([[[1.0]]]),
        process_cov=np.array([[400.0]]),
    )
    sampled = PosteriorFieldSamples4D(
        grid=grid,
        times=times,
        samples=dynamics.sample(int(n_samples), rng),
        provenance={"method": "synthetic_joint_4d_samples"},
    )
    occurrence = BiostratConstraint(
        location=grid.coordinates,
        taxon="Synthetic-Zone",
        present=True,
        first_appearance=60.0,
        last_appearance=50.0,
        tolerance=1.0,
        provenance={"scenario": "synthetic_4d"},
    )
    updated, report = update_sampled_field_posterior_4d(
        sampled,
        [[], [timed_likelihood(biostrat_constraint_likelihood(occurrence, grid), 1.0)]],
        n_samples=int(n_samples),
        rng=rng,
    )
    dynamics_error = abs(float(dynamics.mean_array[-1, 0] - truth[-1, 0]))
    updated_error = abs(float(updated.mean_array[-1, 0] - truth[-1, 0]))
    metrics = {
        "dynamics_final_error": dynamics_error,
        "biostrat_updated_final_error": updated_error,
        "biostrat_updated_final_mean": float(updated.mean_array[-1, 0]),
        "biostrat_effective_sample_size": report.effective_sample_size,
        "joint_start_final_covariance": float(dynamics.cross_covariance(0.0, 1.0)[0, 0]),
    }
    return Synthetic4DAssimilationResult(
        grid=grid,
        times=times,
        truth=truth,
        dynamics_posterior=dynamics,
        sampled_posterior=sampled,
        biostrat_updated_posterior=updated,
        update_report=report,
        metrics=metrics,
    )


def _rmse(values: np.ndarray, truth: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(values, dtype=float) - np.asarray(truth, dtype=float)) ** 2)))
