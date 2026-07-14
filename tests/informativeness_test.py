"""Data-informativeness diagnostic (workstream A2): does the data or the prior set the width?

Builds a gravity(+shallow-borehole) inversion over a two-layer grid: a shallow layer sampled directly by
boreholes, and a deep layer gravity alone cannot resolve at any usable signal-to-noise. The deep layer's
posterior narrows only because the smoothness prior links it to better-constrained neighbours -- it is
"prior dominated" -- while the shallow layer is genuinely data-driven. Checks that `variance_reduction`
/ `prior_dominated_mask` see that split, and that `region_mass`'s `prior_dominated` flag reports it, so a
driller-facing mass/tonnage number is never emitted without the honesty flag work-plan A2 requires.
"""

import unittest

import numpy as np

from mixle_pde.field_inversion import FieldGaussianPrior, linear_gaussian_invert
from mixle_pde.informativeness import (
    prior_dominated_mask,
    prior_marginal_variance,
    region_prior_dominated,
    variance_reduction,
)
from mixle_pde.latent import Field3D
from mixle_pde.observations import (
    ForwardOperatorRegistry,
    Observation,
    borehole_forward_operator,
    gravity_forward_operator,
)
from mixle_pde.posterior_query import region_mass


def _two_layer_grid():
    """A shallow layer (directly sampled by boreholes) over a deep layer gravity alone cannot resolve."""
    xs = np.linspace(0.0, 100.0, 6)
    ys = np.linspace(0.0, 100.0, 6)
    shallow = np.array([[x, y, -10.0] for y in ys for x in xs], dtype=float)
    deep = np.array([[x, y, -2000.0] for y in ys for x in xs], dtype=float)
    coords = np.vstack([shallow, deep])
    grid = Field3D(coordinates=coords, spacing=20.0, units="kg/m^3", property_name="density_contrast", bounds=None)
    shallow_mask = np.zeros(grid.n, dtype=bool)
    shallow_mask[: shallow.shape[0]] = True
    return grid, shallow_mask, ~shallow_mask


class InformativenessDiagnosticTest(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(0)
        self.grid, self.shallow_mask, self.deep_mask = _two_layer_grid()
        self.volumes = np.full(self.grid.n, 20.0**3, dtype=float)

        self.registry = ForwardOperatorRegistry()
        self.registry.register(gravity_forward_operator(self.grid.coordinates, self.volumes))
        self.registry.register(borehole_forward_operator())

        gx, gy = np.meshgrid(np.linspace(0, 100, 5), np.linspace(0, 100, 5))
        grav_loc = np.column_stack([gx.ravel(), gy.ravel(), np.full(gx.size, 5.0)])
        jac = self.registry.get("gravity").jacobian(self.grid, grav_loc)
        grav_noise = 1.0e-3
        truth = np.zeros(self.grid.n)
        grav_val = jac @ truth + self.rng.normal(0, grav_noise, size=jac.shape[0])
        self.gravity = Observation(
            kind="gravity", location=grav_loc, value=grav_val, noise_cov=np.full(jac.shape[0], grav_noise**2)
        )

        # boreholes sample the shallow layer directly; the deep layer only ever sees gravity's
        # ill-posed, noise-swamped signal at 2 km range.
        bore_noise = 5.0
        n_shallow = int(self.shallow_mask.sum())
        bore_val = truth[self.shallow_mask] + self.rng.normal(0, bore_noise, size=n_shallow)
        self.borehole = Observation(
            kind="borehole",
            location=self.grid.coordinates[self.shallow_mask],
            value=bore_val,
            noise_cov=np.full(n_shallow, bore_noise**2),
        )

        self.prior = FieldGaussianPrior(
            mean=0.0, smoothness_precision=2.0e-3, marginal_precision=1.0e-5, length_scale=25.0, neighbors=6
        )
        self.posterior = linear_gaussian_invert(self.grid, [self.gravity, self.borehole], self.registry, self.prior)
        self.prior_var = prior_marginal_variance(self.prior, self.grid)
        self.posterior_var = self.posterior.marginal_variance

    def test_deep_data_starved_cells_show_low_variance_reduction(self):
        reduction = variance_reduction(self.prior_var, self.posterior_var)
        self.assertLess(float(np.mean(reduction[self.deep_mask])), 0.1)
        self.assertGreater(float(np.mean(reduction[self.shallow_mask])), 0.5)

    def test_prior_dominated_mask_flags_the_deep_layer_only(self):
        mask = prior_dominated_mask(self.prior_var, self.posterior_var)
        self.assertTrue(np.all(mask[self.deep_mask]))
        self.assertFalse(np.any(mask[self.shallow_mask]))

    def test_region_prior_dominated_matches_the_per_cell_diagnostic(self):
        deep_weights = np.where(self.deep_mask, self.volumes, 0.0)
        shallow_weights = np.where(self.shallow_mask, self.volumes, 0.0)
        self.assertTrue(region_prior_dominated(self.prior_var, self.posterior_var, deep_weights))
        self.assertFalse(region_prior_dominated(self.prior_var, self.posterior_var, shallow_weights))

    def test_region_mass_carries_the_prior_dominated_flag(self):
        deep_quantity = region_mass(
            self.posterior, self.deep_mask, self.volumes, prior_var=self.prior_var, posterior_var=self.posterior_var
        )
        shallow_quantity = region_mass(
            self.posterior,
            self.shallow_mask,
            self.volumes,
            prior_var=self.prior_var,
            posterior_var=self.posterior_var,
        )
        self.assertIs(deep_quantity.prior_dominated, True)
        self.assertIs(shallow_quantity.prior_dominated, False)

    def test_region_mass_default_flag_preserves_existing_behaviour(self):
        # no prior_var/posterior_var supplied -> unchanged, pre-A2 call sites keep working
        quantity = region_mass(self.posterior, self.deep_mask, self.volumes)
        self.assertFalse(quantity.prior_dominated)


if __name__ == "__main__":
    unittest.main()
