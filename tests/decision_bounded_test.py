"""A6 DoD: bounded-field ``region_mass``/``derived_quantity`` must route through samples.

A linear functional of the *unconstrained* Gaussian (the pre-fix closed form) is the wrong space for a
bounded field -- porosity lives in ``(0, 1)`` via a logit transform, but the unconstrained mean is an
unbounded real number, not a probability. ``region_mass`` on a bounded posterior must dispatch to the
sample path (:func:`mixle_pde.posterior_query.derived_quantity_physical`) and agree with a large Monte
Carlo reference, while the old unconstrained closed form is grossly wrong.
"""

import unittest

import numpy as np

from mixle_pde.latent import Field3D, PosteriorField3D
from mixle_pde.posterior_query import SampledDerivedQuantity, region_mass


def _grid(n_side=4):
    xs = np.linspace(0.0, 30.0, n_side)
    ys = np.linspace(0.0, 30.0, n_side)
    return np.array([[x, y, 0.0] for y in ys for x in xs], dtype=float)


class BoundedRegionMassTest(unittest.TestCase):
    def test_sample_path_matches_monte_carlo_reference_and_beats_old_closed_form(self):
        coords = _grid()
        grid = Field3D(coordinates=coords, spacing=10.0, units="frac", property_name="porosity", bounds=(0.0, 1.0))
        n = grid.n

        rng = np.random.default_rng(0)
        mean_u = rng.normal(loc=1.5, scale=0.3, size=n)  # unconstrained (logit) mean
        diag_var = np.full(n, 0.3)  # sizeable unconstrained variance -> the Jensen gap is material
        post = PosteriorField3D(grid=grid, mean=mean_u, diag_var=diag_var)

        mask = grid.coordinates[:, 0] < 15.0  # west half of the grid
        vols = np.full(n, 1000.0)  # m^3 per cell
        weights = np.where(mask, vols, 0.0)

        # 200k-draw Monte Carlo reference for sum(vol * porosity) over the region, physical units.
        mc_draws = post.sample(200_000, np.random.default_rng(1))
        reference = float((mc_draws @ weights).mean())

        dq = region_mass(post, mask, vols)
        self.assertIsInstance(dq, SampledDerivedQuantity)
        rel_err = abs(dq.mean - reference) / abs(reference)
        self.assertLess(rel_err, 0.03, f"sample-path region_mass relative error {rel_err:.4f} exceeds 3%")

        # The OLD buggy closed form: a linear functional of the raw unconstrained Gaussian mean, with no
        # bound transform applied at all. Reproduced here (not called through the fixed API) to prove the
        # regression A6 fixes would otherwise be silent.
        old_mean = float(weights @ post.mean)
        old_rel_err = abs(old_mean - reference) / abs(reference)
        self.assertGreater(old_rel_err, 0.20, f"old closed form is only {old_rel_err:.4f} off; weak fixture")


if __name__ == "__main__":
    unittest.main()
