"""Definition-of-Done test for E1: `PosteriorField3D` conforms to the shared IC-1 `Posterior`
protocol (`mixle.reason.posterior_protocol.Posterior`).

Fits a small linear-Gaussian field posterior the same way any real caller would (`linear_gaussian_invert`
over a borehole observation), then exercises exactly the IC-1 surface: `isinstance` against the frozen
protocol, the plural `samples(n, rng)` method, the reconciled `credible_interval(level)`, and
`derived_quantity(fn, n, rng)` carrying the `prior_dominated` honesty flag.
"""

import numpy as np
from mixle.reason.posterior_protocol import Posterior

from mixle_pde.field_inversion import FieldGaussianPrior, linear_gaussian_invert
from mixle_pde.latent import Field3D
from mixle_pde.observations import ForwardOperatorRegistry, Observation, borehole_forward_operator


def _fit_small_posterior():
    coords = np.array([[float(i), 0.0, -10.0] for i in range(6)])
    grid = Field3D(coordinates=coords, spacing=1.0, units="ppm", property_name="cu_ppm")
    registry = ForwardOperatorRegistry()
    registry.register(borehole_forward_operator())
    truth = np.array([2.0, 3.0, 4.0, 8.0, 12.0, 16.0])
    obs = Observation(
        kind="borehole",
        location=grid.coordinates[[0, 2, 4]],
        value=truth[[0, 2, 4]],
        noise_cov=np.full(3, 0.25),
    )
    prior = FieldGaussianPrior(mean=0.0, smoothness_precision=1.0e-3, marginal_precision=1.0e-4)
    return linear_gaussian_invert(grid, [obs], registry, prior)


def test_fitted_field_posterior_conforms_to_ic1_posterior_protocol():
    fit = _fit_small_posterior()

    assert isinstance(fit, Posterior)

    rng = np.random.default_rng(0)
    draws = fit.samples(16, rng)
    assert draws.shape == (16, fit.grid.n)

    lo, hi = fit.credible_interval(0.9)
    assert np.all(lo <= hi)

    dq = fit.derived_quantity(lambda m: m.sum(1), 64, rng)
    assert dq.prior_dominated in (True, False)
    assert dq.samples.shape == (64,)
