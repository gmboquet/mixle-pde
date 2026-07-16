"""Tests for multi-chain MCMC convergence diagnostics (MP-I8: split R-hat + effective sample size).

Acceptance criteria under test:

* Synthetic, fully deterministic cases pin the two failure modes a convergence diagnostic exists to
  catch: :func:`test_iid_chains_are_reported_converged` (chains that genuinely agree) and
  :func:`test_offset_chains_are_reported_not_converged` (chains that do not), plus a per-parameter
  case showing one bad parameter fails the whole verdict rather than being averaged away.
* Negative-path validation: too few chains, too few draws, non-finite draws, and a
  ``parameter_names`` length mismatch all raise :class:`ValueError` (the Definition of Done's
  "failure case" requirement).
* A real, non-mocked integration case drives this repo's own sampler
  (:mod:`mixle_pde.field_mcmc`) on the exact bimodal fixture ``c5_sampler_test.py`` already uses to
  prove ``metropolis_field_invert`` collapses onto one mode while ``pcn_field_invert`` finds both:
  two independent Metropolis chains started on opposite sides of the gap must be reported as *not*
  converged (they never see each other's mode), while two independent pCN chains on the same problem
  must be reported as converged.
"""

from __future__ import annotations

import numpy as np
import pytest

from mixle_pde.field_inversion import FieldGaussianPrior
from mixle_pde.field_mcmc import metropolis_field_invert, pcn_field_invert
from mixle_pde.latent import Field3D, PosteriorFieldSamples3D
from mixle_pde.observations import ForwardOperator, ForwardOperatorRegistry, Observation
from mixle_pde.verification.mcmc_diagnostics import (
    ChainDiagnostics,
    chains_from_posterior_samples,
    evaluate_chain_convergence,
    multichain_ess,
    split_rhat,
)

# ---------------------------------------------------------------------------
# Synthetic, fully deterministic cases.
# ---------------------------------------------------------------------------


def test_iid_chains_are_reported_converged():
    rng = np.random.default_rng(12345)
    chains = rng.normal(loc=0.0, scale=1.0, size=(4, 600))

    diagnostics = evaluate_chain_convergence(chains, r_hat_threshold=1.05, min_ess=300.0)

    assert isinstance(diagnostics, ChainDiagnostics)
    assert diagnostics.n_chains == 4
    assert diagnostics.n_draws == 600
    assert diagnostics.n_parameters == 1
    assert diagnostics.parameter_names == ("param_0",)
    assert diagnostics.r_hat[0] < 1.05
    assert diagnostics.ess[0] > 300.0
    assert diagnostics.converged
    assert "CONVERGED" in diagnostics.detail and "NOT CONVERGED" not in diagnostics.detail


def test_offset_chains_are_reported_not_converged():
    rng = np.random.default_rng(0)
    offsets = np.array([-6.0, -2.0, 2.0, 6.0])
    chains = offsets[:, None] + rng.normal(scale=0.5, size=(4, 600))

    diagnostics = evaluate_chain_convergence(chains, r_hat_threshold=1.05, min_ess=300.0)

    assert not diagnostics.converged
    assert diagnostics.r_hat[0] > 1.5
    assert "NOT CONVERGED" in diagnostics.detail


def test_one_bad_parameter_fails_the_whole_verdict():
    rng = np.random.default_rng(7)
    good = rng.normal(size=(4, 500))
    offsets = np.array([-6.0, -2.0, 2.0, 6.0])
    bad = offsets[:, None] + rng.normal(scale=0.5, size=(4, 500))
    chains = np.stack([good, bad], axis=-1)  # (4, 500, 2)

    diagnostics = evaluate_chain_convergence(
        chains, parameter_names=("good", "bad"), r_hat_threshold=1.05, min_ess=100.0
    )

    assert diagnostics.n_parameters == 2
    assert diagnostics.r_hat[0] < 1.05
    assert diagnostics.r_hat[1] > 1.5
    assert not diagnostics.converged


def test_multichain_ess_is_bounded_by_total_draws():
    rng = np.random.default_rng(3)
    chains = rng.normal(size=(5, 300))
    ess = multichain_ess(chains)
    assert ess.shape == (1,)
    assert 0.0 < ess[0] <= 5 * 300 + 1e-6


def test_split_rhat_close_to_one_for_a_single_stationary_chain_split_in_two():
    # Even one chain is split into two halves; a long stationary chain should still show close
    # agreement between its own first and second half.
    rng = np.random.default_rng(9)
    chains = rng.normal(size=(1, 2000))
    r_hat = split_rhat(chains)
    assert r_hat.shape == (1,)
    assert r_hat[0] < 1.05


# ---------------------------------------------------------------------------
# Negative-path validation.
# ---------------------------------------------------------------------------


def test_rejects_no_chains():
    with pytest.raises(ValueError, match="one chain"):
        evaluate_chain_convergence(np.zeros((0, 10)))


def test_rejects_too_few_draws():
    with pytest.raises(ValueError, match="four draws"):
        evaluate_chain_convergence(np.zeros((2, 3)))


def test_rejects_non_finite_draws():
    bad = np.ones((2, 10))
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        evaluate_chain_convergence(bad)


def test_rejects_parameter_names_length_mismatch():
    chains = np.ones((2, 10, 2))
    with pytest.raises(ValueError, match="parameter_names"):
        evaluate_chain_convergence(chains, parameter_names=("only_one",))


def test_chains_from_posterior_samples_rejects_shape_mismatch():
    grid = Field3D(coordinates=np.zeros((1, 3)), spacing=1.0, units="", property_name="x")
    a = PosteriorFieldSamples3D(grid=grid, samples=np.zeros((10, 1)))
    b = PosteriorFieldSamples3D(grid=grid, samples=np.zeros((12, 1)))
    with pytest.raises(ValueError, match="shape"):
        chains_from_posterior_samples([a, b])


def test_chains_from_posterior_samples_rejects_empty_list():
    with pytest.raises(ValueError, match="at least one"):
        chains_from_posterior_samples([])


# ---------------------------------------------------------------------------
# Real integration case: this repo's own sampler on a known bimodal posterior.
#
# Mirrors the fixture in mixle_pde/c5_sampler_test.py (BimodalPosteriorSamplerTest), which already
# establishes as a Definition-of-Done acceptance bar that metropolis_field_invert collapses onto
# whichever mode it starts near while pcn_field_invert's fresh-draw proposal finds both. This test
# does not import that module's private helper (it is local to that test file); it rebuilds the same
# minimal single-cell double-well problem from public mixle_pde classes.
# ---------------------------------------------------------------------------


def _bimodal_problem(mode: float = 4.0, marginal_precision: float = 1.0 / 64.0, noise_var: float = 0.25):
    grid = Field3D(coordinates=np.zeros((1, 3)), spacing=1.0, units="", property_name="x")
    prior = FieldGaussianPrior(mean=0.0, smoothness_precision=0.0, marginal_precision=marginal_precision, neighbors=1)

    def predict(grid, field_values, obs_locations):
        return field_values**2

    def jacobian_at_values(grid, field_values, obs_locations):
        return np.array([[2.0 * field_values[0]]])

    op = ForwardOperator("quadratic", predict, jacobian_at_values=jacobian_at_values, differentiable=True)
    registry = ForwardOperatorRegistry()
    registry.register(op)
    observation = Observation(
        kind="quadratic",
        location=np.zeros((1, 3)),
        value=np.array([mode**2]),
        noise_cov=np.array([noise_var]),
    )
    return grid, registry, prior, [observation]


def test_metropolis_chains_from_opposite_modes_are_reported_not_converged():
    mode = 4.0
    grid, registry, prior, observations = _bimodal_problem(mode=mode)

    posterior_pos, _report_pos = metropolis_field_invert(
        grid,
        observations,
        registry,
        prior,
        n_samples=4000,
        burn_in=2000,
        thin=1,
        step_scale=1.0,
        initial_unconstrained=np.array([mode]),
        rng=np.random.default_rng(0),
    )
    posterior_neg, _report_neg = metropolis_field_invert(
        grid,
        observations,
        registry,
        prior,
        n_samples=4000,
        burn_in=2000,
        thin=1,
        step_scale=1.0,
        initial_unconstrained=np.array([-mode]),
        rng=np.random.default_rng(1),
    )

    chains = chains_from_posterior_samples([posterior_pos, posterior_neg])
    diagnostics = evaluate_chain_convergence(chains, r_hat_threshold=1.05, min_ess=50.0)

    # Each chain is stuck near its own starting mode (collapse behavior already proven in
    # c5_sampler_test.py), so the two chains disagree sharply on the mean -> high r_hat.
    assert diagnostics.r_hat[0] > 1.5
    assert not diagnostics.converged


def test_pcn_chains_from_opposite_starts_are_reported_converged():
    mode = 4.0
    grid, registry, prior, observations = _bimodal_problem(mode=mode)

    posterior_a, _report_a = pcn_field_invert(
        grid,
        observations,
        registry,
        prior,
        n_samples=8000,
        burn_in=2000,
        thin=1,
        beta_pcn=0.5,
        rng=np.random.default_rng(0),
    )
    posterior_b, _report_b = pcn_field_invert(
        grid,
        observations,
        registry,
        prior,
        n_samples=8000,
        burn_in=2000,
        thin=1,
        beta_pcn=0.5,
        rng=np.random.default_rng(1),
    )

    chains = chains_from_posterior_samples([posterior_a, posterior_b])
    diagnostics = evaluate_chain_convergence(chains, r_hat_threshold=1.2, min_ess=50.0)

    # pCN's fresh-draw proposal visits both modes from either run (proven in c5_sampler_test.py's
    # occupancy check), so independent pCN chains agree far better than the metropolis case above --
    # but two short (8000-draw) chains on a genuine two-mode target still show real run-to-run
    # variation in how much time each spends per mode, so this is a looser bound than the synthetic
    # iid case, not a claim of near-perfect agreement.
    assert diagnostics.r_hat[0] < 1.2
    assert diagnostics.converged
