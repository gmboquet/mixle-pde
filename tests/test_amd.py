"""Definition-of-Done test for G5 (tailings seepage + acid-mine-drainage reactive transport).

Verification target: the classical "uniform steady flow, first-order irreversible reaction"
analytical solution used throughout the reactive-transport literature to benchmark sulfide-oxidation
/ AMD column codes (e.g. the plug-flow limit of the van Genuchten & Alves (1982) advection-dispersion-
decay solution; see also Domenico & Schwartz, *Physical and Chemical Hydrogeology*, for the same
steady first-order-decay verification case). For a column fed at a constant sulfide concentration
``C0`` at the inlet, with pseudo-first-order oxidation rate ``k`` and advective velocity ``v``, the
steady-state sulfide profile is ``C0 exp(-k x / v)``; by stoichiometric mass balance the produced
sulfate and acidity profiles are ``product_in + yield * C0 * (1 - exp(-k x / v))``. The column here
is strongly advection-dominated (grid Peclet ~ 1), so the numerical profile should track this
dispersion-free analytic reference to within the loose ``rtol=1e-1`` the work order calls for (the
kinetics themselves are only a first-cut approximation).
"""

import importlib.util
import unittest

import numpy as np

HAS_TORCH = importlib.util.find_spec("torch") is not None

if HAS_TORCH:
    from mixle_pde.dynamics import AdvectionDiffusionOperator
    from mixle_pde.reactive_transport import (
        AMD_SPECIES,
        ReactiveTransport,
        amd_reaction,
        amd_reactions_step,
        effluent_assay,
        effluent_log_likelihood,
        ph_from_concentration,
    )


class AmdReactionUnitTestCase(unittest.TestCase):
    """Unit-level checks on the bare kinetic law, independent of transport."""

    def test_species_order(self):
        self.assertEqual(AMD_SPECIES, ("sulfide", "SO4", "metal", "H"))

    @unittest.skipUnless(HAS_TORCH, "requires PyTorch")
    def test_amd_reaction_shape_and_sign(self):
        state = np.array([[1.0e-3, 1.0e-5, 0.0, 1.0e-7]])
        stoich = {"SO4": 2.0, "Fe": 1.0, "H": 4.0}
        deriv = amd_reaction(state, rate_const=0.05, stoichiometry=stoich)
        self.assertIsInstance(deriv, np.ndarray)
        self.assertEqual(deriv.shape, state.shape)
        # sulfide is consumed, every product is released, in the fixed stoichiometric ratio.
        self.assertLess(deriv[0, 0], 0.0)
        self.assertGreater(deriv[0, 1], 0.0)
        self.assertGreater(deriv[0, 3], 0.0)
        rate = -deriv[0, 0]
        np.testing.assert_allclose(deriv[0, 1], stoich["SO4"] * rate)
        np.testing.assert_allclose(deriv[0, 3], stoich["H"] * rate)

    @unittest.skipUnless(HAS_TORCH, "requires PyTorch")
    def test_amd_reaction_zero_sulfide_is_inert(self):
        state = np.array([0.0, 1.0e-5, 0.0, 1.0e-7])
        deriv = amd_reaction(state, rate_const=0.1, stoichiometry={"SO4": 2.0, "H": 4.0})
        np.testing.assert_allclose(deriv, 0.0)

    def test_ph_from_concentration(self):
        np.testing.assert_allclose(ph_from_concentration(1.0e-4), 4.0)
        np.testing.assert_allclose(ph_from_concentration([1e-3, 1e-7]), [3.0, 7.0])


@unittest.skipUnless(HAS_TORCH, "requires PyTorch")
class AmdColumnBenchmarkTestCase(unittest.TestCase):
    """The Definition-of-Done acceptance test: a 1-D sulfide-oxidation column."""

    def setUp(self):
        self.n_cells = 41
        self.length = 100.0
        self.velocity = 1.0
        self.dispersivity = 0.05
        self.rate_const = 0.03
        self.dt = 1.0
        self.n_steps = 400  # >> the L/v = 100 advective transit time, so the profile is at steady state
        self.stoichiometry = {"SO4": 2.0, "Fe": 1.0, "H": 4.0}  # classical pyrite-oxidation yields
        self.sulfide_in = 1.0e-4
        self.so4_background = 1.0e-5
        self.h_background = 10.0**-6.5  # near-neutral pore water, pH ~ 6.5

        # A G1-style transport operator: DynamicsOperator.transition_matrix(dt) is the only interface
        # ReactiveTransport relies on, so this stands in for G1's GroundwaterTransportOperator (which
        # subclasses AdvectionDiffusionOperator) until that operator lands. Explicit stepping is used
        # because the seepage boundary condition is applied by overwriting the inlet cell *after* the
        # transport update -- correct for an explicit step (which only reads old values), but not for
        # an implicit step (which solves interior and boundary jointly, so a post-hoc overwrite doesn't
        # correct the influence the wrong boundary value already had on the interior solve).
        self.transport = AdvectionDiffusionOperator(
            diffusivity=self.dispersivity,
            velocity=self.velocity,
            n=self.n_cells,
            length=self.length,
            bc="neumann",
            scheme="explicit",
        )
        self.reactions = amd_reactions_step(rate_const=self.rate_const, stoichiometry=self.stoichiometry)
        self.seepage_bc = {
            "index": 0,
            "concentration": [self.sulfide_in, self.so4_background, 0.0, self.h_background],
        }
        self.model = ReactiveTransport(self.transport, self.reactions, seepage_bc=self.seepage_bc)

    def _run_to_steady_state(self):
        state = np.zeros((self.n_cells, 4))
        state[:, 1] = self.so4_background
        state[:, 3] = self.h_background
        state[0] = self.seepage_bc["concentration"]
        for _ in range(self.n_steps):
            state = self.model.step(state, self.dt)
        return state

    def _analytic_profile(self, x):
        decay = 1.0 - np.exp(-self.rate_const * x / self.velocity)
        so4 = self.so4_background + self.stoichiometry["SO4"] * self.sulfide_in * decay
        h = self.h_background + self.stoichiometry["H"] * self.sulfide_in * decay
        return so4, -np.log10(h)

    def test_amd_column_matches_first_order_benchmark(self):
        state = self._run_to_steady_state()
        x = self.transport.grid
        so4_ref, ph_ref = self._analytic_profile(x)

        ph_numeric = ph_from_concentration(state[:, 3])
        so4_numeric = state[:, 1]

        # sample interior cells only -- skip the pinned inlet node and the last couple of cells,
        # where the finite column's open-outflow boundary treatment departs from the semi-infinite
        # analytic reference.
        sample = np.arange(5, self.n_cells - 5, 5)
        np.testing.assert_allclose(so4_numeric[sample], so4_ref[sample], rtol=1e-1)
        np.testing.assert_allclose(ph_numeric[sample], ph_ref[sample], rtol=1e-1)

        # sanity: acid-generating AMD chemistry, so pH must drop (and sulfate rise) downstream.
        self.assertLess(ph_numeric[sample[-1]], ph_numeric[sample[0]])
        self.assertGreater(so4_numeric[sample[-1]], so4_numeric[sample[0]])

    def test_effluent_assay_reuses_geochem_contract(self):
        state = self._run_to_steady_state()
        outlet = np.array([self.n_cells - 1])
        location = np.array([[self.length, 0.0, 0.0]])
        assay = effluent_assay(state, outlet, location, elements=("SO4", "metal"), units="mol/L")
        self.assertEqual(assay.elements, ("SO4", "metal"))
        self.assertEqual(assay.value.shape, (1, 2))
        ll = effluent_log_likelihood(assay, assay.value)
        self.assertTrue(np.isfinite(ll))


if __name__ == "__main__":
    unittest.main()
