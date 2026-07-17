"""Rock-physics uncertainty propagation (mixle_pde.rock_physics): Monte Carlo Gassmann + sensitivity.

A single Gassmann fluid substitution takes point-valued inputs (mineral modulus, in-situ and target
fluid moduli, porosity) and returns a point-valued (Vp, Vs, rho). In the field none of those inputs
are known exactly, so a bare `fluid_substitute` call silently hides how much a reservoir prediction can
move on plausible rock-physics uncertainty. These tests check the Monte Carlo wrapper actually widens
with input uncertainty (rather than, say, silently collapsing to the deterministic point estimate) and
that the finite-difference sensitivity helper reports a genuinely nonzero response.
"""

import unittest

import numpy as np

from mixle_pde.rock_physics import (
    fluid_modulus_sensitivity,
    fluid_substitute_uncertain,
    gassmann_ksat,
    velocity_from_moduli,
)

# Same clean-sandstone worked example as rock_physics_test.py.
K_MIN = 36.6  # quartz mineral bulk modulus, GPa
K_DRY = 15.0  # dry frame bulk modulus, GPa
PHI = 0.20  # porosity
K_BRINE = 2.2  # brine bulk modulus, GPa
K_GAS = 0.133  # gas bulk modulus, GPa
RHO_MIN = 2.65  # quartz density, g/cm^3
RHO_BRINE = 1.03  # g/cm^3
RHO_GAS = 0.10  # g/cm^3
MU = 5.0  # shear modulus, GPa (fluid-independent)


def _brine_measurement():
    """A brine-saturated measurement built from the reference frame, so K_sat is exact."""
    K_sat_brine = gassmann_ksat(K_DRY, K_MIN, K_BRINE, PHI)
    rho = (1.0 - PHI) * RHO_MIN + PHI * RHO_BRINE
    Vp, Vs = velocity_from_moduli(K_sat_brine, MU, rho)
    return float(Vp), float(Vs), float(rho)


class FluidSubstituteUncertainTest(unittest.TestCase):
    def _run(self, k_fl_out_std, n=4096, seed=0):
        Vp, Vs, rho = _brine_measurement()
        return fluid_substitute_uncertain(
            Vp,
            Vs,
            rho,
            phi=PHI,
            K_min=K_MIN,
            rho_min=RHO_MIN,
            K_fl_in=K_BRINE,
            rho_fl_in=RHO_BRINE,
            K_fl_out=K_GAS,
            rho_fl_out=RHO_GAS,
            priors={"K_fl_out": (K_GAS, k_fl_out_std)},
            n=n,
            rng=np.random.default_rng(seed),
        )

    def test_nonzero_prior_std_gives_nonzero_output_std(self):
        out = self._run(k_fl_out_std=0.02)
        self.assertEqual(out["Vp"].shape, (4096,))
        self.assertEqual(out["Vs"].shape, (4096,))
        self.assertEqual(out["rho"].shape, (4096,))
        self.assertGreater(float(np.std(out["Vp"])), 0.0)

    def test_doubling_prior_std_strictly_increases_output_std(self):
        out_small = self._run(k_fl_out_std=0.02)
        out_large = self._run(k_fl_out_std=0.04)
        self.assertGreater(float(np.std(out_large["Vp"])), float(np.std(out_small["Vp"])))

    def test_no_priors_collapses_to_the_deterministic_point(self):
        # Nothing in priors -> every draw uses the passed-in nominal values, so the ensemble has zero
        # spread and matches the deterministic fluid_substitute call exactly.
        from mixle_pde.rock_physics import fluid_substitute

        Vp, Vs, rho = _brine_measurement()
        out = fluid_substitute_uncertain(
            Vp,
            Vs,
            rho,
            phi=PHI,
            K_min=K_MIN,
            rho_min=RHO_MIN,
            K_fl_in=K_BRINE,
            rho_fl_in=RHO_BRINE,
            K_fl_out=K_GAS,
            rho_fl_out=RHO_GAS,
            priors={},
            n=64,
            rng=np.random.default_rng(1),
        )
        self.assertAlmostEqual(float(np.std(out["Vp"])), 0.0, places=10)
        Vp_ref, Vs_ref, rho_ref = fluid_substitute(
            Vp,
            Vs,
            rho,
            phi=PHI,
            K_min=K_MIN,
            rho_min=RHO_MIN,
            K_fl_in=K_BRINE,
            rho_fl_in=RHO_BRINE,
            K_fl_out=K_GAS,
            rho_fl_out=RHO_GAS,
        )
        self.assertAlmostEqual(float(out["Vp"][0]), float(Vp_ref), places=8)


class FluidModulusSensitivityTest(unittest.TestCase):
    def test_sensitivity_is_nonzero_and_complete(self):
        Vp, Vs, rho = _brine_measurement()
        sens = fluid_modulus_sensitivity(
            Vp,
            Vs,
            rho,
            phi=PHI,
            K_min=K_MIN,
            rho_min=RHO_MIN,
            K_fl_in=K_BRINE,
            rho_fl_in=RHO_BRINE,
            K_fl_out=K_GAS,
            rho_fl_out=RHO_GAS,
        )
        self.assertIn("Vp", sens)
        self.assertIn("Vs", sens)
        self.assertIn("rho", sens)
        self.assertNotEqual(sens["Vp"], 0.0)


if __name__ == "__main__":
    unittest.main()
