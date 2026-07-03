"""Biot-Gassmann fluid substitution (mixle_pde.rock_physics): analytical + round-trip + gradient checks."""

import unittest

import numpy as np

from mixle_pde.rock_physics import (
    fluid_substitute,
    gassmann_kdry,
    gassmann_ksat,
    moduli_from_velocity,
    velocity_from_moduli,
)

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# A clean sandstone (moduli in GPa), the standard Gassmann worked-example regime.
K_MIN = 36.6  # quartz mineral bulk modulus
K_DRY = 15.0  # dry frame bulk modulus
PHI = 0.20  # porosity
K_BRINE = 2.2  # brine bulk modulus
K_GAS = 0.133  # gas bulk modulus


class GassmannModulusTest(unittest.TestCase):
    def test_ksat_matches_published_worked_example(self):
        # Forward Gassmann for the sandstone above, brine-saturated. Hand-computed reference:
        #   num = (1 - 15/36.6)^2, den = 0.2/2.2 + 0.8/36.6 - 15/36.6^2, K_sat = 15 + num/den.
        ref_brine = 18.42912123155869
        ref_gas = 15.229984784918521
        self.assertAlmostEqual(gassmann_ksat(K_DRY, K_MIN, K_BRINE, PHI), ref_brine, places=8)
        self.assertAlmostEqual(gassmann_ksat(K_DRY, K_MIN, K_GAS, PHI), ref_gas, places=8)

    def test_inverse_gassmann_recovers_dry_frame(self):
        # K_dry -> K_sat -> K_dry must be exact (machine precision), the inverse used by substitution.
        K_sat = gassmann_ksat(K_DRY, K_MIN, K_BRINE, PHI)
        K_dry_rec = gassmann_kdry(K_sat, K_MIN, K_BRINE, PHI)
        self.assertAlmostEqual(K_dry_rec, K_DRY, places=12)

    def test_gas_lowers_ksat_below_brine(self):
        # Gassmann sanity: the stiffer (brine) fluid gives a stiffer saturated rock than gas.
        kb = gassmann_ksat(K_DRY, K_MIN, K_BRINE, PHI)
        kg = gassmann_ksat(K_DRY, K_MIN, K_GAS, PHI)
        self.assertGreater(kb, kg)
        self.assertGreater(kg, K_DRY)  # any fluid stiffens the dry frame

    def test_moduli_velocity_roundtrip(self):
        rho = 2.3  # g/cm^3
        Vp, Vs = 3.5, 2.1  # km/s
        K, mu = moduli_from_velocity(Vp, Vs, rho)
        Vp2, Vs2 = velocity_from_moduli(K, mu, rho)
        self.assertAlmostEqual(float(Vp2), Vp, places=10)
        self.assertAlmostEqual(float(Vs2), Vs, places=10)

    def test_vectorized_over_a_field(self):
        # The transform is elementwise, so a porosity/dry-modulus field runs in one call.
        K_dry = np.linspace(10.0, 20.0, 7)
        phi = np.linspace(0.1, 0.3, 7)
        K_sat = gassmann_ksat(K_dry, K_MIN, K_BRINE, phi)
        self.assertEqual(K_sat.shape, (7,))
        for i in range(7):
            self.assertAlmostEqual(K_sat[i], gassmann_ksat(K_dry[i], K_MIN, K_BRINE, phi[i]), places=12)


class FluidSubstitutionTest(unittest.TestCase):
    def setUp(self):
        # A brine-saturated sandstone measurement built from the reference frame, so K_sat is exact.
        self.phi = PHI
        self.K_min = K_MIN
        self.rho_min = 2.65  # quartz g/cm^3
        # A soft reservoir shear modulus, the regime where a gas fill lowers Vp (the bright-spot case).
        # With a very stiff frame the density drop can dominate and Vp rises instead -- still correct
        # Gassmann physics, but the modulus softening is the signature we assert here.
        self.mu = 5.0  # shear modulus GPa (fluid-independent)
        self.rho_brine, self.rho_gas = 1.03, 0.10  # g/cm^3
        K_sat_brine = gassmann_ksat(K_DRY, K_MIN, K_BRINE, PHI)
        # in-situ density: solid frame + brine in the pores
        self.rho = (1.0 - self.phi) * self.rho_min + self.phi * self.rho_brine
        self.Vp, self.Vs = velocity_from_moduli(K_sat_brine, self.mu, self.rho)

    def test_roundtrip_brine_gas_brine(self):
        # Substituting brine -> gas -> brine must recover (Vp, Vs, rho) to ~1e-6.
        Vp_g, Vs_g, rho_g = fluid_substitute(
            self.Vp,
            self.Vs,
            self.rho,
            phi=self.phi,
            K_min=self.K_min,
            rho_min=self.rho_min,
            K_fl_in=K_BRINE,
            rho_fl_in=self.rho_brine,
            K_fl_out=K_GAS,
            rho_fl_out=self.rho_gas,
        )
        Vp_b, Vs_b, rho_b = fluid_substitute(
            Vp_g,
            Vs_g,
            rho_g,
            phi=self.phi,
            K_min=self.K_min,
            rho_min=self.rho_min,
            K_fl_in=K_GAS,
            rho_fl_in=self.rho_gas,
            K_fl_out=K_BRINE,
            rho_fl_out=self.rho_brine,
        )
        self.assertAlmostEqual(float(Vp_b), float(self.Vp), places=6)
        self.assertAlmostEqual(float(Vs_b), float(self.Vs), places=6)
        self.assertAlmostEqual(float(rho_b), float(self.rho), places=6)

    def test_gas_lowers_vp_and_density(self):
        # Producing brine -> gas drops Vp and bulk density (the bright-spot / DHI signature).
        Vp_g, Vs_g, rho_g = fluid_substitute(
            self.Vp,
            self.Vs,
            self.rho,
            phi=self.phi,
            K_min=self.K_min,
            rho_min=self.rho_min,
            K_fl_in=K_BRINE,
            rho_fl_in=self.rho_brine,
            K_fl_out=K_GAS,
            rho_fl_out=self.rho_gas,
        )
        self.assertLess(float(Vp_g), float(self.Vp))
        self.assertLess(float(rho_g), float(self.rho))
        # Vs rises slightly: mu is unchanged but density drops, so Vs = sqrt(mu/rho) goes up.
        self.assertGreater(float(Vs_g), float(self.Vs))

    def test_substitution_matches_direct_gassmann(self):
        # An independent check: substitute to gas, then reconstruct K_sat_gas directly from the known frame.
        Vp_g, Vs_g, rho_g = fluid_substitute(
            self.Vp,
            self.Vs,
            self.rho,
            phi=self.phi,
            K_min=self.K_min,
            rho_min=self.rho_min,
            K_fl_in=K_BRINE,
            rho_fl_in=self.rho_brine,
            K_fl_out=K_GAS,
            rho_fl_out=self.rho_gas,
        )
        K_gas, mu_gas = moduli_from_velocity(Vp_g, Vs_g, rho_g)
        K_sat_gas_ref = gassmann_ksat(K_DRY, K_MIN, K_GAS, PHI)
        self.assertAlmostEqual(float(K_gas), K_sat_gas_ref, places=6)
        self.assertAlmostEqual(float(mu_gas), self.mu, places=6)  # shear unchanged


@unittest.skipUnless(HAS_TORCH, "requires PyTorch")
class DifferentiabilityTest(unittest.TestCase):
    def setUp(self):
        torch.set_default_dtype(torch.float64)
        from mixle_pde.ops import make_ops

        self.ops = make_ops()

    def _base(self):
        rho_min = 2.65
        rho_brine, rho_gas = 1.03, 0.10
        mu = 20.0
        K_sat_brine = gassmann_ksat(K_DRY, K_MIN, K_BRINE, PHI)
        rho = (1.0 - PHI) * rho_min + PHI * rho_brine
        Vp, Vs = velocity_from_moduli(K_sat_brine, mu, rho, ops=self.ops)
        return dict(Vp=float(Vp), Vs=float(Vs), rho=rho, rho_min=rho_min, rho_gas=rho_gas, rho_brine=rho_brine)

    def test_dVp_dsaturation_matches_finite_difference(self):
        # Partial gas saturation via a fluid mix: Woodside (Reuss) average of brine/gas moduli and a
        # linear density mix. dVp_out / dS_gas from autograd must match a central finite difference.
        b = self._base()

        def vp_out(S_gas):
            # Reuss (isostress) fluid mix -- the standard effective pore-fluid modulus for a uniform mix.
            inv_kfl = (1.0 - S_gas) / K_BRINE + S_gas / K_GAS
            K_fl_out = 1.0 / inv_kfl
            rho_fl_out = (1.0 - S_gas) * b["rho_brine"] + S_gas * b["rho_gas"]
            Vp, _, _ = fluid_substitute(
                self.ops.tensor(b["Vp"]),
                self.ops.tensor(b["Vs"]),
                self.ops.tensor(b["rho"]),
                phi=PHI,
                K_min=K_MIN,
                rho_min=b["rho_min"],
                K_fl_in=K_BRINE,
                rho_fl_in=b["rho_brine"],
                K_fl_out=K_fl_out,
                rho_fl_out=rho_fl_out,
                ops=self.ops,
            )
            return Vp

        S = torch.tensor(0.4, requires_grad=True)
        vp = vp_out(S)
        vp.backward()
        g_auto = float(S.grad)
        self.assertTrue(np.isfinite(g_auto))

        eps = 1e-6
        g_fd = (float(vp_out(torch.tensor(0.4 + eps))) - float(vp_out(torch.tensor(0.4 - eps)))) / (2 * eps)
        self.assertAlmostEqual(g_auto, g_fd, places=5)
        # A real fluid effect: Vp genuinely responds to saturation.
        self.assertGreater(abs(g_auto), 1e-3)

    def test_dVp_dphi_matches_finite_difference(self):
        # Reparameterize into porosity: dVp_out / dphi via autograd vs finite difference.
        b = self._base()

        def vp_out(phi):
            Vp, _, _ = fluid_substitute(
                self.ops.tensor(b["Vp"]),
                self.ops.tensor(b["Vs"]),
                self.ops.tensor(b["rho"]),
                phi=phi,
                K_min=K_MIN,
                rho_min=b["rho_min"],
                K_fl_in=K_BRINE,
                rho_fl_in=b["rho_brine"],
                K_fl_out=K_GAS,
                rho_fl_out=b["rho_gas"],
                ops=self.ops,
            )
            return Vp

        phi = torch.tensor(0.20, requires_grad=True)
        vp = vp_out(phi)
        vp.backward()
        g_auto = float(phi.grad)
        self.assertTrue(np.isfinite(g_auto))

        eps = 1e-6
        g_fd = (float(vp_out(torch.tensor(0.20 + eps))) - float(vp_out(torch.tensor(0.20 - eps)))) / (2 * eps)
        self.assertAlmostEqual(g_auto, g_fd, places=5)


if __name__ == "__main__":
    unittest.main()
