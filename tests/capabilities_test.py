"""PDE modeling capability catalog and readiness checks."""

import unittest

from mixle_pde.capabilities import (
    DEFAULT_REQUIRED_CAPABILITIES,
    assert_required_modeling,
    capability_catalog,
    get_capability,
    missing_required_dependencies,
    readiness_report,
    run_required_modeling_checks,
    run_verification_scenario,
    verification_scenarios,
)


class CapabilityCatalogTest(unittest.TestCase):
    def test_required_capabilities_are_declared(self):
        catalog = {cap.id: cap for cap in capability_catalog()}
        for capability_id in DEFAULT_REQUIRED_CAPABILITIES:
            self.assertIn(capability_id, catalog)
            self.assertGreater(len(catalog[capability_id].solver_symbols), 0)
            self.assertGreater(len(catalog[capability_id].scenario_ids), 0)

    def test_dependency_report_is_structured(self):
        report = readiness_report()
        self.assertEqual(report["required"], list(DEFAULT_REQUIRED_CAPABILITIES))
        self.assertIsInstance(report["missing_dependencies"], dict)
        self.assertEqual(
            {cap["id"] for cap in report["capabilities"] if cap["required_for_release"]},
            set(DEFAULT_REQUIRED_CAPABILITIES),
        )

    def test_unknown_capability_raises(self):
        with self.assertRaises(KeyError):
            get_capability("not-a-capability")


class VerificationScenarioTest(unittest.TestCase):
    def test_every_declared_scenario_exists(self):
        scenario_ids = {scenario.id for scenario in verification_scenarios()}
        for capability in capability_catalog():
            for scenario_id in capability.scenario_ids:
                self.assertIn(scenario_id, scenario_ids)

    def test_lightweight_scenarios_pass(self):
        # These scenarios cover the release-critical PDE modeling surfaces without running the full suite.
        for scenario_id in (
            "mesh_3d_4d_measure",
            "earth_observation_likelihoods",
            "earth_forward_operator_contract",
            "earth_static_linear_inversion",
            "earth_bounded_gauss_newton",
            "earth_dc_resistivity_nonlinear_inversion",
            "earth_aem_layered_nonlinear_observation",
            "earth_layered_mt_nonlinear_inversion",
            "earth_mt_2d_te_nonlinear_observation",
            "earth_mt_3d_nonlinear_observation",
            "earth_csem_3d_nonlinear_observation",
            "earth_4d_assimilation",
            "earth_ensemble_4d_nonlinear_assimilation",
            "earth_prior_coupling",
            "earth_sparse_prior_precision",
            "earth_sparse_posterior_factorization",
            "earth_posterior_extraction",
            "earth_posterior_calibration",
            "heat_fourier_decay",
            "gravity_linearity",
        ):
            result = run_verification_scenario(scenario_id)
            self.assertTrue(result.passed, result.message)
            self.assertEqual(result.id, scenario_id)

    def test_required_modeling_checks_pass_when_dependencies_are_available(self):
        missing = missing_required_dependencies()
        if missing:
            self.skipTest(f"missing dependencies: {missing}")
        results = run_required_modeling_checks()
        self.assertEqual({result.capability_id for result in results}, set(DEFAULT_REQUIRED_CAPABILITIES))
        self.assertTrue(all(result.passed for result in results), [result.as_dict() for result in results])
        self.assertEqual(assert_required_modeling(), results)


if __name__ == "__main__":
    unittest.main()
