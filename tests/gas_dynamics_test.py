"""WS-14: compressible Euler gas dynamics -- exact Riemann solver + HLL finite-volume convergence."""

import unittest

import numpy as np

from mixle_pde.gas_dynamics import (
    _star_region,
    engine_cylinder_volume,
    exact_riemann_solution,
    simulate_zero_d_combustion,
    solve_euler_1d,
)


class GasDynamicsTest(unittest.TestCase):
    def test_sod_star_state(self):
        p_star, u_star = _star_region((1.0, 0.0, 1.0), (0.125, 0.0, 0.1), 1.4)
        self.assertAlmostEqual(p_star, 0.30313, places=4)  # the textbook Sod star pressure
        self.assertAlmostEqual(u_star, 0.92745, places=4)  # and star velocity

    def test_exact_solution_endpoints(self):
        # far from the discontinuity the exact solution returns the undisturbed left/right states
        ex = exact_riemann_solution((1.0, 0.0, 1.0), (0.125, 0.0, 0.1), np.array([-0.5, 0.5]), 0.2)
        self.assertTrue(np.allclose(ex[0], [1.0, 0.0, 1.0]))
        self.assertTrue(np.allclose(ex[1], [0.125, 0.0, 0.1]))

    def test_fv_converges_to_exact(self):
        t = 0.2
        errs = {}
        for n in (150, 400):
            x = np.linspace(0.0, 1.0, n)
            dx = x[1] - x[0]
            r0 = np.where(x < 0.5, 1.0, 0.125)
            p0 = np.where(x < 0.5, 1.0, 0.1)
            r, u, p = solve_euler_1d(r0, np.zeros(n), p0, dx, t)
            ex = exact_riemann_solution((1.0, 0.0, 1.0), (0.125, 0.0, 0.1), x - 0.5, t)
            errs[n] = float(np.mean(np.abs(r - ex[:, 0])))
        self.assertLess(errs[400], errs[150])  # convergence under refinement
        self.assertLess(errs[400], 0.015)  # first-order HLL is already close on a fine grid

    def test_positivity(self):
        # density and pressure stay positive through the shock interaction
        n = 400
        x = np.linspace(0.0, 1.0, n)
        r, u, p = solve_euler_1d(
            np.where(x < 0.5, 1.0, 0.125), np.zeros(n), np.where(x < 0.5, 1.0, 0.1), x[1] - x[0], 0.2
        )
        self.assertTrue(np.all(r > 0.0))
        self.assertTrue(np.all(p > 0.0))

    def test_engine_cylinder_volume_bounds(self):
        time = np.array([0.0, 0.005, 0.01])
        volume = engine_cylinder_volume(time, clearance_volume=1.0e-4, swept_volume=9.0e-4, rpm=3000.0)

        self.assertAlmostEqual(volume.min(), 1.0e-4)
        self.assertAlmostEqual(volume.max(), 1.0e-3)

    def test_zero_d_combustion_pressure_rises_and_consumes_fuel(self):
        time = np.linspace(0.0, 0.05, 101)
        result = simulate_zero_d_combustion(
            time,
            initial_temperature=1200.0,
            initial_pressure=101325.0,
            initial_fuel_fraction=0.02,
            volume=1.0e-3,
            heat_release=4.0e7,
            pre_exponential=800.0,
            activation_temperature=6000.0,
        )

        self.assertGreater(result.temperature[-1], result.temperature[0])
        self.assertGreater(result.pressure[-1], result.pressure[0])
        self.assertLess(result.fuel_fraction[-1], result.fuel_fraction[0])
        self.assertTrue(np.all(result.fuel_fraction >= 0.0))


if __name__ == "__main__":
    unittest.main()
