"""Environmental field assembler (mixle_pde.env_data): analytic interpolation, differentiability, masking, loaders."""

import importlib
import os
import tempfile
import unittest

import numpy as np

from mixle_pde.env_data import (
    apply_mask,
    assemble_field,
    load_dem,
    load_era5_profile,
    load_gebco,
    load_woa_argo,
    seabed_mask,
)

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def _has(mod):
    return importlib.util.find_spec(mod) is not None


class AssembleAnalyticTest(unittest.TestCase):
    def test_linear_profile_exact_at_nodes(self):
        # A linear profile c(z) = c0 + g z sampled at two depths must reproduce c0 + g z at every grid node to
        # ~1e-10, because linear interpolation is exact for a linear function.
        c0, g = 1500.0, 0.017
        prof_z = np.array([0.0, 200.0])
        prof_c = c0 + g * prof_z
        z_grid = np.linspace(0.0, 200.0, 11)
        r_grid = np.linspace(0.0, 5000.0, 7)
        field = assemble_field(prof_z, prof_c, z_grid, r_grid).reshape(len(z_grid), len(r_grid))
        expected = (c0 + g * z_grid).reshape(-1, 1) * np.ones((1, len(r_grid)))
        self.assertLess(np.max(np.abs(field - expected)), 1e-10)

    def test_range_independent_broadcasts(self):
        # Every range column is identical for a range-independent profile.
        z_grid = np.linspace(0.0, 100.0, 6)
        r_grid = np.linspace(0.0, 1000.0, 4)
        field = assemble_field([0.0, 100.0], [1500.0, 1520.0], z_grid, r_grid).reshape(6, 4)
        for ir in range(4):
            self.assertTrue(np.allclose(field[:, ir], field[:, 0]))

    def test_endpoint_clamp(self):
        # Grid depths past the profile span clamp to the nearest profile endpoint (no extrapolation).
        field = assemble_field([10.0, 20.0], [1500.0, 1510.0], np.array([0.0, 30.0]), np.array([0.0]))
        self.assertAlmostEqual(field[0], 1500.0, places=10)  # above the top sample -> top value
        self.assertAlmostEqual(field[1], 1510.0, places=10)  # below the bottom sample -> bottom value

    def test_range_varying_interpolates(self):
        # Two columns: uniform 1500 at r=0, uniform 1600 at r=1000. A depth-uniform field means the midpoint
        # range r=500 must be the exact analytic average 1550 everywhere in depth.
        prof_z = np.array([0.0, 100.0])
        vals = np.array([[1500.0, 1600.0], [1500.0, 1600.0]])  # (n_depths, n_cols)
        z_grid = np.linspace(0.0, 100.0, 5)
        r_grid = np.array([0.0, 500.0, 1000.0])
        field = assemble_field(prof_z, vals, z_grid, r_grid, ranges=[0.0, 1000.0]).reshape(5, 3)
        self.assertTrue(np.allclose(field[:, 0], 1500.0, atol=1e-10))
        self.assertTrue(np.allclose(field[:, 1], 1550.0, atol=1e-10))
        self.assertTrue(np.allclose(field[:, 2], 1600.0, atol=1e-10))

    def test_range_varying_depth_and_range_bilinear(self):
        # Full bilinear check: value at (z, r) = c00 + gz*z + gr*r reproduced exactly at an interior node.
        c00, gz, gr = 1500.0, 0.02, 0.001
        prof_z = np.array([0.0, 200.0])
        ranges = np.array([0.0, 4000.0])
        vals = np.array([[c00 + gr * r for r in ranges], [c00 + gz * 200.0 + gr * r for r in ranges]])
        z_grid = np.array([50.0])
        r_grid = np.array([1000.0])
        field = assemble_field(prof_z, vals, z_grid, r_grid, ranges=ranges)
        self.assertAlmostEqual(field[0], c00 + gz * 50.0 + gr * 1000.0, places=8)


@unittest.skipUnless(HAS_TORCH, "torch required for differentiability check")
class DifferentiabilityTest(unittest.TestCase):
    def test_gradient_flows_to_control_point(self):
        # A solver-side scalar (sum of the assembled field) must be differentiable w.r.t. a profile control
        # point, and the autograd gradient must match a finite difference.
        prof_z = np.array([0.0, 100.0, 200.0])
        base = np.array([1500.0, 1510.0, 1520.0])
        z_grid = np.linspace(0.0, 200.0, 9)
        r_grid = np.linspace(0.0, 3000.0, 5)

        def loss(vals):
            field = assemble_field(prof_z, vals, z_grid, r_grid)
            return field.pow(2).sum()

        p = torch.tensor(base, dtype=torch.float64, requires_grad=True)
        loss(p).backward()
        grad = p.grad.clone().numpy()

        h = 1e-3
        fd = np.zeros_like(base)
        for i in range(base.size):
            vp = base.copy()
            vp[i] += h
            vm = base.copy()
            vm[i] -= h
            fp = loss(torch.tensor(vp, dtype=torch.float64)).item()
            fm = loss(torch.tensor(vm, dtype=torch.float64)).item()
            fd[i] = (fp - fm) / (2 * h)
        self.assertTrue(np.all(np.isfinite(grad)))
        self.assertTrue(np.allclose(grad, fd, rtol=1e-6, atol=1e-6))

    def test_range_varying_gradient_finite(self):
        prof_z = np.array([0.0, 100.0])
        base = np.array([[1500.0, 1600.0], [1500.0, 1600.0]])
        z_grid = np.linspace(0.0, 100.0, 4)
        r_grid = np.linspace(0.0, 1000.0, 3)
        p = torch.tensor(base, dtype=torch.float64, requires_grad=True)
        field = assemble_field(prof_z, p, z_grid, r_grid, ranges=[0.0, 1000.0])
        field.sum().backward()
        self.assertTrue(np.all(np.isfinite(p.grad.numpy())))
        self.assertGreater(np.abs(p.grad.numpy()).sum(), 0.0)


class MaskingTest(unittest.TestCase):
    def test_sloping_seabed_flags_below(self):
        # A seabed sloping from 50 m at r=0 to 150 m at r=max. A node is below-seabed (masked) exactly when its
        # depth exceeds the local seabed depth.
        z_grid = np.linspace(0.0, 200.0, 21)  # 10 m spacing
        r_grid = np.linspace(0.0, 1000.0, 11)
        D = np.linspace(50.0, 150.0, 11)
        mask = seabed_mask(D, z_grid, r_grid).reshape(21, 11)
        for ir in range(11):
            expected = z_grid > D[ir]
            self.assertTrue(np.array_equal(mask[:, ir], expected))
        # first column: seabed at 50 m -> depths 60..200 masked (15 nodes of 21)
        self.assertEqual(int(mask[:, 0].sum()), int(np.sum(z_grid > 50.0)))

    def test_terrain_flips_sense(self):
        z_grid = np.array([0.0, 10.0, 20.0, 30.0])
        r_grid = np.array([0.0, 1.0])
        D = np.array([15.0, 15.0])  # terrain top at 15 m
        mask = seabed_mask(D, z_grid, r_grid, terrain=True).reshape(4, 2)
        # below terrain (z < 15): depths 0 and 10 masked; 20 and 30 kept
        self.assertTrue(np.array_equal(mask[:, 0], np.array([True, True, False, False])))

    def test_apply_mask_fills_and_preserves(self):
        field = np.arange(6.0)
        mask = np.array([False, True, False, True, False, False])
        out = apply_mask(field, mask, fill=-1.0)
        self.assertTrue(np.array_equal(out, np.array([0.0, -1.0, 2.0, -1.0, 4.0, 5.0])))

    @unittest.skipUnless(HAS_TORCH, "torch required")
    def test_apply_mask_torch_keeps_gradient(self):
        f = torch.arange(6, dtype=torch.float64, requires_grad=True)
        mask = np.array([False, True, False, True, False, False])
        out = apply_mask(f, mask, fill=0.0)
        out.sum().backward()
        # gradient is 1 on kept nodes, 0 on filled nodes
        self.assertTrue(np.array_equal(f.grad.numpy(), (~mask).astype(float)))


class LoaderErrorTest(unittest.TestCase):
    def test_missing_dep_raises_named_error(self):
        # When the optional dep is absent, the loader must raise a clear ImportError naming the backend/extra.
        # If the dep happens to be installed, a bogus path still fails (not silently) -- assert it raises.
        for loader, mod, name in (
            (load_gebco, "xarray", "GEBCO"),
            (load_woa_argo, "xarray", "World Ocean Atlas"),
            (load_dem, "rasterio", "DEM"),
            (load_era5_profile, "xarray", "ERA5"),
        ):
            with self.assertRaises((ImportError, FileNotFoundError, OSError, ValueError)) as cm:
                if loader is load_dem:
                    loader("/no/such/file.tif")
                elif loader is load_gebco:
                    loader("/no/such/file.nc", lon=(0, 1), lat=(0, 1))
                else:
                    loader("/no/such/file.nc", lon=0.0, lat=0.0)
            if not _has(mod):
                # dep missing: must be an ImportError that names the backend module
                self.assertIsInstance(cm.exception, ImportError)
                self.assertIn(mod, str(cm.exception))


@unittest.skipUnless(_has("xarray"), "xarray required for the netCDF loader round-trip")
class NetcdfRoundTripTest(unittest.TestCase):
    def test_woa_argo_profile_round_trip(self):
        # Write a tiny synthetic WOA-style netCDF in-test and round-trip a profile through the loader.
        import xarray as xr

        depth = np.array([0.0, 50.0, 100.0, 200.0])
        temp = np.array([[[20.0, 15.0, 10.0, 4.0]]])  # (lat, lon, depth)
        ds = xr.Dataset(
            {"temperature": (("lat", "lon", "depth"), temp)},
            coords={"lat": [45.0], "lon": [10.0], "depth": depth},
        )
        fd, path = tempfile.mkstemp(suffix=".nc")
        os.close(fd)
        try:
            ds.to_netcdf(path)
            z, v = load_woa_argo(path, lon=10.0, lat=45.0, var="temperature")
            self.assertTrue(np.allclose(z, depth))
            self.assertTrue(np.allclose(v, temp[0, 0]))
            # feed it straight into the assembler
            field = assemble_field(z, v, np.array([25.0]), np.array([0.0]))
            self.assertAlmostEqual(field[0], 17.5, places=10)  # linear between 20 (0 m) and 15 (50 m)
        finally:
            os.remove(path)

    def test_gebco_raster_round_trip(self):
        import xarray as xr

        lon = np.array([9.0, 10.0, 11.0])
        lat = np.array([44.0, 45.0])
        elev = np.array([[-100.0, -200.0, -300.0], [-150.0, -250.0, -350.0]])
        ds = xr.Dataset({"elevation": (("lat", "lon"), elev)}, coords={"lat": lat, "lon": lon})
        fd, path = tempfile.mkstemp(suffix=".nc")
        os.close(fd)
        try:
            ds.to_netcdf(path)
            glon, glat, gelev = load_gebco(path, lon=(9.0, 11.0), lat=(44.0, 45.0))
            self.assertTrue(np.allclose(glon, lon))
            self.assertTrue(np.allclose(glat, lat))
            self.assertTrue(np.allclose(gelev, elev))
        finally:
            os.remove(path)


if __name__ == "__main__":
    unittest.main()
