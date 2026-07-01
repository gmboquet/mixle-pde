"""Spatial cross-modal RAG over one volume (mixle_pde.reasoning.SpatialFieldStore)."""

import unittest

import numpy as np

from mixle_pde.reasoning import SpatialFieldStore


def _field():
    # 6x6 grid with a smooth bump; noisy per-cell data.
    g = 6
    xs, ys = np.meshgrid(np.arange(g), np.arange(g), indexing="ij")
    cells = np.stack([xs.ravel(), ys.ravel()], axis=1).astype(float)
    truth = np.exp(-((cells[:, 0] - 2.0) ** 2 + (cells[:, 1] - 3.0) ** 2) / 2.0)
    rng = np.random.RandomState(0)
    data = truth + rng.normal(0, 0.05, size=truth.shape)
    return cells, truth, data


class SpatialFieldStoreTest(unittest.TestCase):
    def setUp(self):
        self.cells, self.truth, self.data = _field()
        self.sfs = SpatialFieldStore(self.cells, self.data, tile_radius=1.5, coarse_sd=0.8, fine_sd=0.05)

    def test_tiles_are_local(self):
        # radius 1.5 reaches the diagonal (sqrt2 ~ 1.41): an interior tile is the 3x3 Moore
        # neighbourhood (9 cells); a corner tile is only 4. Tiles stay local, not global.
        interior = self.sfs.nearest_cell([2.0, 3.0])
        corner = self.sfs.nearest_cell([0.0, 0.0])
        self.assertEqual(len(self.sfs.tiles[interior]), 9)
        self.assertEqual(len(self.sfs.tiles[corner]), 4)
        self.assertLess(len(self.sfs.tiles[interior]), self.sfs.n)  # never the whole volume

    def test_location_query_matches_full_solve_at_target(self):
        store = self.sfs.store()
        prior = self.sfs.prior(sd=1.0)
        target = self.sfs.nearest_cell([2.0, 3.0])

        # location-anchored: only tiles near the target, conditioning on raw sub-volumes.
        local, steps = store.assimilate(prior, self.cells[target], k=6, query=[target], epsilon=0.0)
        # full-volume solve: condition on every tile's raw evidence.
        full = prior
        for members in self.sfs.tiles:
            e = self.sfs._fine(members)
            full = full.update(e.H, e.y, e.R)

        local_mean = float(local.mean()[target])
        full_mean = float(full.mean()[target])
        # the local query recovers the target value the full solve gives, within tolerance,
        # without touching the rest of the volume.
        self.assertAlmostEqual(local_mean, full_mean, delta=0.1)
        self.assertAlmostEqual(local_mean, float(self.truth[target]), delta=0.15)
        # raw sub-volumes were actually fetched for this query
        self.assertTrue(any(s.fidelity == "raw" for s in steps))

    def test_local_query_leaves_far_cells_at_prior(self):
        store = self.sfs.store()
        prior = self.sfs.prior(sd=1.0)
        target = self.sfs.nearest_cell([2.0, 3.0])
        far = self.sfs.nearest_cell([5.0, 5.0])
        local, _ = store.assimilate(prior, self.cells[target], k=6, query=[target], epsilon=0.0)
        # a cell far from the query neighbourhood is essentially untouched (still ~prior sd).
        self.assertGreater(local.sd()[far], 0.9)
        self.assertLess(local.sd()[target], 0.5)  # the target got sharpened

    def test_active_retrieval_targets_informative_tile(self):
        store = self.sfs.store()
        prior = self.sfs.prior(sd=1.0)
        target = self.sfs.nearest_cell([2.0, 3.0])
        idx, gain = store.next_evidence(prior, query=[target], fidelity="fine")
        self.assertGreaterEqual(gain, 0.0)
        # the most informative tile for the target sits near it
        self.assertLess(np.linalg.norm(self.cells[idx] - self.cells[target]), 2.0)


if __name__ == "__main__":
    unittest.main()
