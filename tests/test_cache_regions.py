"""Local cache metrics must see small/brief changes and reduce unequal SP shards."""
import unittest
import torch
from cache.regions import ProbeRegions


class ProbeRegionTests(unittest.TestCase):
    def test_small_target_not_diluted_by_background_or_time(self):
        mask = torch.zeros(1, 1, 4, 10, 10)
        mask[:, :, 2, 5, 5] = 1
        regions = ProbeRegions(mask, shape=(4, 10, 10))
        previous = torch.ones(1, 400, 8)
        current = previous.clone()
        current[:, 255] = 2
        global_distance = (current - previous).abs().sum() / previous.abs().sum()
        self.assertLess(global_distance, .01)
        self.assertEqual(regions.distance(regions.sums(current, previous)), 1.)

    def test_unequal_shards_aggregate_before_ratio(self):
        torch.manual_seed(42)
        mask = torch.randint(0, 2, (2, 1, 3, 5, 7)).float()
        previous = torch.randn(2, 105, 8)
        current = previous + torch.randn_like(previous) * .1
        full = ProbeRegions(mask, shape=(3, 5, 7)).sums(current, previous)
        pieces = []
        for start, end in ((0, 26), (26, 52), (52, 78), (78, 105)):
            shard = slice(start, end)
            regions = ProbeRegions(mask, shape=(3, 5, 7), shard=shard)
            pieces.append(regions.sums(current[:, shard], previous[:, shard]))
        torch.testing.assert_close(sum(pieces), full, rtol=0, atol=0)

    def test_empty_mask_and_nonfinite(self):
        regions = ProbeRegions(torch.zeros(1, 1, 2, 2, 2), shape=(2, 2, 2))
        previous = torch.ones(1, 8, 2)
        self.assertEqual(regions.distance(regions.sums(previous, previous)), 0.)
        current = previous.clone(); current[:, 0] = float('nan')
        self.assertTrue(torch.isnan(torch.tensor(regions.distance(regions.sums(current, previous)))))

    def test_shape_validation(self):
        with self.assertRaises(ValueError):
            ProbeRegions(None, shape=(2, 2, 2))
        regions = ProbeRegions(torch.zeros(1, 1, 2, 2, 2), shape=(2, 2, 2))
        with self.assertRaises(ValueError):
            regions.sums(torch.ones(1, 7, 2), torch.ones(1, 7, 2))
