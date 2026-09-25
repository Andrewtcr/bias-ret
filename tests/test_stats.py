"""Check the resampling helpers behind the paper's bootstrap CIs and permutation p-values."""
import unittest

import numpy as np

from src.stats import cluster_bootstrap_mean, signflip_permutation_p


N_PERM = 999


class SignFlipPermutationTests(unittest.TestCase):
    def test_p_value_counts_the_observed_sign_pattern_as_one_permutation(self):
        # p = (b + 1) / (B + 1): a finite permutation test can never support p = 0.
        # Forty equal deltas are matched only by an all-same-sign draw (odds 2^-39), so b = 0.
        floor = signflip_permutation_p(np.ones(40), n_perm=N_PERM, seed=0)
        self.assertEqual(floor["p_two_sided"], 1 / (N_PERM + 1))
        # A zero observed mean is matched by every draw, so b = B.
        ceiling = signflip_permutation_p(np.array([-2.0, -1.0, 1.0, 2.0] * 10), n_perm=N_PERM, seed=0)
        self.assertEqual(ceiling["p_two_sided"], 1.0)

    def test_shifted_deltas_are_significant_and_symmetric_deltas_are_not(self):
        rng = np.random.default_rng(1)
        shifted = rng.normal(1.0, 1.0, size=50)
        self.assertLess(signflip_permutation_p(shifted, n_perm=N_PERM, seed=0)["p_two_sided"], 0.01)
        noise = rng.normal(0.0, 1.0, size=25)
        symmetric = np.concatenate([noise, -noise])
        self.assertGreater(signflip_permutation_p(symmetric, n_perm=N_PERM, seed=0)["p_two_sided"], 0.5)

    def test_fixed_seed_reproduces_the_result(self):
        # Paper p-values are regenerated from committed data, so the seed must pin them.
        deltas = np.random.default_rng(2).normal(0.2, 1.0, size=30)
        self.assertEqual(signflip_permutation_p(deltas, n_perm=N_PERM, seed=42),
                         signflip_permutation_p(deltas, n_perm=N_PERM, seed=42))


class ClusterBootstrapTests(unittest.TestCase):
    def setUp(self):
        # 10 clusters of 5 identical observations: 50 values, only 10 independent ones.
        self.values = np.repeat(np.random.default_rng(0).normal(size=10), 5)
        self.clusters = np.repeat(np.arange(10), 5)

    def test_point_estimate_is_the_sample_mean_inside_a_reproducible_ci(self):
        mean, low, high = cluster_bootstrap_mean(self.values, self.clusters, n_boot=2000, seed=42)
        self.assertEqual(mean, self.values.mean())
        self.assertLess(low, mean)
        self.assertLess(mean, high)
        self.assertEqual((mean, low, high),
                         cluster_bootstrap_mean(self.values, self.clusters, n_boot=2000, seed=42))
        # The paper reports 95% intervals: the width should be about 2 * 1.96 cluster SEs.
        se = np.unique(self.values).std() / np.sqrt(10)
        self.assertTrue(0.9 < (high - low) / (2 * 1.96 * se) < 1.1)

    def test_point_estimate_pools_observations_across_unequal_clusters(self):
        # One cluster of a single 0 and one of nine 1s: the pooled mean is 0.9,
        # while a mean of cluster means would give 0.5.
        values = np.array([0.0] + [1.0] * 9)
        clusters = np.array([0] + [1] * 9)
        mean, _, _ = cluster_bootstrap_mean(values, clusters, n_boot=200, seed=0)
        self.assertAlmostEqual(mean, 0.9)

    def test_resampling_whole_clusters_widens_the_ci(self):
        # Treating correlated observations as independent would understate the
        # uncertainty; here the true width ratio is about sqrt(5).
        _, low, high = cluster_bootstrap_mean(self.values, self.clusters, n_boot=2000, seed=0)
        _, ind_low, ind_high = cluster_bootstrap_mean(self.values, np.arange(50), n_boot=2000, seed=0)
        self.assertGreater(high - low, 1.5 * (ind_high - ind_low))


if __name__ == "__main__":
    unittest.main()
