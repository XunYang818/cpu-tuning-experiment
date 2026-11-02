"""Scientific protocol invariants; runs entirely in memory."""

import sys

sys.dont_write_bytecode = True
import unittest
from types import SimpleNamespace
import numpy as np
from cpu_tuning.features import aggregate_features
from cpu_tuning.protocol import resample_ids, check_splits
from cpu_tuning.mlp import prepare


class ProtocolTests(unittest.TestCase):
    def test_split_overlap_even_with_complete_sample_coverage(self):
        check_splits([np.array([0, 1]), np.array([2]), np.array([3])], np.arange(4), 4)
        with self.assertRaises(ValueError):
            check_splits(
                [np.array([0, 1]), np.array([1, 2]), np.array([3])], np.arange(4), 4
            )

    def test_duplicate_directions_and_self_loops_are_binary(self):
        X = np.array([[0.0], [2.0], [9.0]], dtype=np.float32)
        e = np.array([[0, 1], [1, 0], [0, 1], [0, 0]])
        h1, h2, info = aggregate_features(X, e)
        np.testing.assert_allclose(h1, [[1.0], [1.0], [9.0]])
        np.testing.assert_allclose(h2, h1)
        self.assertEqual(info["isolates_before_self_loops"], 1)
        self.assertEqual(info["undirected_edges_without_self"], 1)

    def test_second_hop_and_invalid_node_ids(self):
        X = np.array([[0.0], [3.0], [6.0]], dtype=np.float32)
        h1, h2, _ = aggregate_features(X, np.array([[0, 1], [1, 2]]))
        np.testing.assert_allclose(h1, [[1.5], [3.0], [4.5]])
        np.testing.assert_allclose(h2, [[2.25], [3.0], [3.75]])
        for e in (np.array([[0, 3]]), np.array([[0, 0.5]])):
            with self.assertRaises(ValueError):
                aggregate_features(X, e)

    def test_oversampling_never_introduces_holdout_ids(self):
        y = np.array([0, 0, 0, 1, 1, 2, 2])
        ids = np.array([0, 1, 2, 3, 5])
        sampled = resample_ids(ids, y, 4, 42)
        self.assertTrue(set(sampled).issubset(set(ids)))
        self.assertTrue(set(ids).issubset(set(sampled)))
        np.testing.assert_array_equal(
            np.unique(y[sampled], return_counts=True)[1], [4, 4, 4]
        )
        np.testing.assert_array_equal(sampled, resample_ids(ids, y, 4, 42))

    def test_scaler_does_not_fit_holdout_features(self):
        X = np.array([[0.0], [2.0], [4.0], [1000.0]], dtype=np.float32)
        y = np.array([0, 0, 1, 1])
        args = SimpleNamespace(ros_floor=3, seed=42)
        a = prepare(X, [], y, np.array([0, 1, 2]), np.array([3]), 0, True, args)
        modified = X.copy()
        modified[3] = 100000
        b = prepare(modified, [], y, np.array([0, 1, 2]), np.array([3]), 0, True, args)
        np.testing.assert_array_equal(a[0], b[0])
        self.assertNotEqual(float(a[2][0, 0]), float(b[2][0, 0]))


if __name__ == "__main__":
    unittest.main()
