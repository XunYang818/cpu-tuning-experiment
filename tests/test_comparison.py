"""Cross-model comparisons must preserve pairing and account for search costs."""

import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from cpu_tuning.comparison import build_comparison, match_run
from cpu_tuning.protocol import ARMS


class ComparisonTests(unittest.TestCase):
    def metadata(self):
        common = dict(
            python="python", numpy="numpy", sklearn="sklearn",
            package_source_sha256={"module.py": "source"},
            timing_environment={"host": "host", "threadpools": []},
            files={key: {"sha256": key} for key in ("features", "labels", "edges")},
        )
        tree = dict(common, thread_environment={"OMP_NUM_THREADS": "1"},
                    artifact_sha256={"split_indices.npz": "split"})
        mlp = deepcopy(dict(common, threads={"OMP_NUM_THREADS": "1"}, source_split_sha256="split"))
        return tree, mlp

    def test_pair_rejects_different_environment_data_source_or_split(self):
        tree, mlp = self.metadata()
        for key, value in (
            ("timing_environment", {"host": "other"}),
            ("package_source_sha256", {"module.py": "other"}),
            ("threads", {}), ("source_split_sha256", "other"),
            ("files", {key: {"sha256": "other"} for key in tree["files"]}),
        ):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "comparison differs"):
                match_run(Path("tree"), tree, Path("mlp"), dict(mlp, **{key: value}))

    def test_pair_checks_actual_split_membership(self):
        tree, mlp = self.metadata()
        split = dict(sampled=np.arange(4), train=np.array([0, 1]),
                     validation=np.array([2]), test=np.array([3]))
        for other in (split, dict(split, test=np.array([2]))):
            left, right = MagicMock(), MagicMock()
            left.__enter__.return_value = split
            right.__enter__.return_value = other
            with patch("cpu_tuning.comparison.np.load", side_effect=[left, right]):
                if other is split:
                    self.assertEqual(match_run(Path("tree"), tree, Path("mlp"), mlp), (3, 1))
                else:
                    with self.assertRaisesRegex(ValueError, "test IDs"):
                        match_run(Path("tree"), tree, Path("mlp"), mlp)

    def fixtures(self):
        trees, mlps = [], []
        for seed in (42, 43):
            tree, mlp = self.metadata()
            tree["arguments"] = {"tasks": ["classification"]}
            metrics = dict(accuracy=0.5, macro_f1=0.4, weighted_f1=0.45)
            results = {("classification", name): dict(
                n_train=3, n_eval=1, metrics=metrics, fit_seconds=2.0, predict_seconds=0.1,
            ) for name in ("rf_baseline", "rf_tuned", "rf_balanced", "decision_tree")}
            records = [dict(task="classification", stage="selection", search_seconds=8.0)]
            trees.append((seed, Path(f"tree-{seed}"), tree, records, results))
            mlp.update(seed=seed, graph_cache={"compute_seconds": 3.0, "save_seconds": 0.5},
                       cache_access_seconds=0.2)
            mlps.append(dict(
                path=Path(f"mlp-{seed}"), meta=mlp,
                validation=[dict(name=name, search_seconds=4.0) for name, _, _ in ARMS],
                test=[dict(name=name, order=order, n_train_original=3, n_test=1,
                           metrics=dict(metrics, macro_f1=0.6), fit_seconds=1.0, predict_seconds=0.2)
                      for name, order, _ in ARMS],
                selection=dict(selected="raw", search_seconds=22.0),
            ))
        return trees, mlps

    def test_shared_baseline_pairing_and_costs(self):
        trees, mlps = self.fixtures()
        with (
            patch("cpu_tuning.comparison.load_runs", return_value=(trees[::-1], {})),
            patch("cpu_tuning.comparison.match_run", return_value=(3, 1)),
        ):
            rows, means, paired, costs, provenance = build_comparison([], mlps)
        self.assertEqual(len(rows), 18)
        self.assertEqual(sum(row["selected_mlp"] for row in rows), 2)
        for row in rows:
            expected = 4.0 if row["name"].startswith("mlp_") else 8.0 if row["name"] == "rf_tuned" else 0.0
            self.assertEqual(row["search_seconds"], expected)
        delta = next(row for row in paired if row["name"] == "mlp_raw" and row["metric"] == "macro_f1")
        self.assertAlmostEqual(delta["mean_delta"], 0.2)
        self.assertEqual(delta["sd_delta"], 0)
        self.assertTrue(all(row["n"] == 2 for row in means))
        self.assertEqual(costs[0]["mlp_all_arms_search_seconds"], 22.0)
        self.assertTrue(all(row["timing_environment_matched"] for row in provenance))

    def test_incomplete_pairing_or_missing_search_time_is_rejected(self):
        trees, mlps = self.fixtures()
        with patch("cpu_tuning.comparison.load_runs", return_value=(trees[:1], {})):
            with self.assertRaisesRegex(ValueError, "same seeds"):
                build_comparison([], mlps)
        mlps[0]["validation"][0].pop("search_seconds")
        with (
            patch("cpu_tuning.comparison.load_runs", return_value=(trees, {})),
            patch("cpu_tuning.comparison.match_run", return_value=(3, 1)),
            self.assertRaisesRegex(ValueError, "search time"),
        ):
            build_comparison([], mlps)

    def test_invalid_durations_are_rejected(self):
        for value in (-1, float("nan"), float("inf")):
            trees, mlps = self.fixtures()
            mlps[0]["test"][0]["fit_seconds"] = value
            with (
                self.subTest(value=value),
                patch("cpu_tuning.comparison.load_runs", return_value=(trees, {})),
                patch("cpu_tuning.comparison.match_run", return_value=(3, 1)),
                self.assertRaisesRegex(ValueError, "invalid comparison value"),
            ):
                build_comparison([], mlps)


if __name__ == "__main__":
    unittest.main()
