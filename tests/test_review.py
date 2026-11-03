"""Regression checks for evaluation, selection and node-ID validation."""

import unittest
import json
import sys
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import numpy as np
import networkx as nx

from cpu_tuning.baseline import measures, supervised
from cpu_tuning.io import read_edges, completed_run, sha256
from cpu_tuning.graph_restarts import main as graph_restarts_main
from cpu_tuning.protocol import ARMS, check_splits
from cpu_tuning.mlp_summary import audit_class_rows
from cpu_tuning.features import graph_cache


class ReviewTests(unittest.TestCase):
    def test_graph_restarts_reject_missing_or_different_runtime_and_source(self):
        versions = {
            "python": sys.version,
            "numpy": np.__version__,
            "networkx": nx.__version__,
            "package_source_sha256": {
                p.name: sha256(p)
                for p in sorted((Path(__file__).resolve().parents[1] / "cpu_tuning").glob("*.py"))
            },
        }
        for key in versions:
            for missing in (False, True):
                meta = deepcopy(versions)
                if missing:
                    meta.pop(key)
                else:
                    meta[key] = {"different.py": "different"} if isinstance(meta[key], dict) else "different"
                with (
                    self.subTest(key=key, missing=missing),
                    patch.object(sys, "argv", [
                        "graph-restarts", "--run", "run", "--data", "data", "--out", "out",
                    ]),
                    patch("cpu_tuning.graph_restarts.completed_run", return_value=(meta, [])),
                    patch("cpu_tuning.graph_restarts.data_files") as data_files,
                    patch.object(Path, "mkdir") as mkdir,
                    self.assertRaisesRegex(ValueError, key + " differs from the main run"),
                ):
                    graph_restarts_main()
                data_files.assert_not_called()
                mkdir.assert_not_called()

    def test_cache_from_another_implementation_cannot_be_reused(self):
        meta = {"signature": {
            "features_sha256": "feature", "edges_sha256": "edge",
            "protocol": "undirected-binary-self-loop-row-mean-v1",
            "source_sha256": "old-source", "shape": [3, 1],
        }}
        with (
            patch.object(Path, "exists", return_value=True),
            patch.object(Path, "read_text", return_value=json.dumps(meta)),
            patch("cpu_tuning.features.sha256", side_effect=["edge", "current-source"]),
            self.assertRaisesRegex(ValueError, "signature mismatch"),
        ):
            graph_cache(Path("cache"), np.ones((3, 1)), Path("edges"), "feature")

    def test_unfinished_baseline_cannot_be_reused(self):
        for meta in ({}, {"status": "running"}):
            with (
                patch.object(Path, "read_text", return_value=json.dumps(meta)),
                self.assertRaisesRegex(ValueError, "unfinished"),
            ):
                completed_run(Path("run"))

    def test_changed_baseline_artifact_cannot_be_reused(self):
        for name in ("split_indices.npz", "class_recall.csv"):
            meta = {"status": "complete", "artifact_sha256": {name: "old"}}
            with (
                self.subTest(artifact=name),
                patch.object(Path, "read_text", return_value=json.dumps(meta)),
                patch.object(Path, "is_file", return_value=True),
                patch("cpu_tuning.io.sha256", return_value="changed"),
                self.assertRaisesRegex(ValueError, "changed baseline artifact"),
            ):
                completed_run(Path("run"), (name,))

    def test_long_tail_table_is_recomputed_from_predictions(self):
        rows = []
        predictions = {"truth": np.array([0, 1])}
        for name, _, ros in ARMS:
            predictions[name] = np.array([0, 0])
            for label, original in ((0, 2), (1, 1)):
                rows.append({
                    "name": name, "label": str(label),
                    "original_train_count": str(original),
                    "fitted_train_count": str(3 if ros else original),
                    "test_count": "1", "rare10": "True",
                    "precision": str(0.5 if label == 0 else 0),
                    "recall": str(1 if label == 0 else 0),
                    "f1": str(2 / 3 if label == 0 else 0),
                })
        args = (predictions, np.array([0, 0, 1]), np.arange(3), np.array([0, 1]), np.array([1, 0]), 3)
        audit_class_rows(rows, *args)
        for field, changed in (("recall", "0.9"), ("fitted_train_count", "99"), ("rare10", "False")):
            corrupted = deepcopy(rows)
            corrupted[0][field] = changed
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "differs from predictions"):
                audit_class_rows(corrupted, *args)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            audit_class_rows(rows + rows[:1], *args)
        with self.assertRaisesRegex(ValueError, "missing"):
            audit_class_rows(rows[:-1], *args)

    def test_macro_f1_keeps_classes_absent_from_evaluation(self):
        result = measures("classification", [0, 0], [0, 0], labels=[0, 1])
        self.assertEqual(result["accuracy"], 1.0)
        self.assertEqual(result["macro_f1"], 0.5)
        self.assertEqual(result["weighted_f1"], 1.0)

    def test_classification_selects_macro_f1_before_accuracy(self):
        def evaluate(out, task, name, stage, params, *args):
            return {
                "status": "ok",
                "name": name,
                "params": params,
                "metrics": {
                    "accuracy": 0.9 if params["min_samples_leaf"] == 5 else 0.8,
                    "macro_f1": 0.7 if params["min_samples_leaf"] == 2 else 0.2,
                },
            }

        with (
            patch("cpu_tuning.baseline.fit_eval", side_effect=evaluate),
            patch("cpu_tuning.baseline.record") as record,
        ):
            supervised(
                Path("."),
                "classification",
                None,
                None,
                np.array([0]),
                np.array([1]),
                np.array([2]),
                None,
                [],
            )
        selection = record.call_args.args[1]
        self.assertEqual(selection["params"], {"min_samples_leaf": 2})
        self.assertEqual(selection["criterion"], "macro_f1, then accuracy")

    def test_fractional_and_nonfinite_edges_are_rejected_before_cast(self):
        for edges in (
            [[0, 0.5]],
            [[0, np.nan]],
            [[0, np.inf]],
            [[0, -1]],
            [[0, 3]],
        ):
            with (
                self.subTest(edges=edges),
                patch("cpu_tuning.io.read_array", return_value=np.array(edges)),
                self.assertRaises(ValueError),
            ):
                read_edges(Path("edge.csv.gz"), 3)
        with patch("cpu_tuning.io.read_array", return_value=np.array([[0.0, 2.0]])):
            np.testing.assert_array_equal(read_edges(Path("edge.csv.gz"), 3), [[0, 2]])

    def test_sampled_ids_must_be_a_unique_integer_vector(self):
        parts = [np.array([0]), np.array([1]), np.array([2])]
        for sampled in (np.array([0.0, 1.0, 2.0]), np.array([[0, 1, 2]])):
            with self.subTest(sampled=sampled), self.assertRaises(ValueError):
                check_splits(parts, sampled, 3)


if __name__ == "__main__":
    unittest.main()
