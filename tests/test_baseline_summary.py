"""Task-aware aggregation and validation of long-tail reporting inputs."""

import csv
import io
import unittest
from copy import deepcopy
from itertools import combinations
from pathlib import Path
from unittest.mock import patch

from cpu_tuning.baseline_summary import EXPECTED, load_runs, load_class_recalls, make_plot


class BaselineSummaryTests(unittest.TestCase):
    def test_every_task_subset_requires_only_its_own_artifacts_and_results(self):
        tasks = ("classification", "regression", "graph")
        for count in range(1, 4):
            for selected in combinations(tasks, count):
                meta = {"arguments": {"tasks": selected, "seed": 42}}
                rows = []
                for task, name in sorted(EXPECTED):
                    if task not in selected:
                        continue
                    row = {"task": task, "status": "ok"}
                    if task == "graph":
                        row.update(stage="comparison", resolution=float(name))
                    else:
                        row.update(stage="test", name=name)
                    rows.append(row)
                with (
                    self.subTest(tasks=selected),
                    patch("cpu_tuning.baseline_summary.completed_run", return_value=(meta, rows)),
                    patch("cpu_tuning.baseline_summary.verify_artifacts") as verify,
                    patch("cpu_tuning.baseline_summary.config_signature", return_value={}),
                ):
                    loaded, _ = load_runs([Path("run")])
                    self.assertEqual(set(loaded[0][4]), {k for k in EXPECTED if k[0] in selected})
                    expected = set()
                    if set(selected) & {"classification", "regression"}:
                        expected.add("split_indices.npz")
                    if "classification" in selected:
                        expected.add("class_recall.csv")
                    if "graph" in selected:
                        expected.add("graph_nodes.npy")
                    self.assertEqual(set(verify.call_args.args[2]), expected)

    def test_subset_still_rejects_missing_results(self):
        meta = {"arguments": {"tasks": ["graph"], "seed": 42}}
        rows = [{"task": "graph", "stage": "comparison", "resolution": 1.0}]
        with (
            patch("cpu_tuning.baseline_summary.completed_run", return_value=(meta, rows)),
            patch("cpu_tuning.baseline_summary.verify_artifacts"),
            patch("cpu_tuning.baseline_summary.config_signature", return_value={}),
            self.assertRaisesRegex(ValueError, "missing or unexpected result"),
        ):
            load_runs([Path("run")])

    def test_plot_includes_only_available_tasks_and_handles_zero_mae(self):
        rows = [dict(task="regression", metric="mae", model="rf_baseline", mean=0, sd=0)]
        with patch.object(Path, "write_text") as write:
            make_plot(Path("comparison.svg"), rows)
        svg = write.call_args.args[0]
        self.assertIn("MAE (years; lower is better)", svg)
        self.assertNotIn("Accuracy", svg)
        self.assertNotIn("Graph modularity", svg)

    def recall_fixture(self):
        meta = {"data": {"label_counts": {"0": 6, "1": 4}}}
        models = sorted(name for task, name in EXPECTED if task == "classification")
        results = {("classification", name): {"n_eval": 3} for name in models}
        rows = [
            dict(model=name, label=label, test_count=count, recall=0.5)
            for name in models for label, count in (("0", 2), ("1", 1))
        ]
        return meta, results, rows

    def read_recalls(self, rows, meta, results):
        stream = io.StringIO()
        writer = csv.DictWriter(stream, fieldnames=["model", "label", "test_count", "recall"])
        writer.writeheader()
        writer.writerows(rows)
        stream.seek(0)
        with patch.object(Path, "open", return_value=stream):
            return load_class_recalls(Path("run"), meta, results)

    def test_valid_recalls_are_parsed(self):
        meta, results, rows = self.recall_fixture()
        self.assertEqual(self.read_recalls(rows, meta, results), rows)

    def test_invalid_counts_and_recalls_are_rejected(self):
        meta, results, rows = self.recall_fixture()
        for field, value in (
            ("recall", "999"), ("recall", "nan"), ("recall", "inf"),
            ("recall", "-0.1"), ("test_count", "-1"),
            ("test_count", "1.5"), ("test_count", "4"),
            ("test_count", "0"), ("test_count", "1"),
        ):
            altered = deepcopy(rows)
            altered[0][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                self.read_recalls(altered, meta, results)
        for altered in (rows[:-1], rows + rows[:1], [dict(rows[0], model="unknown"), *rows[1:]]):
            with self.subTest(rows=altered), self.assertRaises(ValueError):
                self.read_recalls(altered, meta, results)

    def test_count_totals_must_match_evaluation_size(self):
        meta, results, rows = self.recall_fixture()
        for row in rows:
            row["test_count"] = 1
        with self.assertRaisesRegex(ValueError, "evaluation size"):
            self.read_recalls(rows, meta, results)


if __name__ == "__main__":
    unittest.main()
