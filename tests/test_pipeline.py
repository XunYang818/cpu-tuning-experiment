"""Small CLI pipeline; generated artifacts stay in the configured process directory."""

import csv
import gzip
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from itertools import combinations
from pathlib import Path

import numpy as np
from cpu_tuning.io import completed_run, sha256

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(
    os.environ.get("CPU_TUNING_TEST_ROOT"),
    "set CPU_TUNING_TEST_ROOT for pipeline artifacts",
)
class PipelineTests(unittest.TestCase):
    def test_all_task_subsets_can_be_run_and_summarized(self):
        artifact_root = Path(os.environ["CPU_TUNING_TEST_ROOT"])
        artifact_root.mkdir(parents=True, exist_ok=True)
        work = Path(tempfile.mkdtemp(prefix="subsets-", dir=artifact_root))
        raw = work / "raw"
        raw.mkdir()
        rng = np.random.RandomState(7)
        size = 60
        features = rng.normal(size=(size, 4))
        for name, data in (
            ("node-feat.csv.gz", features),
            ("node-label.csv.gz", np.arange(size) % 3),
            ("node-year.csv.gz", 2010 + features[:, 0]),
            ("edge.csv.gz", np.column_stack((np.arange(size), (np.arange(size) + 1) % size))),
        ):
            with gzip.open(raw / name, "wt", encoding="utf-8") as f:
                np.savetxt(f, data, delimiter=",")
        env = dict(
            os.environ, PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1",
            MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
            PYTHONIOENCODING="utf-8",
        )

        def run(command, *args, success=True):
            result = subprocess.run(
                [sys.executable, "-B", str(ROOT / "run.py"), command, *map(str, args)],
                cwd=ROOT, env=env, capture_output=True, encoding="utf-8",
            )
            if success:
                self.assertEqual(result.returncode, 0, msg=result.stderr)
            else:
                self.assertNotEqual(result.returncode, 0)
            return result

        tasks = ("classification", "regression", "graph")
        for count in range(1, 4):
            for selected in combinations(tasks, count):
                with self.subTest(tasks=selected):
                    tag = "-".join(selected)
                    paths = [work / f"{tag}-{seed}" for seed in (42, 43)]
                    for seed, path in zip((42, 43), paths):
                        run(
                            "tune", "--data", raw, "--out", path,
                            "--tasks", *selected, "--seed", seed, "--samples", size,
                            "--graph-nodes", size, "--trees", 1, "--jobs", 1,
                        )
                        meta = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
                        self.assertEqual("class_recall.csv" in meta["artifact_sha256"], "classification" in selected)
                    out = work / f"{tag}-summary"
                    run("summarize-baseline", "--runs", *paths, "--out", out)
                    self.assertTrue((out / "mean_sd.csv").is_file())
                    self.assertTrue((out / "comparison.svg").is_file())
                    self.assertEqual((out / "recall_by_class.csv").exists(), "classification" in selected)
                    with (out / "per_seed.csv").open(encoding="utf-8-sig", newline="") as f:
                        self.assertEqual({row["task"] for row in csv.DictReader(f)}, set(selected))

        damaged = work / "damaged-recall"
        shutil.copytree(work / "classification-42", damaged)
        with (damaged / "class_recall.csv").open("a", encoding="utf-8") as f:
            f.write("\n")
        result = run(
            "summarize-baseline", "--runs", damaged, work / "classification-43",
            "--out", work / "damaged-summary", success=False,
        )
        self.assertIn("changed baseline artifact: class_recall.csv", result.stderr)
        self.assertFalse((work / "damaged-summary").exists())

    def test_baseline_to_mlp_with_all_five_arms(self):
        artifact_root = Path(os.environ["CPU_TUNING_TEST_ROOT"])
        artifact_root.mkdir(parents=True, exist_ok=True)
        # mkdtemp creates a unique directory; no automatic deletion is performed.
        work = Path(tempfile.mkdtemp(prefix="pipeline-中文-", dir=artifact_root))
        raw = work / "data" / "raw"
        raw.mkdir(parents=True)
        rng = np.random.RandomState(19)
        size = 180
        labels = np.arange(size) % 3
        features = rng.normal(size=(size, 6)).astype(np.float32)
        features[:, 0] += labels
        edges = np.vstack(
            [
                np.column_stack((np.arange(size), (np.arange(size) + 1) % size)),
                rng.randint(0, size, (360, 2)),
            ]
        )
        for filename, data, fmt in (
            ("node-feat.csv.gz", features, "%.6f"),
            ("node-label.csv.gz", labels, "%d"),
            ("node-year.csv.gz", 2010 + features[:, 1], "%.6f"),
            ("edge.csv.gz", edges, "%d"),
        ):
            with gzip.open(raw / filename, "wt", encoding="utf-8") as f:
                np.savetxt(f, data, fmt=fmt, delimiter=",")

        env = dict(
            os.environ,
            PYTHONDONTWRITEBYTECODE="1",
            PYTHONIOENCODING="utf-8",
            OMP_NUM_THREADS="1",
            MKL_NUM_THREADS="1",
            OPENBLAS_NUM_THREADS="1",
            MPLCONFIGDIR=str(work / "matplotlib-cache"),
        )

        def run(command, *args, success=True):
            result = subprocess.run(
                [sys.executable, "-B", str(ROOT / "run.py"), command, *map(str, args)],
                cwd=ROOT,
                env=env,
                capture_output=True,
                text=True,
                encoding="utf-8",
                check=False,
            )
            (work / (command + ".log")).write_text(
                result.stdout + result.stderr, encoding="utf-8"
            )
            if success:
                self.assertEqual(result.returncode, 0, msg=result.stderr)
            else:
                self.assertNotEqual(result.returncode, 0)
            return result

        # A graph-only run must not attempt classification stratification.
        run(
            "baseline",
            "--data",
            raw,
            "--out",
            work / "graph-only",
            "--tasks",
            "graph",
            "--samples",
            2,
            "--graph-nodes",
            90,
        )
        self.assertFalse((work / "graph-only" / "split_indices.npz").exists())

        run(
            "benchmark",
            "--data",
            raw,
            "--out",
            work / "benchmark",
            "--samples",
            size,
            "--trees",
            2,
            "--jobs",
            1,
            2,
            "--graph-nodes",
            90,
        )
        bench = [
            json.loads(line)
            for line in (work / "benchmark" / "results.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        self.assertEqual(
            {r["name"] for r in bench},
            {
                "rf_classification",
                "rf_regression",
                "mlp",
                "logistic_saga",
                "hist_gradient_boosting",
                "rf_full_classification",
                "louvain_teacher",
            },
        )
        self.assertEqual(
            [r["n_jobs"] for r in bench if r["name"] == "rf_classification"], [1, 2]
        )
        saga = next(r for r in bench if r["name"] == "logistic_saga")
        self.assertIsNone(saga["n_jobs"])
        self.assertIsNone(saga["model_params"].get("n_jobs"))
        self.assertEqual(saga["teacher_n_jobs"], 4)
        self.assertTrue((work / "benchmark" / "full_split_indices.npz").exists())
        expected_rng = np.random.RandomState(42)
        expected_rng.choice(size, size, replace=False)
        expected_nodes = expected_rng.choice(size, 90, replace=False)
        np.testing.assert_array_equal(
            np.load(work / "benchmark" / "graph_nodes.npy"), expected_nodes
        )
        import networkx as nx

        mask = np.isin(edges[:, 0], expected_nodes) & np.isin(
            edges[:, 1], expected_nodes
        )
        graph = nx.Graph()
        graph.add_edges_from(edges[mask].tolist())
        graph_row = next(r for r in bench if r["name"] == "louvain_teacher")
        self.assertEqual(
            (graph_row["nodes"], graph_row["edges"], graph_row["self_loops"]),
            (
                graph.number_of_nodes(),
                graph.number_of_edges(),
                nx.number_of_selfloops(graph),
            ),
        )

        run(
            "baseline",
            "--data",
            raw,
            "--out",
            work / "baseline",
            "--samples",
            size,
            "--graph-nodes",
            90,
            "--trees",
            2,
            "--jobs",
            1,
        )
        run(
            "mlp",
            "--data",
            raw,
            "--run",
            work / "baseline",
            "--cache",
            work / "cache",
            "--out",
            work / "mlp",
            "--max-epochs",
            2,
            "--patience",
            1,
            "--ros-floor",
            50,
        )
        result = json.loads(
            (work / "mlp" / "final_summary.json").read_text(encoding="utf-8")
        )
        self.assertEqual(
            [r["name"] for r in result["test"]],
            ["raw", "raw_ros", "hop1", "hop2", "hop2_ros"],
        )
        for row in result["test"]:
            self.assertEqual(row["status"], "ok")
            self.assertLessEqual(row["epochs"], 2)
            self.assertGreaterEqual(row["n_train_fit"], row["n_train_original"])
        with np.load(work / "mlp" / "test_predictions.npz") as predictions:
            np.testing.assert_array_equal(
                predictions["truth"], labels[predictions["node_ids"]]
            )
            self.assertEqual(len(predictions.files), 7)

        # A valid saved split may lack one class in its fitting partition.
        run(
            "baseline",
            "--data",
            raw,
            "--out",
            work / "baseline-2",
            "--samples",
            size,
            "--graph-nodes",
            90,
            "--trees",
            2,
            "--jobs",
            1,
            "--seed",
            43,
        )
        relocated_data = work / "relocated-data"
        shutil.copytree(raw, relocated_data)
        relocated_run = work / "relocated-baseline"
        shutil.copytree(work / "baseline", relocated_run)
        run(
            "graph-restarts",
            "--data",
            relocated_data,
            "--run",
            relocated_run,
            "--out",
            work / "restarts",
            "--restarts",
            2,
        )
        run(
            "graph-restarts",
            "--data",
            relocated_data,
            "--run",
            work / "baseline-2",
            "--out",
            work / "restarts-2",
            "--restarts",
            2,
        )
        run(
            "summarize-baseline",
            "--runs",
            relocated_run,
            work / "baseline-2",
            "--graph-restarts",
            work / "restarts",
            work / "restarts-2",
            "--out",
            work / "baseline-summary",
        )
        self.assertTrue((work / "baseline-summary" / "mean_sd.csv").exists())
        for filename in ("per_seed.csv", "graph_restarts.csv", "provenance.json"):
            text = (work / "baseline-summary" / filename).read_text(
                encoding="utf-8-sig"
            )
            self.assertNotIn(str(work), text)
        altered_run = work / "altered-baseline"
        shutil.copytree(work / "baseline-2", altered_run)
        altered_meta = altered_run / "metadata.json"
        metadata = json.loads(altered_meta.read_text(encoding="utf-8"))
        metadata["timing_environment"]["logical_cpus"] += 1
        altered_meta.write_text(json.dumps(metadata), encoding="utf-8")
        mismatch = subprocess.run(
            [
                sys.executable,
                "-B",
                str(ROOT / "run.py"),
                "summarize-baseline",
                "--runs",
                str(relocated_run),
                str(altered_run),
                "--out",
                str(work / "bad-summary"),
            ],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=False,
        )
        self.assertNotEqual(mismatch.returncode, 0)
        self.assertIn("incompatible", mismatch.stderr)
        # Match by seed, not argument order, and compare all nine classifiers.
        run(
            "mlp", "--data", relocated_data, "--run", work / "baseline-2",
            "--cache", work / "cache", "--out", work / "mlp-comparison-2",
            "--max-epochs", 2, "--patience", 1, "--ros-floor", 50,
        )
        comparison_args = (
            "--data", relocated_data, "--runs", work / "mlp", work / "mlp-comparison-2",
            "--tuning-runs", work / "baseline-2", relocated_run,
        )
        run("summarize-mlp", *comparison_args, "--out", work / "model-summary")
        with (work / "model-summary" / "model_comparison.csv").open(
            encoding="utf-8-sig", newline=""
        ) as f:
            comparison = list(csv.DictReader(f))
        self.assertEqual(len(comparison), 18)
        self.assertEqual(sum(row["selected_mlp"] == "True" for row in comparison), 2)
        self.assertEqual({row["seed"] for row in comparison}, {"42", "43"})
        for row in comparison:
            self.assertEqual(int(row["n_test"]), 54)
            self.assertGreaterEqual(float(row["search_seconds"]), 0)
            if row["name"].startswith("mlp_hop"):
                self.assertEqual(row["setting"], "transductive")
        tree_baseline = next(row for row in comparison if row["name"] == "rf_baseline" and row["seed"] == "42")
        tree_result = next(
            json.loads(line) for line in (relocated_run / "results.jsonl").read_text(encoding="utf-8").splitlines()
            if json.loads(line).get("name") == "rf_baseline" and json.loads(line)["task"] == "classification"
        )
        self.assertEqual(float(tree_baseline["macro_f1"]), tree_result["metrics"]["macro_f1"])
        for filename in ("model_comparison_mean_sd.csv", "model_comparison_paired.csv", "pipeline_costs.csv", "audit.json"):
            self.assertTrue((work / "model-summary" / filename).is_file())
            self.assertNotIn(str(work), (work / "model-summary" / filename).read_text(encoding="utf-8-sig"))
        rejected = run(
            "summarize-mlp", "--data", relocated_data,
            "--runs", work / "mlp", work / "mlp-comparison-2",
            "--tuning-runs", relocated_run, "--out", work / "missing-seed-summary", success=False,
        )
        self.assertIn("same seeds", rejected.stderr)
        self.assertFalse((work / "missing-seed-summary").exists())
        split_path = work / "baseline-2" / "split_indices.npz"
        with np.load(split_path) as split:
            sampled = split["sampled"].copy()
            train = split["train"].copy()
            val = split["validation"].copy()
            test = split["test"].copy()
        test = np.concatenate([test, train[labels[train] == 2], val[labels[val] == 2]])
        train = train[labels[train] != 2]
        val = val[labels[val] != 2]
        np.savez_compressed(
            split_path, sampled=sampled, train=train, validation=val, test=test
        )
        with self.assertRaisesRegex(ValueError, "changed baseline artifact"):
            completed_run(work / "baseline-2", ("split_indices.npz",))
        # This intentionally edited split is a synthetic missing-class fixture.
        # Update its manifest explicitly; production consumers never do this.
        fixture_meta_path = work / "baseline-2" / "metadata.json"
        fixture_meta = json.loads(fixture_meta_path.read_text(encoding="utf-8"))
        fixture_meta["artifact_sha256"]["split_indices.npz"] = sha256(split_path)
        fixture_meta_path.write_text(json.dumps(fixture_meta), encoding="utf-8")

        # Simulate interruption after writing only the first cache file.
        interrupted = work / "interrupted-cache"
        interrupted.mkdir()
        (interrupted / "hop1.npy").write_bytes(b"incomplete")
        run(
            "mlp",
            "--data",
            relocated_data,
            "--run",
            work / "baseline-2",
            "--cache",
            interrupted,
            "--out",
            work / "mlp-2",
            "--max-epochs",
            2,
            "--patience",
            1,
            "--ros-floor",
            50,
        )
        self.assertEqual(
            json.loads((work / "mlp-2" / "protocol.json").read_text(encoding="utf-8"))[
                "class_labels"
            ],
            [0, 1, 2],
        )
        relocated_mlp = work / "relocated-mlp"
        shutil.copytree(work / "mlp", relocated_mlp)
        run(
            "summarize-mlp",
            "--data",
            relocated_data,
            "--runs",
            relocated_mlp,
            work / "mlp-2",
            "--out",
            work / "summary",
        )
        self.assertTrue((work / "summary" / "audit.json").exists())
        for filename in ("per_seed.csv", "audit.json"):
            self.assertNotIn(
                str(work), (work / "summary" / filename).read_text(encoding="utf-8-sig")
            )
        run("plot-mlp", "--summary", work / "summary", "--out", work / "figures")
        ablation = (work / "figures" / "ablation.svg").read_text(encoding="utf-8")
        for expected in ("2 seeds", "3 classes", "floor = 50"):
            self.assertIn(expected, ablation)
        self.assertIn(
            "2-seed mean",
            (work / "figures" / "runtime.svg").read_text(encoding="utf-8"),
        )

        # Auditing must reject changed graph data even when labels still match.
        altered_data = work / "altered-data"
        shutil.copytree(raw, altered_data)
        with gzip.open(altered_data / "edge.csv.gz", "wt", encoding="utf-8") as f:
            f.write("0,1\n")
        mismatch = subprocess.run(
            [
                sys.executable,
                "-B",
                str(ROOT / "run.py"),
                "summarize-mlp",
                "--data",
                str(altered_data),
                "--runs",
                str(relocated_mlp),
                str(work / "mlp-2"),
                "--out",
                str(work / "bad-data-summary"),
            ],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=False,
        )
        self.assertNotEqual(mismatch.returncode, 0)
        self.assertIn("missing or changed edges", mismatch.stderr)


if __name__ == "__main__":
    unittest.main()
