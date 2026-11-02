"""Small CLI pipeline; generated artifacts stay in the configured process directory."""

import gzip
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(
    os.environ.get("CPU_TUNING_TEST_ROOT"),
    "set CPU_TUNING_TEST_ROOT for pipeline artifacts",
)
class PipelineTests(unittest.TestCase):
    def test_baseline_to_mlp_with_all_five_arms(self):
        artifact_root = Path(os.environ["CPU_TUNING_TEST_ROOT"])
        artifact_root.mkdir(parents=True, exist_ok=True)
        # mkdtemp creates a unique directory; no automatic deletion is performed.
        work = Path(tempfile.mkdtemp(prefix="pipeline-", dir=artifact_root))
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
            OMP_NUM_THREADS="1",
            MKL_NUM_THREADS="1",
            OPENBLAS_NUM_THREADS="1",
        )

        def run(command, *args):
            result = subprocess.run(
                [sys.executable, "-B", str(ROOT / "run.py"), command, *map(str, args)],
                cwd=ROOT,
                env=env,
                capture_output=True,
                text=True,
                encoding="utf-8",
            )
            (work / (command + ".log")).write_text(
                result.stdout + result.stderr, encoding="utf-8"
            )
            self.assertEqual(result.returncode, 0, msg=result.stderr)

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


if __name__ == "__main__":
    unittest.main()
