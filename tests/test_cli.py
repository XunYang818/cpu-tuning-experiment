"""Command dispatch and compatibility checks, without running model training."""

import subprocess
import sys
import unittest
from pathlib import Path

from cpu_tuning.cli import COMMANDS

ROOT = Path(__file__).resolve().parents[1]


class CommandTests(unittest.TestCase):
    def test_every_command_exposes_its_own_options(self):
        expected = {
            "benchmark": "--data",
            "baseline": "--data",
            "summarize-baseline": "--runs",
            "graph-restarts": "--restarts",
            "mlp": "--cache",
            "summarize-mlp": "--runs",
            "plot-mlp": "--summary",
        }
        for command in COMMANDS:
            with self.subTest(command=command):
                result = subprocess.run(
                    [sys.executable, "-B", str(ROOT / "run.py"), command, "--help"],
                    capture_output=True,
                    text=True,
                    cwd=ROOT,
                    check=True,
                )
                self.assertIn(expected[command], result.stdout)

    def test_legacy_entry_points_still_run(self):
        for filename in (
            "experiment.py",
            "aggregate.py",
            "graph_restarts.py",
            "mlp_experiment.py",
            "summarize_mlp.py",
            "plot_mlp.py",
        ):
            with self.subTest(filename=filename):
                result = subprocess.run(
                    [sys.executable, "-B", str(ROOT / filename), "--help"],
                    capture_output=True,
                    text=True,
                    cwd=ROOT,
                    check=True,
                )
                self.assertIn("--out", result.stdout)

    def test_unknown_command_fails(self):
        result = subprocess.run(
            [sys.executable, "-B", str(ROOT / "run.py"), "missing-command"],
            capture_output=True,
            text=True,
            cwd=ROOT,
        )
        self.assertNotEqual(result.returncode, 0)


if __name__ == "__main__":
    unittest.main()
