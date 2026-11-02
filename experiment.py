"""Compatibility entry point; new usage: python -B run.py --help."""

import sys

sys.dont_write_bytecode = True

from cpu_tuning.io import dump, sha256, read_array, record
from cpu_tuning.baseline import (
    measures,
    make_model,
    fit_eval,
    supervised,
    graph_experiment,
    summarize,
    main,
)

__all__ = [
    "dump",
    "sha256",
    "read_array",
    "record",
    "measures",
    "make_model",
    "fit_eval",
    "supervised",
    "graph_experiment",
    "summarize",
    "main",
]

if __name__ == "__main__":
    main()
