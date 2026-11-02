"""Compatibility entry point; new usage: python -B run.py --help."""

import sys

sys.dont_write_bytecode = True

from cpu_tuning.features import aggregate_features, graph_cache, make_input
from cpu_tuning.protocol import ARMS, MLP_PARAMS, resample_ids, check_splits, scores
from cpu_tuning.mlp import prepare, validate_arm, final_arm, main

__all__ = [
    "aggregate_features",
    "graph_cache",
    "make_input",
    "ARMS",
    "MLP_PARAMS",
    "resample_ids",
    "check_splits",
    "scores",
    "prepare",
    "validate_arm",
    "final_arm",
    "main",
]

if __name__ == "__main__":
    main()
