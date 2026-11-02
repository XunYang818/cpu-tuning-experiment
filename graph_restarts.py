"""Compatibility entry point; new usage: python -B run.py --help."""

import sys

sys.dont_write_bytecode = True

from cpu_tuning.graph_restarts import main

__all__ = ["main"]

if __name__ == "__main__":
    main()
