"""Compatibility entry point; new usage: python -B run.py --help."""

import sys

sys.dont_write_bytecode = True

from cpu_tuning.mlp_summary import write_csv, main

__all__ = ["write_csv", "main"]

if __name__ == "__main__":
    main()
