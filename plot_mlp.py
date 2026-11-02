"""Compatibility entry point; new usage: python -B run.py --help."""

import sys

sys.dont_write_bytecode = True

from cpu_tuning.plots import rows, save, main

__all__ = ["rows", "save", "main"]

if __name__ == "__main__":
    main()
