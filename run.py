"""Unified entry point: python -B run.py --help."""

import sys

sys.dont_write_bytecode = True

from cpu_tuning.cli import main

if __name__ == "__main__":
    main()
