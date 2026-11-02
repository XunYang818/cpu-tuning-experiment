"""Compatibility entry point; new usage: python -B run.py --help."""

import sys

sys.dont_write_bytecode = True

from cpu_tuning.baseline_summary import (
    write_csv,
    config_signature,
    load_runs,
    make_plot,
    main,
)

__all__ = ["write_csv", "config_signature", "load_runs", "make_plot", "main"]

if __name__ == "__main__":
    main()
