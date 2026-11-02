"""Alternative entry point: python -B -m cpu_tuning."""

import sys

sys.dont_write_bytecode = True

from .cli import main

if __name__ == "__main__":
    main()
