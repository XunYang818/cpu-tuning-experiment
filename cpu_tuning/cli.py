"""Dispatch commands lazily so plotting is independent of model training."""

import argparse
import importlib
import sys

COMMANDS = {
    "benchmark": ("benchmark", "Run the teacher's CPU timing comparisons"),
    "baseline": ("baseline", "Run classification, regression and Louvain baselines"),
    "summarize-baseline": ("baseline_summary", "Summarize complete baseline runs"),
    "graph-restarts": (
        "graph_restarts",
        "Compare Louvain initializations on a saved graph",
    ),
    "mlp": ("mlp", "Run the five declared MLP and graph-feature ablations"),
    "summarize-mlp": ("mlp_summary", "Audit predictions and summarize MLP runs"),
    "plot-mlp": ("plots", "Render figures from an audited MLP summary"),
}


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        description="CPU tuning experiment: one entry point for all stages.",
        epilog="Use run.py COMMAND --help for command-specific options.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("command", nargs="?", choices=COMMANDS)
    parser.description += "\n\n" + "\n".join(
        f"  {name:22} {description}" for name, (_, description) in COMMANDS.items()
    )
    if not args or args[0] in ("-h", "--help"):
        parser.print_help()
        return
    command = args.pop(0)
    if command not in COMMANDS:
        parser.error(f"unknown command: {command}")
    module = importlib.import_module(f"cpu_tuning.{COMMANDS[command][0]}")
    original_argv = sys.argv
    try:
        sys.argv = [f"run.py {command}", *args]
        module.main()
    finally:
        sys.argv = original_argv
