"""Aggregate complete, comparable experiment runs without accessing the test data."""

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from xml.sax.saxutils import escape

EXPECTED = {
    ("classification", "rf_baseline"),
    ("classification", "rf_tuned"),
    ("classification", "rf_balanced"),
    ("classification", "decision_tree"),
    ("regression", "rf_baseline"),
    ("regression", "rf_tuned"),
    ("regression", "hist_gradient_boosting"),
    ("regression", "extra_trees"),
    ("graph", "1.0"),
    ("graph", "0.5"),
    ("graph", "1.5"),
}


def write_csv(path, fields, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def config_signature(meta):
    args = meta["arguments"]
    return {
        "data_sha256": {k: v["sha256"] for k, v in meta["files"].items()},
        "samples": args["samples"],
        "graph_nodes": args["graph_nodes"],
        "trees": args["trees"],
        "jobs": args["jobs"],
        "tasks": sorted(args["tasks"]),
        "python": meta["python"],
        "numpy": meta["numpy"],
        "sklearn": meta["sklearn"],
        "networkx": meta["networkx"],
        "thread_environment": meta["thread_environment"],
    }


def load_runs(paths):
    loaded = []
    signature = None
    method_signature = None
    seeds = set()
    for path in paths:
        meta = json.loads((path / "metadata.json").read_text(encoding="utf-8-sig"))
        sig = config_signature(meta)
        if signature is None:
            signature = sig
        elif sig != signature:
            raise ValueError(
                f"incompatible dataset, configuration, versions or thread settings: {path}"
            )
        seed = meta["arguments"]["seed"]
        if seed in seeds:
            raise ValueError(f"duplicate seed: {seed}")
        seeds.add(seed)
        records = [
            json.loads(line)
            for line in (path / "results.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        if any(r.get("status") != "ok" for r in records):
            raise ValueError(f"failed run: {path}")
        methods = {}
        for r in records:
            if r.get("stage") in ("validation", "test") and r.get("name") != "rf_tuned":
                key = (r["task"], r["stage"], r["name"])
                if key in methods:
                    raise ValueError(f"duplicate model record {key}: {path}")
                methods[key] = {
                    k: v
                    for k, v in r.get("model_params", {}).items()
                    if k != "random_state"
                }
        if method_signature is None:
            method_signature = methods
        elif methods != method_signature:
            raise ValueError(f"model parameters differ between runs: {path}")
        result = {}
        for r in records:
            if r["stage"] == "test":
                key = (r["task"], r["name"])
            elif r["stage"] == "comparison":
                key = (r["task"], str(r["resolution"]))
            else:
                continue
            if key in result:
                raise ValueError(f"duplicate result {key}: {path}")
            result[key] = r
        if set(result) != EXPECTED:
            raise ValueError(
                f"missing or unexpected result in {path}: {set(result) ^ EXPECTED}"
            )
        loaded.append((seed, path, meta, records, result))
    return sorted(loaded), signature


def make_plot(path, grouped):
    panels = [
        ("classification", "accuracy", "Accuracy", 1.0),
        ("classification", "macro_f1", "Macro F1", 1.0),
        ("regression", "mae", "MAE (years; lower is better)", None),
        ("graph", "modularity_at_1", "Graph modularity at resolution 1", 1.0),
    ]
    height = 80 + sum(
        55
        + 28 * len([x for x in grouped if x["task"] == task and x["metric"] == metric])
        for task, metric, _, _ in panels
    )
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="25" y="30" font-family="Arial" font-size="19">Multi-seed mean and sample standard deviation</text>',
        '<text x="25" y="52" font-family="Arial" font-size="12">Bars show means; whiskers show mean ± 1 SD; scales differ between panels.</text>',
    ]
    y = 75
    for task, metric, title, bound in panels:
        rows = [r for r in grouped if r["task"] == task and r["metric"] == metric]
        scale = bound or max(r["mean"] + r["sd"] for r in rows) * 1.1
        svg.append(
            f'<text x="25" y="{y}" font-family="Arial" font-weight="bold" font-size="15">{escape(title)}</text>'
        )
        y += 18
        for r in rows:
            mean, sd = r["mean"], r["sd"]
            x = 280
            factor = 480 / scale
            left, right = max(0, mean - sd) * factor, min(scale, mean + sd) * factor
            svg.append(
                f'<text x="25" y="{y + 13}" font-family="Arial" font-size="12">{escape(r["model"])}</text>'
            )
            svg.append(
                f'<rect x="{x}" y="{y + 2}" width="{max(0, mean) * factor:.2f}" height="14" fill="#4779ad"/>'
            )
            svg.append(
                f'<line x1="{x + left:.2f}" x2="{x + right:.2f}" y1="{y + 9}" y2="{y + 9}" stroke="#111" stroke-width="2"/>'
            )
            svg.append(
                f'<text x="780" y="{y + 13}" font-family="Arial" font-size="12">{mean:.4f} ± {sd:.4f}</text>'
            )
            y += 28
        y += 37
    svg.append("</svg>")
    path.write_text("\n".join(svg), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", type=Path, required=True)
    parser.add_argument("--graph-restarts", nargs="*", type=Path, default=[])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if len(args.runs) < 2:
        parser.error("at least two complete runs required")
    runs, signature = load_runs(args.runs)
    detailed, samples = [], defaultdict(list)
    for seed, path, meta, records, results in runs:
        for (task, model), r in results.items():
            metrics = dict(r.get("metrics", {}))
            if task == "graph":
                metrics.update(
                    {
                        k: r[k]
                        for k in (
                            "communities",
                            "modularity_at_1",
                            "modularity_at_resolution",
                        )
                    }
                )
                model = "resolution_" + model
            metrics.update(
                {k: r[k] for k in ("fit_seconds", "predict_seconds") if k in r}
            )
            for metric, value in metrics.items():
                if not isinstance(value, (int, float)) or not math.isfinite(value):
                    raise ValueError(f"invalid metric: {path}")
                detailed.append(
                    {
                        "seed": seed,
                        "task": task,
                        "model": model,
                        "metric": metric,
                        "value": value,
                        "source": str(path / "results.jsonl"),
                    }
                )
                samples[(task, model, metric)].append(value)
        for r in records:
            if r.get("stage") == "selection":
                detailed.append(
                    {
                        "seed": seed,
                        "task": r["task"],
                        "model": "parameter_search",
                        "metric": "search_seconds",
                        "value": r["search_seconds"],
                        "source": str(path / "results.jsonl"),
                    }
                )
                samples[(r["task"], "parameter_search", "search_seconds")].append(
                    r["search_seconds"]
                )
    grouped = [
        {
            "task": task,
            "model": model,
            "metric": metric,
            "n": len(values),
            "mean": statistics.mean(values),
            "sd": statistics.stdev(values),
            "minimum": min(values),
            "maximum": max(values),
        }
        for (task, model, metric), values in sorted(samples.items())
    ]
    lookup = {
        (r["seed"], r["task"], r["model"], r["metric"]): r["value"] for r in detailed
    }
    paired = []
    for task, model, metric in [
        ("classification", "rf_tuned", "accuracy"),
        ("classification", "rf_tuned", "macro_f1"),
        ("classification", "rf_balanced", "accuracy"),
        ("classification", "rf_balanced", "macro_f1"),
        ("regression", "rf_tuned", "mae"),
        ("regression", "extra_trees", "mae"),
        ("regression", "hist_gradient_boosting", "mae"),
    ]:
        differences = [
            lookup[(seed, task, model, metric)]
            - lookup[(seed, task, "rf_baseline", metric)]
            for seed, *_ in runs
        ]
        paired.append(
            {
                "task": task,
                "model": model,
                "metric": metric,
                "comparison": "model minus rf_baseline",
                "n": len(differences),
                "mean_delta": statistics.mean(differences),
                "sd_delta": statistics.stdev(differences),
                "per_seed_delta": json.dumps(
                    dict(zip([seed for seed, *_ in runs], differences))
                ),
            }
        )
    # Per-class recall: keep the original full-dataset class size and exclude
    # seeds where a class has no test examples instead of treating it as zero.
    labels = runs[0][2]["data"]["label_counts"]
    class_values = defaultdict(list)
    for seed, path, meta, records, result in runs:
        if meta["data"]["label_counts"] != labels:
            raise ValueError("class counts differ between runs")
        with (path / "class_recall.csv").open(encoding="utf-8-sig", newline="") as f:
            recall_rows = list(csv.DictReader(f))
        seen = set()
        for r in recall_rows:
            key = (r["model"], r["label"])
            if key in seen:
                raise ValueError(f"duplicate per-class recall: {path} {key}")
            seen.add(key)
            if (
                r["model"] in ("rf_baseline", "rf_balanced")
                and int(r["test_count"]) > 0
            ):
                class_values[key].append(
                    (seed, int(r["test_count"]), float(r["recall"]))
                )
        expected = {
            (model, label)
            for model in ("rf_baseline", "rf_balanced")
            for label in labels
        }
        if not expected.issubset(seen):
            raise ValueError(f"missing per-class recall: {path}")
    per_class = []
    for label, count in sorted(labels.items(), key=lambda kv: (kv[1], int(kv[0]))):
        base = {s: (n, value) for s, n, value in class_values[("rf_baseline", label)]}
        balanced = {
            s: (n, value) for s, n, value in class_values[("rf_balanced", label)]
        }
        if set(base) != set(balanced) or any(
            base[s][0] != balanced[s][0] for s in base
        ):
            raise ValueError(f"inconsistent per-class test counts for {label}")
        common = sorted(base)
        diffs = [balanced[s][1] - base[s][1] for s in common]
        per_class.append(
            {
                "label": label,
                "dataset_count": count,
                "n_seeds_with_test_examples": len(common),
                "mean_test_count": statistics.mean(base[s][0] for s in common)
                if common
                else "",
                "baseline_mean_recall": statistics.mean(base[s][1] for s in common)
                if common
                else "",
                "balanced_mean_recall": statistics.mean(balanced[s][1] for s in common)
                if common
                else "",
                "mean_paired_delta": statistics.mean(diffs) if common else "",
            }
        )
    graph_rows = []
    by_run = {str(path.resolve()): seed for seed, path, *_ in runs}
    for path in args.graph_restarts:
        result = json.loads((path / "selection.json").read_text(encoding="utf-8"))
        run_path = str(Path(result["main_run"]).resolve())
        if run_path not in by_run:
            raise ValueError(f"graph restart does not match a main run: {path}")
        trials = [
            json.loads(line)
            for line in (path / "trials.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        if len(trials) != result["restarts"] or result["selected"] != max(
            trials, key=lambda x: x["modularity_at_1"]
        ):
            raise ValueError(f"inconsistent graph restart selection: {path}")
        if result["baseline"] != trials[0]:
            raise ValueError(f"inconsistent graph baseline: {path}")
        if not math.isclose(
            result["delta_modularity_at_1"],
            result["selected"]["modularity_at_1"] - trials[0]["modularity_at_1"],
            abs_tol=1e-12,
        ):
            raise ValueError(f"inconsistent graph delta: {path}")
        if not math.isclose(
            result["total_fit_seconds"],
            sum(t["fit_seconds"] for t in trials),
            abs_tol=1e-9,
        ):
            raise ValueError(f"inconsistent graph timing: {path}")
        graph_rows.append(
            {
                "seed": by_run[run_path],
                "baseline_modularity": result["baseline"]["modularity_at_1"],
                "best_modularity": result["selected"]["modularity_at_1"],
                "delta": result["delta_modularity_at_1"],
                "baseline_fit_seconds": result["baseline"]["fit_seconds"],
                "total_fit_seconds": result["total_fit_seconds"],
                "total_scoring_seconds": result["total_scoring_seconds"],
                "selected_seed": result["selected"]["seed"],
                "source": str(path / "selection.json"),
            }
        )
    if graph_rows and (
        {r["seed"] for r in graph_rows} != set(by_run.values())
        or len(graph_rows) != len(runs)
    ):
        raise ValueError("graph restarts must cover each main run exactly once")
    args.out.mkdir(parents=True, exist_ok=False)
    write_csv(
        args.out / "per_seed.csv",
        ["seed", "task", "model", "metric", "value", "source"],
        detailed,
    )
    write_csv(
        args.out / "mean_sd.csv",
        ["task", "model", "metric", "n", "mean", "sd", "minimum", "maximum"],
        grouped,
    )
    write_csv(
        args.out / "paired_deltas.csv",
        [
            "task",
            "model",
            "metric",
            "comparison",
            "n",
            "mean_delta",
            "sd_delta",
            "per_seed_delta",
        ],
        paired,
    )
    write_csv(
        args.out / "recall_by_class.csv",
        [
            "label",
            "dataset_count",
            "n_seeds_with_test_examples",
            "mean_test_count",
            "baseline_mean_recall",
            "balanced_mean_recall",
            "mean_paired_delta",
        ],
        per_class,
    )
    if graph_rows:
        write_csv(
            args.out / "graph_restarts.csv",
            [
                "seed",
                "baseline_modularity",
                "best_modularity",
                "delta",
                "baseline_fit_seconds",
                "total_fit_seconds",
                "total_scoring_seconds",
                "selected_seed",
                "source",
            ],
            sorted(graph_rows, key=lambda r: r["seed"]),
        )
    make_plot(args.out / "comparison.svg", grouped)
    (args.out / "provenance.json").write_text(
        json.dumps(
            {
                "runs": [
                    {"seed": seed, "path": str(path.resolve())}
                    for seed, path, *_ in runs
                ],
                "shared_config": signature,
                "notes": [
                    "Sample standard deviation uses n-1.",
                    "Each seed uses its own sampled nodes and splits; within a seed, models share the test set.",
                    "Graph comparisons use the same induced graph within each seed. Resolutions optimize different objectives; modularity_at_1 evaluates all partitions at a common resolution.",
                    "Timing depends on load. Model fitting excludes data loading, slicing, and metric computation.",
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Aggregated {len(runs)} runs to {args.out}")


if __name__ == "__main__":
    main()
