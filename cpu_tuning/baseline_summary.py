"""Aggregate complete, comparable experiment runs without accessing the test data."""

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from xml.sax.saxutils import escape
from .io import sha256, same_host, completed_run, verify_artifacts

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
    if not meta.get("package_source_sha256"):
        raise ValueError("baseline lacks source hashes; rerun it before aggregation")
    if not meta.get("timing_environment"):
        raise ValueError("baseline lacks timing environment; rerun before aggregation")
    return {
        "timing_environment": meta["timing_environment"],
        "package_source_sha256": meta["package_source_sha256"],
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
        meta, records = completed_run(path, ("results.jsonl",))
        tasks = set(meta["arguments"]["tasks"])
        if not tasks or not tasks <= {"classification", "regression", "graph"}:
            raise ValueError(f"invalid task selection: {path}")
        artifacts = []
        if tasks & {"classification", "regression"}:
            artifacts.append("split_indices.npz")
        if "graph" in tasks:
            artifacts.append("graph_nodes.npy")
        if "classification" in tasks:
            artifacts.append("class_recall.csv")
        verify_artifacts(path, meta, artifacts)
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
        expected = {key for key in EXPECTED if key[0] in tasks}
        if set(result) != expected:
            raise ValueError(
                f"missing or unexpected result in {path}: {set(result) ^ expected}"
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
    panels = [
        panel for panel in panels
        if any(r["task"] == panel[0] and r["metric"] == panel[1] for r in grouped)
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
        scale = bound or max(r["mean"] + r["sd"] for r in rows) * 1.1 or 1.0
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


def load_class_recalls(path, meta, results):
    """Validate every model/class row before using the long-tail comparison."""
    labels = meta["data"]["label_counts"]
    models = {
        name: r for (task, name), r in results.items() if task == "classification"
    }
    expected = {(model, label) for model in models for label in labels}
    with (path / "class_recall.csv").open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    seen = set()
    counts = {}
    validated = []
    for row in rows:
        try:
            key = (row["model"], row["label"])
            count = int(row["test_count"])
            recall = float(row["recall"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid per-class recall row: {path}") from exc
        if key in seen:
            raise ValueError(f"duplicate per-class recall: {path} {key}")
        if key not in expected:
            raise ValueError(f"unexpected per-class recall: {path} {key}")
        model, label = key
        if (
            not math.isfinite(recall)
            or not 0 <= recall <= 1
            or not 0 <= count <= min(models[model]["n_eval"], labels[label])
            or (count == 0 and recall != 0)
        ):
            raise ValueError(f"invalid per-class count or recall: {path} {key}")
        if label in counts and counts[label] != count:
            raise ValueError(f"inconsistent per-class test counts: {path} {label}")
        counts[label] = count
        seen.add(key)
        validated.append(dict(model=model, label=label, test_count=count, recall=recall))
    if seen != expected:
        raise ValueError(f"missing per-class recall: {path}")
    if any(sum(counts.values()) != r["n_eval"] for r in models.values()):
        raise ValueError(f"per-class test counts differ from evaluation size: {path}")
    return validated


def summarize_class_recalls(runs):
    # Per-class recall: keep the original full-dataset class size and exclude
    # seeds where a class has no test examples instead of treating it as zero.
    labels = runs[0][2]["data"]["label_counts"]
    class_values = defaultdict(list)
    for seed, path, meta, records, result in runs:
        if meta["data"]["label_counts"] != labels:
            raise ValueError("class counts differ between runs")
        for r in load_class_recalls(path, meta, result):
            key = (r["model"], r["label"])
            if (
                r["model"] in ("rf_baseline", "rf_balanced")
                and r["test_count"] > 0
            ):
                class_values[key].append(
                    (seed, r["test_count"], r["recall"])
                )
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
    return per_class


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", type=Path, required=True)
    parser.add_argument("--graph-restarts", nargs="*", type=Path, default=[])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if len(args.runs) < 2:
        parser.error("at least two complete runs required")
    runs, signature = load_runs(args.runs)
    tasks = set(signature["tasks"])
    if args.graph_restarts and "graph" not in tasks:
        parser.error("graph-restarts requires runs containing the graph task")
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
                        "source": f"{path.name}/results.jsonl",
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
                        "source": f"{path.name}/results.jsonl",
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
        if task not in tasks:
            continue
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
    per_class = summarize_class_recalls(runs) if "classification" in tasks else []
    graph_rows = []
    by_run = {
        (sha256(path / "metadata.json"), sha256(path / "graph_nodes.npy")): seed
        for seed, path, *_ in runs
    } if "graph" in tasks else {}
    for path in args.graph_restarts:
        result = json.loads((path / "selection.json").read_text(encoding="utf-8"))
        if (not same_host(result.get("timing_environment", {}), signature["timing_environment"])
                or result.get("thread_environment") != signature["thread_environment"]):
            raise ValueError("graph restart timing environment differs from main runs")
        run_path = (result.get("main_metadata_sha256"), result.get("graph_nodes_sha256"))
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
                "source": f"{path.name}/selection.json",
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
    if "classification" in tasks:
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
                    {"seed": seed, "path": path.name}
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
