"""Compare audited tree and MLP results on the same saved classification splits."""

import math
import statistics
from collections import defaultdict

import numpy as np

from .baseline_summary import load_runs


QUALITY = ("accuracy", "macro_f1", "weighted_f1")
TIMES = ("fit_seconds", "predict_seconds", "search_seconds")


def checked_number(value, name, *, duration=False):
    if (
        not isinstance(value, (int, float))
        or not math.isfinite(value)
        or (duration and value < 0)
    ):
        raise ValueError(f"invalid comparison value: {name}")
    return value


def match_run(tree_path, tree_meta, mlp_path, mlp_meta):
    """Reject pairs that only share a seed but differ in data, splits or runtime."""
    for key in (
        "python", "numpy", "sklearn", "package_source_sha256", "timing_environment"
    ):
        if not tree_meta.get(key) or tree_meta[key] != mlp_meta.get(key):
            raise ValueError(f"tree/MLP comparison differs in {key}")
    if tree_meta["thread_environment"] != mlp_meta["threads"]:
        raise ValueError("tree/MLP comparison differs in thread settings")
    for key in ("features", "labels", "edges"):
        if tree_meta["files"][key]["sha256"] != mlp_meta["files"][key]["sha256"]:
            raise ValueError(f"tree/MLP comparison differs in {key} data")
    if (
        tree_meta["artifact_sha256"]["split_indices.npz"]
        != mlp_meta.get("source_split_sha256")
    ):
        raise ValueError("tree/MLP comparison differs in source split")
    with (
        np.load(tree_path / "split_indices.npz", allow_pickle=False) as tree_split,
        np.load(mlp_path / "split_indices.npz", allow_pickle=False) as mlp_split,
    ):
        for key in ("sampled", "train", "validation", "test"):
            if not np.array_equal(tree_split[key], mlp_split[key]):
                raise ValueError(f"tree/MLP comparison differs in {key} IDs")
        return len(tree_split["train"]) + len(tree_split["validation"]), len(tree_split["test"])


def build_comparison(tuning_paths, mlp_runs):
    """Return per-seed rows, mean/SD, paired deltas, shared costs and provenance."""
    tree_runs, _ = load_runs(tuning_paths)
    by_seed = {seed: (path, meta, records, results) for seed, path, meta, records, results in tree_runs}
    if set(by_seed) != {run["meta"]["seed"] for run in mlp_runs}:
        raise ValueError("tree/MLP comparison requires exactly the same seeds")
    rows, costs, provenance = [], [], []
    for run in mlp_runs:
        mlp_path, mlp_meta = run["path"], run["meta"]
        seed = mlp_meta["seed"]
        tree_path, tree_meta, records, results = by_seed[seed]
        if "classification" not in tree_meta["arguments"]["tasks"]:
            raise ValueError("tree/MLP comparison requires classification results")
        n_train, n_test = match_run(tree_path, tree_meta, mlp_path, mlp_meta)
        searches = [r for r in records if r["stage"] == "selection" and r["task"] == "classification"]
        if len(searches) != 1:
            raise ValueError("missing or duplicate tree parameter search")
        tree_search = searches[0]["search_seconds"]
        seed_rows = []
        for (task, name), result in sorted(results.items()):
            if task != "classification":
                continue
            if result["n_train"] != n_train or result["n_eval"] != n_test:
                raise ValueError("tree result sizes differ from saved splits")
            seed_rows.append(dict(
                seed=seed, name=name, setting="feature-only", selected_mlp=False,
                n_train=n_train, n_test=n_test, **{k: result["metrics"][k] for k in QUALITY},
                fit_seconds=result["fit_seconds"], predict_seconds=result["predict_seconds"],
                search_seconds=tree_search if name == "rf_tuned" else 0.0,
                source=f"{tree_path.name}/results.jsonl",
            ))
        validation = {r["name"]: r for r in run["validation"]}
        for result in run["test"]:
            if result["n_train_original"] != n_train or result["n_test"] != n_test:
                raise ValueError("MLP result sizes differ from saved splits")
            search = validation[result["name"]].get("search_seconds")
            if search is None:
                raise ValueError("MLP run lacks per-arm search time; rerun for model comparison")
            seed_rows.append(dict(
                seed=seed, name="mlp_" + result["name"],
                setting="transductive" if result["order"] else "feature-only",
                selected_mlp=result["name"] == run["selection"]["selected"],
                n_train=n_train, n_test=n_test, **{k: result["metrics"][k] for k in QUALITY},
                fit_seconds=result["fit_seconds"], predict_seconds=result["predict_seconds"],
                search_seconds=search, source=f"{mlp_path.name}/results.jsonl",
            ))
        for row in seed_rows:
            for metric in (*QUALITY, *TIMES):
                checked_number(row[metric], metric, duration=metric in TIMES)
        rows.extend(seed_rows)
        cache = mlp_meta["graph_cache"]
        cost = dict(
            seed=seed, mlp_all_arms_search_seconds=run["selection"]["search_seconds"],
            graph_compute_seconds=cache["compute_seconds"], graph_save_seconds=cache["save_seconds"],
            cache_access_seconds=mlp_meta["cache_access_seconds"],
        )
        for key, value in cost.items():
            if key != "seed":
                checked_number(value, key, duration=True)
        costs.append(cost)
        provenance.append(dict(
            seed=seed, tuning_run=tree_path.name, mlp_run=mlp_path.name,
            split_sha256=mlp_meta["source_split_sha256"],
            timing_environment_matched=True,
        ))
    grouped = defaultdict(list)
    for row in rows:
        for metric in (*QUALITY, *TIMES):
            grouped[row["name"], row["setting"], metric].append(row[metric])
    means = [dict(
        name=name, setting=setting, metric=metric, n=len(values),
        mean=statistics.mean(values), sd=statistics.stdev(values),
    ) for (name, setting, metric), values in sorted(grouped.items())]
    reference = {row["seed"]: row for row in rows if row["name"] == "rf_baseline"}
    deltas = defaultdict(list)
    for row in rows:
        if row["name"] != "rf_baseline":
            for metric in (*QUALITY, *TIMES):
                deltas[row["name"], row["setting"], metric].append(
                    row[metric] - reference[row["seed"]][metric]
                )
    paired = [dict(
        name=name, setting=setting, comparison="model minus rf_baseline", metric=metric,
        n=len(values), mean_delta=statistics.mean(values), sd_delta=statistics.stdev(values),
    ) for (name, setting, metric), values in sorted(deltas.items())]
    return rows, means, paired, costs, provenance
