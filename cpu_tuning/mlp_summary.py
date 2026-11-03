"""Audit saved MLP predictions and summarize the predeclared ablations."""

import argparse
import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.dont_write_bytecode = True
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from .protocol import ARMS, scores, check_splits
from .io import read_array, sha256, dump, data_files


def write_csv(path, rows):
    if not rows:
        raise ValueError("empty output")
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def audit_class_rows(rows, predictions, y, train, labels, rare, floor):
    """Recompute the long-tail table from predictions and training membership."""
    by_key = {}
    for row in rows:
        key = (row["name"], int(row["label"]))
        if key in by_key:
            raise ValueError("duplicate per-class result")
        by_key[key] = row
    expected = {(name, int(label)) for name, _, _ in ARMS for label in labels}
    if set(by_key) != expected:
        raise ValueError("missing or unexpected per-class result")
    for name, _, ros in ARMS:
        precision, recall, f1, support = precision_recall_fscore_support(
            predictions["truth"], predictions[name], labels=labels, zero_division=0
        )
        for i, label in enumerate(labels):
            original = int(np.count_nonzero(y[train] == label))
            # Resampling can only duplicate classes present in training.
            fitted = max(original, floor) if ros and original else original
            expected_values = {
                "original_train_count": original,
                "fitted_train_count": fitted,
                "test_count": int(support[i]),
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
            }
            row = by_key[name, int(label)]
            if any(
                not math.isclose(float(row[k]), value, rel_tol=0, abs_tol=1e-12)
                for k, value in expected_values.items()
            ) or row["rare10"] != str(bool(label in rare)):
                raise ValueError(f"per-class result differs from predictions: {name}, {label}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", nargs="+", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--data", required=True, type=Path)
    p.add_argument(
        "--tuning-runs", nargs="+", type=Path,
        help="Completed tune runs for a matched tree/MLP classification comparison",
    )
    args = p.parse_args()
    if len(args.runs) < 2:
        p.error("at least two runs required")
    names = [a[0] for a in ARMS]
    seen = set()
    reference = None
    detailed = []
    paired = []
    selected = []
    class_rows = []
    curves = []
    protocols = []
    mlp_runs = []
    raw_y = None
    for path in args.runs:
        meta = json.loads((path / "protocol.json").read_text(encoding="utf-8"))
        signature = {
            k: meta[k]
            for k in (
                "protocol",
                "arms",
                "mlp_params",
                "max_epochs",
                "patience",
                "ros_floor",
                "source_sha256",
                "python",
                "numpy",
                "scipy",
                "sklearn",
                "threads",
            )
        }
        if not meta.get("timing_environment"):
            raise ValueError(
                "MLP run lacks timing environment; rerun before aggregation"
            )
        signature["timing_environment"] = meta["timing_environment"]
        if not meta.get("package_source_sha256"):
            raise ValueError("MLP run lacks source hashes; rerun before aggregation")
        signature["package_source_sha256"] = meta["package_source_sha256"]
        signature["data_hashes"] = {
            k: meta["files"][k]["sha256"] for k in ("features", "labels", "edges")
        }
        if reference is None:
            reference = signature
        elif reference != signature:
            raise ValueError("incompatible protocols/configuration/data")
        seed = meta["seed"]
        if seed in seen:
            raise ValueError("duplicate seed")
        seen.add(seed)
        source = path
        if sha256(source / "split_indices.npz") != meta["split_sha256"]:
            raise ValueError("source split changed")
        if raw_y is None:
            files = data_files(args.data, meta, ("features", "labels", "edges"))
            raw_y = read_array(files["labels"], np.int64).ravel()
        with np.load(source / "split_indices.npz", allow_pickle=False) as split:
            check_splits(
                [split["train"], split["validation"], split["test"]],
                split["sampled"],
                len(raw_y),
            )
            train = np.concatenate([split["train"], split["validation"]])
            test = split["test"].copy()
        train_labels, counts = np.unique(raw_y[train], return_counts=True)
        rare = train_labels[np.lexsort((train_labels, counts))[:10]]
        # Older runs did not store the vocabulary; they required all classes in
        # every split, so the full label file reconstructs the same vocabulary.
        labels = np.asarray(meta.get("class_labels", np.unique(raw_y)), dtype=np.int64)
        if not np.array_equal(labels, np.unique(raw_y)):
            raise ValueError("class vocabulary differs from the label file")
        records = [
            json.loads(line)
            for line in (path / "results.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        if any(r["status"] != "ok" for r in records):
            raise ValueError("failed experiment")
        summary = json.loads((path / "final_summary.json").read_text(encoding="utf-8"))
        val = summary["validation"]
        results = summary["test"]
        if [r["name"] for r in val] != names or [r["name"] for r in results] != names:
            raise ValueError("missing/duplicate arms")
        if [r for r in records if r["stage"] == "validation"] != val or [
            r for r in records if r["stage"] == "test"
        ] != results:
            raise ValueError("summary differs from raw result records")
        best = max(
            val, key=lambda r: (r["metrics"]["macro_f1"], r["metrics"]["accuracy"])
        )
        if best["name"] != summary["selected"]:
            raise ValueError("incorrect validation selection")
        for v, r in zip(val, results):
            if r["epochs"] != v["best_epoch"] or r["rare10_labels"] != rare.tolist():
                raise ValueError("incorrect epoch/rare-class selection")
            curve = [
                x
                for x in records
                if x["stage"] == "learning_curve" and x["name"] == r["name"]
            ]
            best_point = max(
                curve,
                key=lambda x: (x["metrics"]["macro_f1"], x["metrics"]["accuracy"]),
            )
            if (
                best_point["epoch"] != v["best_epoch"]
                or best_point["metrics"] != v["metrics"]
            ):
                raise ValueError("epoch selection differs from learning curve")
        with np.load(path / "test_predictions.npz", allow_pickle=False) as pred:
            if not np.array_equal(pred["node_ids"], test) or not np.array_equal(
                pred["truth"], raw_y[test]
            ):
                raise ValueError("prediction IDs/targets mismatch")
            for r in results:
                computed = scores(raw_y[test], pred[r["name"]], labels, rare)
                if any(
                    not math.isclose(value, r["metrics"][k], abs_tol=1e-12)
                    for k, value in computed.items()
                ):
                    raise ValueError("metrics do not match saved predictions")
            with (path / "per_class.csv").open(encoding="utf-8-sig", newline="") as f:
                per_class = list(csv.DictReader(f))
            audit_class_rows(
                per_class, pred, raw_y, train, labels, rare, meta["ros_floor"]
            )
        selection_record = next(r for r in records if r["stage"] == "selection")
        if selection_record["selected"] != best["name"]:
            raise ValueError("selection record differs from validation selection")
        mlp_runs.append(dict(
            path=path, meta=meta, validation=val, test=results, selection=selection_record
        ))
        by_name = {r["name"]: r for r in results}
        for r in results:
            metrics = dict(
                r["metrics"],
                fit_seconds=r["fit_seconds"],
                predict_seconds=r["predict_seconds"],
                prep_seconds=r["prep_seconds"],
            )
            metrics["validation_search_fit_seconds"] = next(
                v["fit_seconds"] for v in val if v["name"] == r["name"]
            )
            for metric, value in metrics.items():
                detailed.append(
                    dict(
                        seed=seed,
                        name=r["name"],
                        metric=metric,
                        value=value,
                        source=f"{path.name}/results.jsonl",
                    )
                )
        selected.append(
            dict(
                seed=seed,
                selected=best["name"],
                best_epoch=best["best_epoch"],
                validation_macro_f1=best["metrics"]["macro_f1"],
                test_accuracy=by_name[best["name"]]["metrics"]["accuracy"],
                test_macro_f1=by_name[best["name"]]["metrics"]["macro_f1"],
                test_rare10_macro_f1=by_name[best["name"]]["metrics"][
                    "rare10_macro_f1"
                ],
                search_seconds=selection_record["search_seconds"],
            )
        )
        for left, right in [
            ("raw_ros", "raw"),
            ("hop1", "raw"),
            ("hop2", "raw"),
            ("hop2", "hop1"),
            ("hop2_ros", "hop2"),
        ]:
            for metric in ("accuracy", "macro_f1", "rare10_macro_f1"):
                paired.append(
                    dict(
                        seed=seed,
                        comparison=left + " minus " + right,
                        metric=metric,
                        delta=by_name[left]["metrics"][metric]
                        - by_name[right]["metrics"][metric],
                    )
                )
        for r in records:
            if r["stage"] == "learning_curve":
                curves.append(
                    dict(
                        seed=seed,
                        name=r["name"],
                        epoch=r["epoch"],
                        loss=r["loss"],
                        **r["metrics"],
                    )
                )
        class_rows.extend(dict(seed=seed, **r) for r in per_class)
        protocols.append(
            {
                "seed": seed,
                "path": path.name,
                "cache": meta["graph_cache"],
                "source_split_sha256": meta["split_sha256"],
                "warnings": {r["name"]: r["warnings"] for r in results},
            }
        )
    groups = defaultdict(list)
    for r in detailed:
        groups[(r["name"], r["metric"])].append(r["value"])
    means = [
        dict(
            name=name,
            metric=metric,
            n=len(v),
            mean=statistics.mean(v),
            sd=statistics.stdev(v),
        )
        for (name, metric), v in sorted(groups.items())
    ]
    pair_groups = defaultdict(list)
    for r in paired:
        pair_groups[(r["comparison"], r["metric"])].append(r["delta"])
    pair_means = [
        dict(
            comparison=c,
            metric=m,
            n=len(v),
            mean_delta=statistics.mean(v),
            sd_delta=statistics.stdev(v),
            positive_seeds=sum(x > 0 for x in v),
            negative_seeds=sum(x < 0 for x in v),
        )
        for (c, m), v in sorted(pair_groups.items())
    ]
    comparison = None
    if args.tuning_runs:
        from .comparison import build_comparison

        comparison = build_comparison(args.tuning_runs, mlp_runs)
    args.out.mkdir(parents=True, exist_ok=False)
    if comparison is not None:
        for filename, rows in zip(
            ("model_comparison.csv", "model_comparison_mean_sd.csv",
             "model_comparison_paired.csv", "pipeline_costs.csv"), comparison[:4]
        ):
            write_csv(args.out / filename, rows)
    for filename, rows in [
        ("per_seed.csv", detailed),
        ("mean_sd.csv", means),
        ("paired_deltas.csv", paired),
        ("paired_mean_sd.csv", pair_means),
        ("selected_by_validation.csv", selected),
        ("per_class.csv", class_rows),
        ("learning_curves.csv", curves),
    ]:
        write_csv(args.out / filename, rows)
    write_csv(
        args.out / "validation_selected_mean_sd.csv",
        [
            dict(
                metric=k,
                n=len(selected),
                mean=statistics.mean(r[k] for r in selected),
                sd=statistics.stdev(r[k] for r in selected),
            )
            for k in ("test_accuracy", "test_macro_f1", "test_rare10_macro_f1")
        ],
    )
    feature_count = protocols[0]["cache"]["signature"]["shape"][1]
    write_csv(
        args.out / "architecture.csv",
        [
            dict(
                name=name,
                input_dimensions=(order + 1) * feature_count,
                hidden_units=128,
                classes=len(labels),
                trainable_parameters=((order + 1) * feature_count + 1) * 128
                + (128 + 1) * len(labels),
            )
            for name, order, _ in ARMS
        ],
    )
    dump(
        args.out / "audit.json",
        {
            "prediction_metrics_verified": True,
            "per_class_metrics_verified": True,
            "runs": protocols,
            "shared_protocol": reference,
            "model_comparison": comparison[4] if comparison is not None else [],
            "cost_notes": [
                "Model search_seconds includes per-arm MLP validation or the complete RF parameter search; fixed tree controls have no search.",
                "pipeline_costs reports the shared five-arm MLP search separately; do not add it to per-arm search times.",
                "Graph compute/save costs describe a shared cache build, not a new build for every seed; cache_access may include that build.",
                "Final fit/predict exclude preprocessing. Graph MLP arms use additional transductive information.",
            ],
            "note": "Sample SD uses n-1; test splits previously seen in tree experiments; graph feature arms are transductive.",
        },
    )
    print("Audited predictions and summarized", len(args.runs), "runs to", args.out)


if __name__ == "__main__":
    main()
