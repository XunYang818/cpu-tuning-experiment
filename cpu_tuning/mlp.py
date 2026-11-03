"""Controlled CPU MLP ablations using saved splits and label-free graph features."""

import argparse
import csv
import os
import sys
import time
import traceback
import warnings
from pathlib import Path

import numpy as np
import scipy
import sklearn
from sklearn.metrics import precision_recall_fscore_support
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from .io import (
    read_array, sha256, dump, record, data_files, timing_environment, completed_run,
)
from .protocol import ARMS, MLP_PARAMS, resample_ids, check_splits, scores
from .features import graph_cache, make_input


def prepare(X, hops, y, tr, ev, order, ros, args):
    started = time.perf_counter()
    fit_ids = resample_ids(tr, y, args.ros_floor, args.seed) if ros else np.asarray(tr)
    labels, counts = np.unique(y[tr], return_counts=True)
    rare = labels[np.lexsort((labels, counts))[:10]]
    scaler = StandardScaler()
    A = scaler.fit_transform(make_input(X, hops, fit_ids, order))
    B = scaler.transform(make_input(X, hops, ev, order))
    return A, y[fit_ids], B, rare, fit_ids, time.perf_counter() - started


def validate_arm(out, name, order, ros, X, hops, y, tr, val, args, labels):
    A, yt, B, rare, fit_ids, prep_time = prepare(X, hops, y, tr, val, order, ros, args)
    model = MLPClassifier(random_state=args.seed, **MLP_PARAMS)
    best_key = (-1.0, -1.0)
    best_epoch = 0
    stale = 0
    train_seconds = 0
    predict_seconds = 0
    best_metrics = None
    caught_messages = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for epoch in range(1, args.max_epochs + 1):
            tic = time.perf_counter()
            model.partial_fit(A, yt, classes=labels)
            train_seconds += time.perf_counter() - tic
            tic = time.perf_counter()
            pred = model.predict(B)
            predict_seconds += time.perf_counter() - tic
            metrics = scores(y[val], pred, labels, rare)
            key = (metrics["macro_f1"], metrics["accuracy"])
            if key > best_key:
                best_key = key
                best_epoch = epoch
                best_metrics = metrics
                stale = 0
            else:
                stale += 1
            record(
                out,
                {
                    "stage": "learning_curve",
                    "name": name,
                    "epoch": epoch,
                    "loss": float(model.loss_),
                    "metrics": metrics,
                    "status": "ok",
                    "seed": args.seed,
                },
            )
            if stale >= args.patience:
                break
        caught_messages = [str(w.message) for w in caught]
    r = {
        "stage": "validation",
        "name": name,
        "status": "ok",
        "seed": args.seed,
        "order": order,
        "resampling": ros,
        "best_epoch": best_epoch,
        "epochs_run": epoch,
        "metrics": best_metrics,
        "prep_seconds": prep_time,
        "fit_seconds": train_seconds,
        "predict_seconds": predict_seconds,
        "n_train_original": len(tr),
        "n_train_fit": len(fit_ids),
        "rare10_labels": rare.tolist(),
        "warnings": caught_messages,
        "stop_reason": "patience" if stale >= args.patience else "max_epochs",
        "hit_epoch_limit": epoch == args.max_epochs,
    }
    record(out, r)
    return r


def final_arm(
    out, selection, X, hops, y, tr, test, args, labels, predictions, class_rows
):
    name = selection["name"]
    order = selection["order"]
    ros = selection["resampling"]
    A, yt, B, rare, fit_ids, prep_time = prepare(X, hops, y, tr, test, order, ros, args)
    model = MLPClassifier(random_state=args.seed, **MLP_PARAMS)
    tic = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for _ in range(selection["best_epoch"]):
            model.partial_fit(A, yt, classes=labels)
    fit_time = time.perf_counter() - tic
    tic = time.perf_counter()
    pred = model.predict(B)
    pred_time = time.perf_counter() - tic
    predictions[name] = pred
    precision, recall, f1, support = precision_recall_fscore_support(
        y[test], pred, labels=labels, zero_division=0
    )
    for i, label in enumerate(labels):
        class_rows.append(
            [
                name,
                int(label),
                int(np.count_nonzero(y[tr] == label)),
                int(np.count_nonzero(yt == label)),
                int(support[i]),
                float(precision[i]),
                float(recall[i]),
                float(f1[i]),
                bool(label in rare),
            ]
        )
    r = {
        "stage": "test",
        "name": name,
        "status": "ok",
        "seed": args.seed,
        "order": order,
        "resampling": ros,
        "epochs": selection["best_epoch"],
        "metrics": scores(y[test], pred, labels, rare),
        "prep_seconds": prep_time,
        "fit_seconds": fit_time,
        "predict_seconds": pred_time,
        "n_train_original": len(tr),
        "n_train_fit": len(fit_ids),
        "n_test": len(test),
        "rare10_labels": rare.tolist(),
        "warnings": [str(w.message) for w in caught],
    }
    record(out, r)
    return r


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--run", required=True, type=Path, help="Completed v2 run with fixed splits"
    )
    p.add_argument(
        "--cache",
        required=True,
        type=Path,
        help="Shared label-free graph feature cache",
    )
    p.add_argument("--out", required=True, type=Path)
    p.add_argument(
        "--data", required=True, type=Path, help="Dataset directory or raw directory"
    )
    p.add_argument("--max-epochs", type=int, default=200)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--ros-floor", type=int, default=200)
    a = p.parse_args()
    if min(a.max_epochs, a.patience, a.ros_floor) < 1:
        p.error("epoch, patience and floor must be positive")
    source, _ = completed_run(a.run, ("split_indices.npz", "results.jsonl"))
    a.seed = source["arguments"]["seed"]
    a.out.mkdir(parents=True, exist_ok=False)
    try:
        meta = {
            "protocol": "mlp-external-validation-v1",
            "source_run": a.run.name,
            "seed": a.seed,
            "arms": ARMS,
            "mlp_params": MLP_PARAMS,
            "max_epochs": a.max_epochs,
            "patience": a.patience,
            "ros_floor": a.ros_floor,
            "selection": "per-arm epoch: validation macro-F1 then accuracy; model: same criterion, ties in declared arm order",
            "setting": "raw is feature-only; hop1/hop2 use full unlabeled graph: transductive",
            "test_history": "these test splits were evaluated by earlier tree experiments; this is a predeclared follow-up, not a fresh blind test",
            "files": source["files"],
            "split_sha256": sha256(a.run / "split_indices.npz"),
            "source_sha256": sha256(Path(__file__)),
            "package_source_sha256": {
                p.name: sha256(p) for p in sorted(Path(__file__).parent.glob("*.py"))
            },
            "numpy": np.__version__,
            "python": sys.version,
            "scipy": scipy.__version__,
            "sklearn": sklearn.__version__,
            "threadpools": timing_environment()["threadpools"],
            "timing_environment": timing_environment(),
            "threads": {
                k: os.environ.get(k)
                for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
            },
        }
        dump(
            a.out / "protocol.json", meta
        )  # Freeze all arms and selection rules before evaluation.
        files = data_files(a.data, source, ("features", "labels", "edges"))
        X = read_array(files["features"], np.float32)
        yf = read_array(files["labels"], np.float64)
        if (
            yf.shape != (len(X), 1)
            or not np.isfinite(X).all()
            or not np.isfinite(yf).all()
            or not np.equal(yf, np.floor(yf)).all()
        ):
            raise ValueError("invalid features or labels")
        y = yf[:, 0].astype(np.int64)
        # The baseline records the dataset's class vocabulary before splitting.
        labels = np.array(
            sorted(map(int, source["data"]["label_counts"])), dtype=np.int64
        )
        if not np.array_equal(labels, np.unique(y)):
            raise ValueError("baseline class vocabulary differs from the label file")
        meta["class_labels"] = labels.tolist()
        with np.load(a.run / "split_indices.npz", allow_pickle=False) as z:
            tr, val, test, sampled = [
                z[k].copy() for k in ("train", "validation", "test", "sampled")
            ]
        check_splits([tr, val, test], sampled, len(X))
        np.savez_compressed(
            a.out / "split_indices.npz",
            train=tr,
            validation=val,
            test=test,
            sampled=sampled,
        )
        meta["source_split_sha256"] = meta["split_sha256"]
        meta["split_sha256"] = sha256(a.out / "split_indices.npz")
        tic = time.perf_counter()
        hops, cache_meta = graph_cache(
            a.cache,
            X,
            files["edges"],
            source["files"]["features"]["sha256"],
        )
        meta["cache_access_seconds"] = time.perf_counter() - tic
        meta["graph_cache"] = cache_meta
        dump(a.out / "protocol.json", meta)
        start = time.perf_counter()
        selections = [
            validate_arm(a.out, name, order, ros, X, hops, y, tr, val, a, labels)
            for name, order, ros in ARMS
        ]
        chosen = max(
            selections,
            key=lambda r: (r["metrics"]["macro_f1"], r["metrics"]["accuracy"]),
        )
        record(
            a.out,
            {
                "stage": "selection",
                "status": "ok",
                "seed": a.seed,
                "selected": chosen["name"],
                "criterion": "validation macro_f1, then accuracy",
                "search_seconds": time.perf_counter() - start,
            },
        )
        # Test labels are accessed for scoring only after every validation choice is fixed.
        predictions = {"node_ids": test, "truth": y[test]}
        class_rows = []
        final = [
            final_arm(
                a.out,
                s,
                X,
                hops,
                y,
                np.concatenate([tr, val]),
                test,
                a,
                labels,
                predictions,
                class_rows,
            )
            for s in selections
        ]
        np.savez_compressed(a.out / "test_predictions.npz", **predictions)
        with (a.out / "per_class.csv").open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "name",
                    "label",
                    "original_train_count",
                    "fitted_train_count",
                    "test_count",
                    "precision",
                    "recall",
                    "f1",
                    "rare10",
                ]
            )
            w.writerows(class_rows)
        dump(
            a.out / "final_summary.json",
            {"validation": selections, "selected": chosen["name"], "test": final},
        )
    except Exception:
        record(
            a.out, {"stage": "run", "status": "error", "error": traceback.format_exc()}
        )
        raise


if __name__ == "__main__":
    main()
