"""Teacher-style CPU timing: classifiers, regressors and full-data forest."""

import argparse
import os
import platform
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import networkx as nx
import sklearn
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    HistGradientBoostingRegressor,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from .io import dump, read_array, read_edges, record, sha256, timing_environment


def measure(
    out, name, model, train_x, train_y, test_x, test_y, metric, jobs=None, **details
):
    """Time fit and predict only; keep preprocessing and scoring outside timers."""
    row = {
        "name": name,
        "n_jobs": jobs,
        "n_train": len(train_y),
        "n_test": len(test_y),
        "model_params": model.get_params(deep=False),
        **details,
    }
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            start = time.perf_counter()
            model.fit(train_x, train_y)
            row["fit_seconds"] = time.perf_counter() - start
            start = time.perf_counter()
            pred = model.predict(test_x)
            row["predict_seconds"] = time.perf_counter() - start
        row.update(
            status="ok",
            score_name=metric.__name__,
            score=float(metric(test_y, pred)),
            warnings=[str(w.message) for w in caught],
        )
        if hasattr(model, "estimators_"):
            row["mean_nodes"] = float(
                np.mean([t.tree_.node_count for t in model.estimators_])
            )
            row["mean_depth"] = float(
                np.mean([t.tree_.max_depth for t in model.estimators_])
            )
        if hasattr(model, "n_iter_"):
            row["iterations"] = np.asarray(model.n_iter_).tolist()
    except Exception as exc:
        row.update(status="error", error=str(exc))
        record(out, row)
        raise
    record(out, row)


def graph_benchmark(out, edge_path, node_count, nodes, seed):
    """Preserve the teacher's graph: self-loops retained, isolates omitted."""
    np.save(out / "graph_nodes.npy", nodes)
    edges = read_edges(edge_path, node_count)
    mask = np.isin(edges[:, 0], nodes) & np.isin(edges[:, 1], nodes)
    graph = nx.Graph()
    graph.add_edges_from(edges[mask].tolist())
    if not graph.number_of_edges():
        raise ValueError("induced graph has no edges; increase graph-nodes")
    row = {
        "name": "louvain_teacher",
        "sampled_nodes": len(nodes),
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "self_loops": nx.number_of_selfloops(graph),
        "seed": seed,
    }
    try:
        start = time.perf_counter()
        communities = nx.community.louvain_communities(graph, seed=seed)
        row["fit_seconds"] = time.perf_counter() - start
        row.update(
            status="ok",
            communities=len(communities),
            modularity=float(nx.community.modularity(graph, communities)),
        )
    except Exception as exc:
        row.update(status="error", error=str(exc))
        record(out, row)
        raise
    record(out, row)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--samples", type=int, default=50000)
    p.add_argument("--trees", type=int, default=150)
    p.add_argument("--graph-nodes", type=int, default=30000)
    p.add_argument(
        "--skip-graph", action="store_true", help="Skip teacher-style Louvain benchmark"
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--jobs", type=int, nargs="+", default=[1, 2, 4])
    p.add_argument(
        "--skip-full",
        action="store_true",
        help="Skip the costly full-data forest for a trial run",
    )
    a = p.parse_args()
    if min(a.samples, a.trees, a.graph_nodes, *a.jobs) < 1:
        p.error("samples, trees and jobs must be positive")
    raw = a.data / "raw" if (a.data / "raw").is_dir() else a.data
    year = next(
        (
            raw / n
            for n in ("node-year.csv.gz", "node_year.csv.gz")
            if (raw / n).is_file()
        ),
        None,
    )
    files = {
        "features": raw / "node-feat.csv.gz",
        "labels": raw / "node-label.csv.gz",
        "year": year,
    }
    if not a.skip_graph:
        files["edges"] = raw / "edge.csv.gz"
    if any(v is None or not v.is_file() for v in files.values()):
        raise FileNotFoundError("features, labels and year gzip files are required")
    X = read_array(files["features"], np.float32)
    labels = read_array(files["labels"], np.float64)
    years = read_array(files["year"], np.float64)
    if (
        labels.shape != (len(X), 1)
        or years.shape != (len(X), 1)
        or not all(np.isfinite(v).all() for v in (X, labels, years))
        or not np.equal(labels, np.floor(labels)).all()
        or a.samples > len(X)
        or (not a.skip_graph and a.graph_nodes > len(X))
    ):
        raise ValueError("invalid data shape, values, labels or sample size")
    y, yr = labels[:, 0].astype(np.int64), years[:, 0]
    rng = np.random.RandomState(a.seed)
    sampled = rng.choice(len(X), a.samples, replace=False)
    tr, te = train_test_split(
        sampled, test_size=0.3, stratify=y[sampled], random_state=a.seed
    )
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X[tr].astype(np.float64))
    Xte = scaler.transform(X[te].astype(np.float64))
    a.out.mkdir(parents=True, exist_ok=False)
    dump(
        a.out / "metadata.json",
        {
            "arguments": {k: v for k, v in vars(a).items() if k not in ("data", "out")},
            "files": {
                k: {"name": v.name, "sha256": sha256(v)} for k, v in files.items()
            },
            "package_source_sha256": {
                p.name: sha256(p) for p in sorted(Path(__file__).parent.glob("*.py"))
            },
            "logical_cpus": os.cpu_count(),
            "platform": platform.platform(),
            "python": sys.version,
            "networkx": nx.__version__,
            "timing_environment": timing_environment(),
            "numpy": np.__version__,
            "sklearn": sklearn.__version__,
            "threadpools": timing_environment()["threadpools"],
            "thread_environment": {
                k: os.environ.get(k)
                for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
            },
        },
    )
    np.savez_compressed(a.out / "split_indices.npz", sampled=sampled, train=tr, test=te)

    # Keep the same task order as the supplied teacher script.
    for jobs in a.jobs:
        measure(
            a.out,
            "rf_classification",
            RandomForestClassifier(
                n_estimators=a.trees,
                min_samples_leaf=5,
                n_jobs=jobs,
                random_state=a.seed,
            ),
            X[tr],
            y[tr],
            X[te],
            y[te],
            accuracy_score,
            jobs,
        )
    measure(
        a.out,
        "mlp",
        MLPClassifier(
            hidden_layer_sizes=(128,),
            max_iter=500,
            early_stopping=True,
            n_iter_no_change=20,
            learning_rate_init=3e-4,
            alpha=1e-3,
            random_state=a.seed,
        ),
        Xtr,
        y[tr],
        Xte,
        y[te],
        accuracy_score,
    )
    # Multinomial SAGA does not gain four-way parallelism from n_jobs.
    # Recent sklearn versions deprecate the argument, so record the difference.
    measure(
        a.out,
        "logistic_saga",
        LogisticRegression(
            solver="saga", max_iter=300, tol=1e-3, C=0.1, random_state=a.seed
        ),
        Xtr,
        y[tr],
        Xte,
        y[te],
        accuracy_score,
        teacher_n_jobs=4,
        parallelism_note="n_jobs omitted; use numerical thread environment",
    )
    for jobs in a.jobs:
        measure(
            a.out,
            "rf_regression",
            RandomForestRegressor(
                n_estimators=a.trees,
                min_samples_leaf=5,
                max_features="sqrt",
                n_jobs=jobs,
                random_state=a.seed,
            ),
            X[tr],
            yr[tr],
            X[te],
            yr[te],
            r2_score,
            jobs,
        )
    measure(
        a.out,
        "hist_gradient_boosting",
        HistGradientBoostingRegressor(
            max_iter=300,
            learning_rate=0.05,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
            random_state=a.seed,
        ),
        X[tr],
        yr[tr],
        X[te],
        yr[te],
        r2_score,
    )
    if not a.skip_full:
        full_tr, full_te = train_test_split(
            np.arange(len(X)), test_size=0.2, stratify=y, random_state=a.seed
        )
        np.savez_compressed(
            a.out / "full_split_indices.npz", train=full_tr, test=full_te
        )
        measure(
            a.out,
            "rf_full_classification",
            RandomForestClassifier(
                n_estimators=a.trees, min_samples_leaf=5, n_jobs=2, random_state=a.seed
            ),
            X[full_tr],
            y[full_tr],
            X[full_te],
            y[full_te],
            accuracy_score,
            2,
        )
    if not a.skip_graph:
        # The teacher reuses the sampling RNG, retains self-loops, and adds
        # only endpoints of induced edges (isolated sampled nodes are omitted).
        nodes = rng.choice(len(X), a.graph_nodes, replace=False)
        graph_benchmark(a.out, files["edges"], len(X), nodes, a.seed)


if __name__ == "__main__":
    main()
