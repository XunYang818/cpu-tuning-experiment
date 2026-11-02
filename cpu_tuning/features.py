"""Label-free graph aggregation and content-checked feature caching."""

import json
import time
import numpy as np
from scipy import sparse
from .io import read_array, sha256, dump


def aggregate_features(X, edges):
    """Only node attributes and edge IDs are accepted; no labels or years."""
    if edges.ndim != 2 or edges.shape[1] != 2 or edges.size == 0:
        raise ValueError("invalid edge shape")
    if not np.isfinite(edges).all() or not np.equal(edges, np.floor(edges)).all():
        raise ValueError("edges must be integer node IDs")
    edges = edges.astype(np.int64)
    if edges.min() < 0 or edges.max() >= len(X):
        raise ValueError("edge ID out of range")
    rows = np.concatenate([edges[:, 0], edges[:, 1]])
    cols = np.concatenate([edges[:, 1], edges[:, 0]])
    A = sparse.csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(len(X), len(X))
    )
    A.sum_duplicates()
    A.data.fill(1)
    A.setdiag(1)
    A.sort_indices()
    degree = np.asarray(A.sum(axis=1)).ravel()
    P = sparse.diags(1 / degree, format="csr") @ A
    if not np.allclose(np.asarray(P.sum(axis=1)).ravel(), 1, atol=1e-6):
        raise ValueError("row normalization failed")
    H1 = np.asarray(P @ X, dtype=np.float32)
    H2 = np.asarray(P @ H1, dtype=np.float32)
    return (
        H1,
        H2,
        {
            "nodes": len(X),
            "adjacency_nnz_with_self": int(A.nnz),
            "undirected_edges_without_self": int((A.nnz - len(X)) // 2),
            "isolates_before_self_loops": int(np.count_nonzero(degree == 1)),
        },
    )


def graph_cache(cache, X, edge_path, feature_hash):
    signature = {
        "features_sha256": feature_hash,
        "edges_sha256": sha256(edge_path),
        "protocol": "undirected-binary-self-loop-row-mean-v1",
        "shape": list(X.shape),
    }
    if cache.exists():
        meta = json.loads((cache / "metadata.json").read_text(encoding="utf-8"))
        if meta["signature"] != signature:
            raise ValueError("graph cache signature mismatch")
        arrays = []
        for name in ("hop1", "hop2"):
            path = cache / (name + ".npy")
            if sha256(path) != meta["array_sha256"][name]:
                raise ValueError("corrupted graph cache")
            a = np.load(path, mmap_mode="r", allow_pickle=False)
            if a.shape != X.shape or a.dtype != np.float32:
                raise ValueError("cache shape/dtype mismatch")
            arrays.append(a)
        return arrays, meta
    cache.mkdir(parents=True, exist_ok=False)
    edges = read_array(edge_path, np.float64)
    tic = time.perf_counter()
    H1, H2, info = aggregate_features(X, edges)
    compute_time = time.perf_counter() - tic
    tic = time.perf_counter()
    np.save(cache / "hop1.npy", H1)
    np.save(cache / "hop2.npy", H2)
    save_time = time.perf_counter() - tic
    meta = {
        "signature": signature,
        "graph": info,
        "compute_seconds": compute_time,
        "save_seconds": save_time,
        "array_sha256": {n: sha256(cache / (n + ".npy")) for n in ("hop1", "hop2")},
        "setting": "transductive: all node features and edges, no labels/years",
    }
    dump(cache / "metadata.json", meta)
    return [H1, H2], meta


def make_input(X, hops, ids, order):
    blocks = [X[ids]] + [hops[i][ids] for i in range(order)]
    return np.hstack(blocks).astype(np.float32, copy=False)
