"""Compare one Louvain run with multiple initializations on the same saved graph."""

import argparse
import json
import os
import sys
import time
from pathlib import Path
import networkx as nx
import numpy as np

sys.dont_write_bytecode = True
from .io import read_edges, sha256, data_files, timing_environment, same_host, completed_run


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--run", required=True, type=Path, help="Completed main experiment directory"
    )
    p.add_argument("--out", required=True, type=Path, help="New output directory")
    p.add_argument("--restarts", type=int, default=3)
    p.add_argument("--data", required=True, type=Path)
    a = p.parse_args()
    if a.restarts < 2:
        p.error("restarts must be at least two")
    meta, rows = completed_run(a.run, ("graph_nodes.npy", "results.jsonl"))
    if nx.__version__ != meta["networkx"]:
        raise ValueError("networkx version differs from the main run")
    if not same_host(meta.get("timing_environment", {}), timing_environment()):
        raise ValueError("host differs from the main run")
    threads = {
        k: os.environ.get(k)
        for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
    }
    if threads != meta["thread_environment"]:
        raise ValueError("thread settings differ from the main run")
    baseline = [
        r
        for r in rows
        if r.get("task") == "graph"
        and r.get("resolution") == 1.0
        and r.get("status") == "ok"
    ]
    if len(baseline) != 1 or any(r.get("status") != "ok" for r in rows):
        raise ValueError("main run is incomplete or failed")
    edge_path = data_files(a.data, meta, ("edges",))["edges"]
    nodes = np.load(a.run / "graph_nodes.npy", allow_pickle=False)
    node_count = meta["data"]["features_shape"][0]
    if (
        nodes.ndim != 1
        or not np.issubdtype(nodes.dtype, np.integer)
        or len(nodes) != meta["arguments"]["graph_nodes"]
        or len(np.unique(nodes)) != len(nodes)
        or np.any(nodes < 0)
        or np.any(nodes >= node_count)
    ):
        raise ValueError("invalid saved node set")
    edges = read_edges(edge_path, node_count)
    selected = np.zeros(node_count, dtype=bool)
    selected[nodes] = True
    induced = edges[selected[edges[:, 0]] & selected[edges[:, 1]]]
    g = nx.Graph()
    g.add_nodes_from(map(int, nodes))
    g.add_edges_from(induced.tolist())
    g.remove_edges_from(list(nx.selfloop_edges(g)))
    if (
        g.number_of_nodes() != baseline[0]["graph"]["nodes"]
        or g.number_of_edges() != baseline[0]["graph"]["edges"]
    ):
        raise ValueError("graph differs from baseline")
    a.out.mkdir(parents=True, exist_ok=False)
    trials = []
    seed = meta["arguments"]["seed"]
    for i in range(a.restarts):
        tic = time.perf_counter()
        communities = nx.community.louvain_communities(g, seed=seed + i, resolution=1.0)
        fit_time = time.perf_counter() - tic
        tic = time.perf_counter()
        score = float(nx.community.modularity(g, communities, resolution=1.0))
        score_time = time.perf_counter() - tic
        trials.append(
            {
                "seed": seed + i,
                "modularity_at_1": score,
                "communities": len(communities),
                "fit_seconds": fit_time,
                "scoring_seconds": score_time,
            }
        )
        with (a.out / "trials.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(trials[-1]) + "\n")
            f.flush()
        print(json.dumps(trials[-1]), flush=True)
    if not np.isclose(
        trials[0]["modularity_at_1"], baseline[0]["modularity_at_1"], rtol=0, atol=1e-12
    ):
        raise ValueError("recreated baseline modularity differs from main run")
    best = max(trials, key=lambda t: t["modularity_at_1"])
    result = {
        "main_run": a.run.name,
        "timing_environment": timing_environment(),
        "thread_environment": threads,
        "main_metadata_sha256": sha256(a.run / "metadata.json"),
        "graph_nodes_sha256": sha256(a.run / "graph_nodes.npy"),
        "edge_sha256": meta["files"]["edges"]["sha256"],
        "baseline": trials[0],
        "selected": best,
        "restarts": len(trials),
        "total_fit_seconds": sum(t["fit_seconds"] for t in trials),
        "total_scoring_seconds": sum(t["scoring_seconds"] for t in trials),
        "delta_modularity_at_1": best["modularity_at_1"] - trials[0]["modularity_at_1"],
        "note": "Selection maximizes in-sample modularity, not external community quality; graph construction excluded from timings.",
    }
    (a.out / "selection.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
