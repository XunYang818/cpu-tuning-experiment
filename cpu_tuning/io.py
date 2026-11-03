"""Shared data loading, hashing and append-only result writing."""

import gzip
import hashlib
import json
import os
import platform
from threadpoolctl import threadpool_info
import numpy as np
from pathlib import Path


def verify_artifacts(path, metadata, names):
    """Check the saved artifacts required by the selected downstream task."""
    for name in names:
        expected = metadata.get("artifact_sha256", {}).get(name)
        artifact = path / name
        if not expected or not artifact.is_file() or sha256(artifact) != expected:
            raise ValueError(f"missing or changed baseline artifact: {name}")


def completed_run(path, artifacts=()):
    """Load a finished baseline and verify the saved inputs reused downstream."""
    meta = json.loads((path / "metadata.json").read_text(encoding="utf-8-sig"))
    if meta.get("status") != "complete":
        raise ValueError("baseline is unfinished or predates completion checks; rerun it")
    verify_artifacts(path, meta, artifacts)
    rows = [
        json.loads(line)
        for line in (path / "results.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    if not rows or any(r.get("status") != "ok" for r in rows):
        raise ValueError("baseline has missing or failed results")
    return meta, rows


def data_files(data, metadata, keys):
    """Resolve data from the supplied directory and verify its recorded content."""
    raw = data / "raw" if (data / "raw").is_dir() else data
    result = {}
    for key in keys:
        entry = metadata["files"][key]
        name = entry.get("name") or Path(entry["path"]).name
        path = raw / name
        if not path.is_file() or sha256(path) != entry["sha256"]:
            raise ValueError(f"missing or changed {key}: {path}")
        result[key] = path
    return result


def dump(path, obj):
    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_array(path, dtype):
    with gzip.open(path, "rt", encoding="utf-8", newline="") as f:
        return np.loadtxt(f, delimiter=",", dtype=dtype, ndmin=2)


def read_edges(path, node_count):
    """Validate before integer conversion; fractional IDs must not be truncated."""
    edges = read_array(path, np.float64)
    if (
        edges.shape[1] != 2
        or edges.size == 0
        or not np.isfinite(edges).all()
        or not np.equal(edges, np.floor(edges)).all()
        or edges.min() < 0
        or edges.max() >= node_count
    ):
        raise ValueError("edge table must have two valid integer node-ID columns")
    return edges.astype(np.int64)


def record(out, result):
    with (out / "results.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
        f.flush()
    print(json.dumps(result, ensure_ascii=False), flush=True)


def timing_environment():
    """Record comparable hardware and numerical runtimes without local paths."""
    keys = (
        "user_api",
        "internal_api",
        "prefix",
        "num_threads",
        "version",
        "threading_layer",
        "architecture",
    )
    try:
        pools = [{k: pool.get(k) for k in keys} for pool in threadpool_info()]
    except OSError as exc:
        raise RuntimeError(
            "Cannot inspect numerical libraries. Activate the Conda environment "
            "before running Python, or use conda run -n cpu-tuning-experiment."
        ) from exc
    pools.sort(key=lambda pool: json.dumps(pool, sort_keys=True))
    return {
        "host": platform.node(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "logical_cpus": os.cpu_count(),
        "threadpools": pools,
    }


def same_host(left, right):
    """Compare host identity and hardware independently of loaded libraries."""
    keys = ("host", "platform", "machine", "processor", "logical_cpus")
    return bool(left.get("host")) and all(
        k in left and k in right and left[k] == right[k] for k in keys
    )
