"""Shared data loading, hashing and append-only result writing."""

import gzip
import hashlib
import json
import numpy as np


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


def record(out, result):
    with (out / "results.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
        f.flush()
    print(json.dumps(result, ensure_ascii=False), flush=True)
