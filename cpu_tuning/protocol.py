"""Declared MLP arms, training-only resampling, split guards and metrics."""

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


ARMS = [
    ("raw", 0, False),
    ("raw_ros", 0, True),
    ("hop1", 1, False),
    ("hop2", 2, False),
    ("hop2_ros", 2, True),
]


MLP_PARAMS = dict(
    hidden_layer_sizes=(128,),
    learning_rate_init=3e-4,
    alpha=1e-3,
    early_stopping=False,
    solver="adam",
    batch_size=200,
)


def resample_ids(ids, y, floor, seed):
    """Add copies from fitting IDs only; original samples are always retained."""
    rng = np.random.RandomState(seed)
    pieces = [np.asarray(ids)]
    for label in np.unique(y[ids]):
        members = np.asarray(ids)[y[ids] == label]
        if len(members) < floor:
            pieces.append(rng.choice(members, size=floor - len(members), replace=True))
    return np.concatenate(pieces)


def check_splits(parts, sampled, node_count):
    if any(
        v.ndim != 1
        or len(v) == 0
        or not np.issubdtype(v.dtype, np.integer)
        or len(np.unique(v)) != len(v)
        or v.min() < 0
        or v.max() >= node_count
        for v in parts
    ):
        raise ValueError("invalid split IDs")
    combined = np.concatenate(parts)
    if (
        len(combined) != len(sampled)
        or len(np.unique(combined)) != len(combined)
        or set(combined) != set(sampled)
    ):
        raise ValueError("overlapping/inconsistent splits")


def scores(y_true, pred, labels, rare):
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "macro_f1": float(
            f1_score(y_true, pred, labels=labels, average="macro", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(y_true, pred, labels=labels, average="weighted", zero_division=0)
        ),
        "rare10_macro_f1": float(
            f1_score(y_true, pred, labels=rare, average="macro", zero_division=0)
        ),
    }
