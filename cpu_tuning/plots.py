"""Render audited CSV summaries with Matplotlib (optional plotting dependency)."""

import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

NAMES = ["raw", "raw_ros", "hop1", "hop2", "hop2_ros"]
LABELS = ["Raw MLP", "Raw + ROS", "1-hop MLP", "2-hop MLP", "2-hop + ROS"]
COLORS = ["#66768a", "#94a4b6", "#539aa7", "#225b90", "#db8b36"]


def rows(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def save(fig, out, name):
    fig.savefig(out / (name + ".png"), dpi=180, bbox_inches="tight")
    fig.savefig(out / (name + ".svg"), bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--summary", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    means = rows(a.summary / "mean_sd.csv")
    values = rows(a.summary / "per_seed.csv")
    curves = rows(a.summary / "learning_curves.csv")
    audit = json.loads((a.summary / "audit.json").read_text(encoding="utf-8"))
    protocol = audit["shared_protocol"]
    seed_count = len(audit["runs"])
    class_count = len({r["label"] for r in rows(a.summary / "per_class.csv")})
    rare_count = min(10, class_count)
    a.out.mkdir(parents=True, exist_ok=False)
    lookup = {(r["name"], r["metric"]): r for r in means}
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.fonttype": "none",
        }
    )
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.8))
    for ax, metric, title in zip(
        axs,
        ["accuracy", "macro_f1", "rare10_macro_f1"],
        [
            "Accuracy",
            f"Macro F1 ({class_count} classes)",
            f"Macro F1 ({rare_count} rarest training classes)",
        ],
    ):
        for i, name in enumerate(NAMES):
            r = lookup[name, metric]
            mean, sd = float(r["mean"]), float(r["sd"])
            ax.bar(i, mean, color=COLORS[i], width=0.65, alpha=0.85)
            ax.errorbar(i, mean, yerr=sd, fmt="none", ecolor="#222", capsize=4)
            points = [
                float(r["value"])
                for r in values
                if r["name"] == name and r["metric"] == metric
            ]
            ax.scatter(
                i + np.linspace(-0.13, 0.13, len(points)),
                points,
                s=22,
                color="#15283c",
                zorder=4,
            )
            ax.text(i, mean + sd + 0.027, f"{mean:.3f}", ha="center", fontsize=9)
        ax.set_xticks(range(5), LABELS, rotation=30, ha="right")
        ax.set_ylim(0, 1)
        ax.set_title(title)
        ax.yaxis.grid(True, alpha=0.18)
        ax.set_axisbelow(True)
    fig.suptitle(
        f"MLP ablation: fixed splits, {seed_count} seeds, mean ± sample SD", fontsize=14
    )
    fig.text(
        0.5,
        0.01,
        f"Dots: individual seeds. Graph-feature arms use the full unlabeled graph (transductive). ROS: minority floor = {protocol['ros_floor']}.",
        ha="center",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0.04, 1, 0.92])
    save(fig, a.out, "ablation")
    fig, ax = plt.subplots(figsize=(9, 4.8))
    bottom = np.zeros(5)
    for metric, label, color in [
        ("prep_seconds", "Resampling + scaling", "#a9bacb"),
        ("fit_seconds", "Final fit", "#225b90"),
        ("predict_seconds", "Predict", "#db8b36"),
    ]:
        vals = np.array([float(lookup[n, metric]["mean"]) for n in NAMES])
        ax.bar(range(5), vals, bottom=bottom, label=label, color=color)
        bottom += vals
    ax.set_xticks(range(5), LABELS)
    ax.set_ylabel(f"Seconds ({seed_count}-seed mean)")
    ax.set_title("Final training and prediction cost")
    ax.legend()
    feature_time = audit["runs"][0]["cache"]["compute_seconds"]
    fig.text(
        0.5,
        0.02,
        f"Excludes validation search and one-time graph feature computation ({feature_time:.2f} s); cache I/O is recorded separately.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    save(fig, a.out, "runtime")
    seeds = sorted({int(r["seed"]) for r in curves})
    fig, axs = plt.subplots(1, len(seeds), figsize=(14, 4.2), sharey=True)
    for ax, seed in zip(np.atleast_1d(axs), seeds):
        for name, label, color in zip(NAMES, LABELS, COLORS):
            data = [r for r in curves if int(r["seed"]) == seed and r["name"] == name]
            x = [int(r["epoch"]) for r in data]
            y = [float(r["macro_f1"]) for r in data]
            ax.plot(x, y, label=label, color=color, lw=1.5)
            best = max(data, key=lambda r: (float(r["macro_f1"]), float(r["accuracy"])))
            ax.scatter(int(best["epoch"]), float(best["macro_f1"]), color=color, s=25)
        ax.set_title(f"Seed {seed}")
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.18)
    np.atleast_1d(axs)[0].set_ylabel("External validation macro F1")
    np.atleast_1d(axs)[-1].legend(fontsize=8)
    fig.suptitle(
        "Learning curves: external validation in original distribution, best epoch marked"
    )
    fig.tight_layout()
    save(fig, a.out, "learning_curves")
    print("Saved PNG and SVG figures to", a.out)


if __name__ == "__main__":
    main()
