#!/usr/bin/env python3
"""
visualize_v2.py -- Validation figures for the v2 primitive (plan items E1, E2).

E1: the estimator collapse. dist2/n concentrates on d for missing txs AND
    substitutions alike (v1's nonnegative coordinates read substitutions
    at a fraction of their true size; zero-mean coordinates read exactly
    the symmetric difference).
E2: the closed-form threshold. Exact small-d distributions of dist2
    against tau^2 = 3n, with log-scale tails.

Run: python visualize_v2.py     (writes figures/figE1_*.png, figE2_*.png)
"""

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import dpmh

OUT = "figures"
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.size": 10, "axes.titlesize": 11,
    "figure.dpi": 200, "savefig.dpi": 200, "axes.grid": True,
    "grid.alpha": 0.3,
})
BLUE, GREEN, ORANGE, RED = "#2E75B6", "#27AE60", "#E67E22", "#C0392B"

N = dpmh.N_ROUND
rng = np.random.default_rng(42)


def sample_dist2(d_missing=0, d_subst=0, trials=400):
    """dist2 samples for a validator missing d_missing txs and
    substituting d_subst txs (each substitution = symdiff 2)."""
    out = []
    for t in range(trials):
        txs = [f"tx-{t}-{i}" for i in range(40)]
        full = dpmh.digest(txs, height=t)
        other = list(txs)
        if d_missing:
            drop = rng.choice(len(other), size=d_missing, replace=False)
            other = [x for i, x in enumerate(other) if i not in set(drop.tolist())]
        for s in range(d_subst):
            other[s] = f"SUB-{t}-{s}"
        out.append(dpmh.dist2(full, dpmh.digest(other, height=t)))
    return np.array(out)


def figure_e1():
    print("E1: estimator collapse (missing vs substituted)...")
    cases = [
        ("miss 1", 1, 0, 1), ("miss 2", 2, 0, 2), ("miss 4", 4, 0, 4),
        ("subst 1", 0, 1, 2), ("subst 2", 0, 2, 4), ("subst 4", 0, 4, 8),
    ]
    data, expected, labels = [], [], []
    for label, dm, ds, d_true in cases:
        data.append(sample_dist2(dm, ds) / N)
        expected.append(d_true)
        labels.append(f"{label}\n(d={d_true})")

    fig, ax = plt.subplots(figsize=(7, 3))
    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, widths=0.55)
    for i, b in enumerate(bp["boxes"]):
        b.set_facecolor(BLUE if i < 3 else ORANGE)
        b.set_alpha(0.7)
    xs = np.arange(1, len(cases) + 1)
    ax.plot(xs, expected, "k--", lw=1.4, label="analytic: dist$^2$/n = d")
    ax.scatter(xs, expected, color="k", zorder=5, s=14)
    ax.set_ylabel("dist$^2$ / n")
    ax.set_title(f"Zero-mean digests estimate symmetric difference "
                 f"regardless of direction (n={N})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{OUT}/figE1_estimator.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {OUT}/figE1_estimator.png")


def figure_e2():
    print("E2: closed-form threshold tau^2 = 3n...")
    fig, ax = plt.subplots(figsize=(7, 3))
    colors = [GREEN, BLUE, ORANGE, RED]
    for d, c in zip((1, 2, 3, 4), colors):
        samples = sample_dist2(d_missing=d, trials=600) / N
        ax.hist(samples, bins=40, density=True, alpha=0.55, color=c,
                label=f"d = {d}")
    ax.axvline(3.0, color="k", ls="--", lw=1.6,
               label=r"$\tau^2 = 3n$ (closed form)")
    ax.set_xlabel("dist$^2$ / n")
    ax.set_ylabel("density")
    ax.set_yscale("log")
    ax.set_title(f"Exact small-d distributions vs the closed-form threshold "
                 f"(n={N})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{OUT}/figE2_threshold.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {OUT}/figE2_threshold.png")


if __name__ == "__main__":
    figure_e1()
    figure_e2()
