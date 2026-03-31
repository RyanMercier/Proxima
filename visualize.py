#!/usr/bin/env python3
"""
Publication figures for Vectree.

Fig 1: Distance-preserving vs distance-destroying
Fig 2: Scale comparison (Tree + Flat + HotStuff + PBFT) to 10K
Fig 3: Byzantine sweep (Tree + Flat + HotStuff)
Fig 4: Fast path probability heatmap
Fig 5: Tree level breakdown (where messages go)
"""

import hashlib, math, os, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import blockchain as _bc
_bc.USE_REAL_BLS = False

from blockchain import (
    N_DIMS, compute_vector, calibrate_threshold,
    make_validators, make_partial_obs, vector_consensus, tree_consensus,
    Blockchain, BLSKeyPair,
)
from hotstuff import hotstuff_consensus, pbft_consensus

OUT = "figures"
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.size": 11, "axes.titlesize": 13,
    "axes.labelsize": 12, "figure.dpi": 200, "savefig.dpi": 200,
    "axes.grid": True, "grid.alpha": 0.3,
})

BLUE = "#2E75B6"
GREEN = "#27AE60"
ORANGE = "#E67E22"
RED = "#C0392B"
GRAY = "#7F8C8D"


def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


def _build_chain(n_hon, n_byz, strategy="drop_half", n_txs=20, miss_prob=0.37):
    """Helper: build chain, submit txs, propose block, return what consensus needs."""
    BLSKeyPair._counter = 1
    sv, sh, sb = make_validators(n_hon, n_byz, strategy)
    sc = Blockchain(sv)
    for i in range(5):
        sc.register_account(f"S-{i}", 100000)
    for i in range(n_txs):
        tx = sc.make_tx(f"S-{i%5}", f"S-{(i+1)%5}", 1.0)
        if tx:
            sc.submit_tx(tx)
    blk = sc.propose_block(sh[0])
    tau = calibrate_threshold(blk.tx_data_strings)
    po = make_partial_obs(sv, len(blk.tx_data_strings), miss_prob=miss_prob)
    return sv, blk, tau, po


# =====================================================================
# Figure 1: Distance-preserving vs distance-destroying
# =====================================================================

def figure1():
    print("Figure 1: Distance comparison...")
    np.random.seed(42)
    txs = [hashlib.sha256(f"tx-{i}".encode()).hexdigest() for i in range(20)]
    ref_hash = hashlib.sha256("".join(txs).encode()).hexdigest()
    ref_vec = compute_vector(txs)

    sha_d, vec_d = [], []
    for miss in range(6):
        sd, vd = [], []
        for _ in range(80):
            if miss == 0:
                sub = list(txs)
            else:
                drop = set(np.random.choice(20, miss, replace=False))
                sub = [tx for i, tx in enumerate(txs) if i not in drop]
            sd.append(sum(a != b for a, b in zip(ref_hash,
                      hashlib.sha256("".join(sub).encode()).hexdigest())))
            vd.append(np.linalg.norm(compute_vector(sub) - ref_vec))
        sha_d.append(sd); vec_d.append(vd)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    for ax, data, color, title in [
        (ax1, sha_d, RED, "SHA-256 (Distance-Destroying)"),
        (ax2, vec_d, BLUE, "Transaction Vectors (Distance-Preserving)")]:
        bp = ax.boxplot(data, tick_labels=[str(m) for m in range(6)],
                        patch_artist=True, widths=0.6)
        for b in bp["boxes"]: b.set_facecolor(color); b.set_alpha(0.7)
        ax.set_xlabel("Transactions Missing")
        ax.set_title(title)
    ax1.set_ylabel("Hamming Distance (hex chars)")
    ax2.set_ylabel("Euclidean Distance (8D)")
    fig.suptitle("Why Distance Matters for Consensus", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "fig1_distance_comparison.png")


# =====================================================================
# Figure 2: Scale comparison to 10K (Tree + Flat + HotStuff + PBFT)
# =====================================================================

def figure2():
    print("Figure 2: Scale comparison to 10K...")
    np.random.seed(42)
    sizes = [50, 100, 200, 500, 1000, 2000, 5000, 10000]
    flat_m, tree_m, hs_m, pbft_m = [], [], [], []

    for N in sizes:
        n_h, n_b = int(N * 0.7), N - int(N * 0.7)

        sv, blk, tau, po = _build_chain(n_h, n_b)
        flat_m.append(vector_consensus(sv, blk, tau, po)["msgs"])

        sv2, blk2, tau2, po2 = _build_chain(n_h, n_b)
        tree_m.append(tree_consensus(sv2, blk2, tau2, po2, 10)["msgs"])

        hs_m.append(hotstuff_consensus(N, n_b, 20)["msgs"])
        pbft_m.append(pbft_consensus(N, n_b, 20)["msgs"])

        print(f"    N={N:>6}: tree={tree_m[-1]:>6,}  flat={flat_m[-1]:>6,}  "
              f"hs={hs_m[-1]:>7,}  pbft={pbft_m[-1]:>11,}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Left: linear, Vectree Tree + Flat + HotStuff
    ax = axes[0]
    ax.plot(sizes, tree_m, "D-", color=GREEN, lw=2.5, ms=6, label="Vectree Tree")
    ax.plot(sizes, flat_m, "o-", color=BLUE, lw=2.5, ms=6, label="Vectree Flat")
    ax.plot(sizes, hs_m, "s--", color=ORANGE, lw=2.5, ms=6, label="HotStuff")
    ax.set_xlabel("Validators (N)"); ax.set_ylabel("Messages per Block")
    ax.set_title("Vectree vs HotStuff (linear)", fontweight="bold")
    ax.legend(fontsize=10)
    r_hs = hs_m[-1] / tree_m[-1]
    ax.annotate(f"{r_hs:.1f}x", xy=(sizes[-1], (tree_m[-1] + hs_m[-1]) / 2),
                fontsize=12, fontweight="bold", color=RED, ha="center")

    # Right: log, all four
    ax = axes[1]
    ax.loglog(sizes, tree_m, "D-", color=GREEN, lw=2.5, ms=6, label="Vectree Tree")
    ax.loglog(sizes, flat_m, "o-", color=BLUE, lw=2.5, ms=6, label="Vectree Flat")
    ax.loglog(sizes, hs_m, "s--", color=ORANGE, lw=2.5, ms=6, label="HotStuff")
    ax.loglog(sizes, pbft_m, "^:", color=RED, lw=2.5, ms=6, label="PBFT")
    ax.set_xlabel("Validators (N)"); ax.set_ylabel("Messages per Block")
    ax.set_title("All protocols (log)", fontweight="bold")
    ax.legend(fontsize=10)

    fig.suptitle("Message Complexity", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "fig2_scale_comparison.png")


# =====================================================================
# Figure 3: Byzantine sweep (Tree + Flat + HotStuff)
# =====================================================================

def figure3():
    print("Figure 3: Byzantine tolerance sweep...")
    np.random.seed(42)

    N = 100
    byz_pcts = [0, 5, 10, 15, 20, 25, 30, 33, 35, 40, 45]
    trials = 10

    flat_s, flat_m, flat_bw = [], [], []
    tree_s, tree_m, tree_bw = [], [], []
    hs_s, hs_m, hs_bw = [], [], []

    for bp in byz_pcts:
        n_b = int(N * bp / 100)
        n_h = N - n_b

        fs, fm, fb = 0, [], []
        ts, tm, tb = 0, [], []
        hs_ok, hm, hb = 0, [], []

        for t in range(trials):
            sv, blk, tau, po = _build_chain(n_h, n_b)

            vr = vector_consensus(sv, blk, tau, po)
            if vr["finalized"]: fs += 1
            fm.append(vr["msgs"]); fb.append(vr["msg_bytes"] / 1024)

            tr = tree_consensus(sv, blk, tau, po, 10)
            if tr["finalized"]: ts += 1
            tm.append(tr["msgs"]); tb.append(tr["msg_bytes"] / 1024)

            hr = hotstuff_consensus(N, n_b, 20, 0.37)
            if hr["finalized"]: hs_ok += 1
            hm.append(hr["msgs"]); hb.append(hr["msg_bytes"] / 1024)

        flat_s.append(fs/trials*100); flat_m.append(np.mean(fm)); flat_bw.append(np.mean(fb))
        tree_s.append(ts/trials*100); tree_m.append(np.mean(tm)); tree_bw.append(np.mean(tb))
        hs_s.append(hs_ok/trials*100); hs_m.append(np.mean(hm)); hs_bw.append(np.mean(hb))

        print(f"    {bp:>2}% byz: flat {fs}/{trials}  tree {ts}/{trials}  hs {hs_ok}/{trials}")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Success rate
    ax = axes[0]
    ax.plot(byz_pcts, tree_s, "D-", color=GREEN, lw=2.5, ms=7, label="Vectree Tree")
    ax.plot(byz_pcts, flat_s, "o-", color=BLUE, lw=2.5, ms=7, label="Vectree Flat")
    ax.plot(byz_pcts, hs_s, "s--", color=ORANGE, lw=2.5, ms=7, label="HotStuff")
    ax.axvline(33, color=RED, ls="--", lw=1.5, alpha=0.7, label="BFT limit (33%)")
    ax.fill_between([33, 45], 0, 100, alpha=0.05, color=RED)
    ax.set_xlabel("Byzantine Validators (%)"); ax.set_ylabel("Consensus Success (%)")
    ax.set_title("Consensus Success Rate", fontweight="bold")
    ax.legend(fontsize=8); ax.set_ylim(-5, 105)

    # Messages
    ax = axes[1]
    ax.plot(byz_pcts, tree_m, "D-", color=GREEN, lw=2.5, ms=7, label="Vectree Tree")
    ax.plot(byz_pcts, flat_m, "o-", color=BLUE, lw=2.5, ms=7, label="Vectree Flat")
    ax.plot(byz_pcts, hs_m, "s--", color=ORANGE, lw=2.5, ms=7, label="HotStuff")
    ax.axvline(33, color=RED, ls="--", lw=1.5, alpha=0.7)
    ax.set_xlabel("Byzantine Validators (%)"); ax.set_ylabel("Messages per Block")
    ax.set_title("Message Cost", fontweight="bold"); ax.legend(fontsize=8)

    # Bandwidth
    ax = axes[2]
    ax.plot(byz_pcts, tree_bw, "D-", color=GREEN, lw=2.5, ms=7, label="Vectree Tree")
    ax.plot(byz_pcts, flat_bw, "o-", color=BLUE, lw=2.5, ms=7, label="Vectree Flat")
    ax.plot(byz_pcts, hs_bw, "s--", color=ORANGE, lw=2.5, ms=7, label="HotStuff")
    ax.axvline(33, color=RED, ls="--", lw=1.5, alpha=0.7)
    ax.set_xlabel("Byzantine Validators (%)"); ax.set_ylabel("Bandwidth (KB)")
    ax.set_title("Bandwidth Cost", fontweight="bold"); ax.legend(fontsize=8)

    fig.suptitle(f"Byzantine Tolerance: N={N}, 37% partial observation, {trials} trials/point",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "fig3_byzantine_sweep.png")


# =====================================================================
# Figure 4: Fast path probability heatmap
# =====================================================================

def figure4():
    print("Figure 4: Fast path probability...")
    miss_rates = np.arange(0.0, 0.42, 0.02)
    n_vals = np.array([5, 7, 10, 15, 20, 30, 50, 70, 100])

    grid = np.zeros((len(n_vals), len(miss_rates)))
    for i, nh in enumerate(n_vals):
        for j, mr in enumerate(miss_rates):
            grid[i, j] = (1 - mr) ** nh * 100

    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(grid, aspect="auto", origin="lower", cmap="RdYlGn", vmin=0, vmax=100,
                   extent=[miss_rates[0]*100, miss_rates[-1]*100, -0.5, len(n_vals)-0.5])
    ax.set_yticks(range(len(n_vals)))
    ax.set_yticklabels([str(n) for n in n_vals])
    ax.set_xlabel("Per-Validator Probability of Missing 1-2 Txs (%)")
    ax.set_ylabel("Number of Honest Validators")
    fig.colorbar(im, ax=ax, label="Fast Path Probability (%)")

    for i, nh in enumerate(n_vals):
        for j, mr in enumerate(miss_rates):
            if j % 3 == 0:
                val = grid[i, j]
                ax.text(mr*100, i, f"{val:.0f}", ha="center", va="center",
                        fontsize=7, color="white" if val < 40 else "black", fontweight="bold")

    # 50% contour
    cx, cy = [], []
    for i, nh in enumerate(n_vals):
        for j in range(len(miss_rates)-1):
            if grid[i,j] >= 50 and grid[i,j+1] < 50:
                frac = (50 - grid[i,j+1]) / (grid[i,j] - grid[i,j+1])
                cx.append((miss_rates[j+1] - frac*(miss_rates[j+1]-miss_rates[j]))*100)
                cy.append(i); break
    if cx:
        ax.plot(cx, cy, "k--", lw=2, alpha=0.7, label="50% fast path")
        ax.legend(fontsize=10, loc="upper right")

    ax.set_title("Optimistic Fast Path: P(fast path) = (1 - miss_rate) ^ N_honest",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    save(fig, "fig4_fast_path.png")


# =====================================================================
# Figure 5: Tree level breakdown
# =====================================================================

def figure5():
    print("Figure 5: Tree level breakdown...")
    np.random.seed(42)

    test_sizes = [1000, 10000, 100000]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax, N in zip(axes, test_sizes):
        n_h, n_b = int(N*0.7), N - int(N*0.7)
        sv, blk, tau, po = _build_chain(n_h, n_b)
        tr = tree_consensus(sv, blk, tau, po, 10)

        # Extract message counts per level from breakdown
        breakdown = tr["msg_breakdown"]
        levels = tr["n_levels"]

        # Group messages: leaf level, each internal level, phase 2
        level_msgs = []
        labels = []

        # Level 0
        l0 = breakdown.get("L0_vector", 0) + breakdown.get("L0_sync", 0) + breakdown.get("L0_summary", 0)
        n_leaves = tr["n_leaves"]
        level_msgs.append(l0)
        labels.append(f"L0\n({n_leaves} leaves)")

        # Internal levels
        for lv in range(1, levels):
            s = breakdown.get(f"L{lv}_summary", 0)
            n_nodes = math.ceil(n_leaves / (10 ** lv))
            level_msgs.append(s)
            labels.append(f"L{lv}\n({max(n_nodes,1)} nodes)")

        # Phase 2: all P2_ messages
        p2 = sum(v for k, v in breakdown.items() if k.startswith("P2_"))
        level_msgs.append(p2)
        labels.append("Phase 2\n(BLS)")

        colors = [BLUE] * levels + [ORANGE]
        bars = ax.bar(range(len(level_msgs)), level_msgs, color=colors, alpha=0.8)

        for bar, val in zip(bars, level_msgs):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(level_msgs)*0.02,
                        f"{val:,}", ha="center", fontsize=9, fontweight="bold")

        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel("Messages")
        ax.set_title(f"N = {N:,}", fontweight="bold")
        ax.text(0.97, 0.95, f"Total: {tr['msgs']:,}", transform=ax.transAxes,
                ha="right", va="top", fontsize=10, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=GRAY))

    fig.suptitle("Vectree Level Breakdown: Most Messages Stay at the Leaves",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "fig5_tree_breakdown.png")


# =====================================================================

def main():
    t0 = time.time()
    print("Generating figures...\n")
    figure1()
    figure2()
    figure3()
    figure4()
    figure5()
    print(f"\nDone in {time.time()-t0:.1f}s. Figures in {OUT}/")


if __name__ == "__main__":
    main()
