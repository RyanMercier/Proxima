#!/usr/bin/env python3
"""
Publication figures for Proxima.

Fig 1: Distance-preserving vs distance-destroying
Fig 2: Scale comparison (Tree + Flat + HotStuff + PBFT) to 10K
Fig 3: Byzantine sweep (Tree + Flat + HotStuff)
Fig 4: Fast path probability heatmap
Fig 5: Tree level breakdown (where messages go)
Fig 6: BLS aggregation bottleneck (projected blst numbers)
Fig 7: Latency model (network RTT + BLS processing)
Fig 8: Cross-shard verification overhead (digest vs 2PC vs receipts)
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

    # Left: linear, Proxima Tree + Flat + HotStuff
    ax = axes[0]
    ax.plot(sizes, tree_m, "D-", color=GREEN, lw=2.5, ms=6, label="Proxima Tree")
    ax.plot(sizes, flat_m, "o-", color=BLUE, lw=2.5, ms=6, label="Proxima Flat")
    ax.plot(sizes, hs_m, "s--", color=ORANGE, lw=2.5, ms=6, label="HotStuff")
    ax.set_xlabel("Validators (N)"); ax.set_ylabel("Messages per Block")
    ax.set_title("Proxima vs HotStuff (linear)", fontweight="bold")
    ax.legend(fontsize=10)
    r_hs = hs_m[-1] / tree_m[-1]
    ax.annotate(f"{r_hs:.1f}x", xy=(sizes[-1], (tree_m[-1] + hs_m[-1]) / 2),
                fontsize=12, fontweight="bold", color=RED, ha="center")

    # Right: log, all four
    ax = axes[1]
    ax.loglog(sizes, tree_m, "D-", color=GREEN, lw=2.5, ms=6, label="Proxima Tree")
    ax.loglog(sizes, flat_m, "o-", color=BLUE, lw=2.5, ms=6, label="Proxima Flat")
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
    ax.plot(byz_pcts, tree_s, "D-", color=GREEN, lw=2.5, ms=7, label="Proxima Tree")
    ax.plot(byz_pcts, flat_s, "o-", color=BLUE, lw=2.5, ms=7, label="Proxima Flat")
    ax.plot(byz_pcts, hs_s, "s--", color=ORANGE, lw=2.5, ms=7, label="HotStuff")
    ax.axvline(33, color=RED, ls="--", lw=1.5, alpha=0.7, label="BFT limit (33%)")
    ax.fill_between([33, 45], 0, 100, alpha=0.05, color=RED)
    ax.set_xlabel("Byzantine Validators (%)"); ax.set_ylabel("Consensus Success (%)")
    ax.set_title("Consensus Success Rate", fontweight="bold")
    ax.legend(fontsize=8); ax.set_ylim(-5, 105)

    # Messages
    ax = axes[1]
    ax.plot(byz_pcts, tree_m, "D-", color=GREEN, lw=2.5, ms=7, label="Proxima Tree")
    ax.plot(byz_pcts, flat_m, "o-", color=BLUE, lw=2.5, ms=7, label="Proxima Flat")
    ax.plot(byz_pcts, hs_m, "s--", color=ORANGE, lw=2.5, ms=7, label="HotStuff")
    ax.axvline(33, color=RED, ls="--", lw=1.5, alpha=0.7)
    ax.set_xlabel("Byzantine Validators (%)"); ax.set_ylabel("Messages per Block")
    ax.set_title("Message Cost", fontweight="bold"); ax.legend(fontsize=8)

    # Bandwidth
    ax = axes[2]
    ax.plot(byz_pcts, tree_bw, "D-", color=GREEN, lw=2.5, ms=7, label="Proxima Tree")
    ax.plot(byz_pcts, flat_bw, "o-", color=BLUE, lw=2.5, ms=7, label="Proxima Flat")
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

    fig.suptitle("Proxima Level Breakdown: Most Messages Stay at the Leaves",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "fig5_tree_breakdown.png")


# =====================================================================
# Figure 6: BLS Aggregation Bottleneck
# =====================================================================

def figure6():
    """Processing time at the aggregator: tree vs flat vs HotStuff."""
    print("Figure 6: BLS aggregation bottleneck...")

    # Production BLS constants (blst benchmarks)
    BLST_AGG_ADD_MS = 0.05
    BLST_AGG_VERIFY_MS = 1.5
    DIST_CHECK_MS = 0.002
    BLOOM_CHECK_MS = 0.01

    sizes = [100, 500, 1000, 5000, 10000, 50000, 100000]
    branching = 10
    byz_frac = 0.3
    partial_frac = 0.37

    flat_proc = []
    tree_proc = []
    hs_proc = []
    ratios = []

    for n in sizes:
        n_honest = int(n * (1 - byz_frac))
        n_partial = int(n_honest * partial_frac)

        # Flat: distance check all + bloom partial + BLS aggregate honest
        fp = (n * DIST_CHECK_MS + n_partial * BLOOM_CHECK_MS +
              n_honest * BLST_AGG_ADD_MS + BLST_AGG_VERIFY_MS)
        flat_proc.append(fp)

        # Tree: leaf does branching validators, internal levels do branching aggregates
        honest_per_leaf = int(branching * (1 - byz_frac))
        partial_per_leaf = int(honest_per_leaf * partial_frac)
        n_leaves = math.ceil(n / branching)
        n_levels = 1
        nodes = n_leaves
        while nodes > 1:
            nodes = math.ceil(nodes / branching)
            n_levels += 1
        leaf = (branching * DIST_CHECK_MS + partial_per_leaf * BLOOM_CHECK_MS +
                honest_per_leaf * BLST_AGG_ADD_MS + BLST_AGG_VERIFY_MS)
        internal = branching * BLST_AGG_ADD_MS + BLST_AGG_VERIFY_MS
        tp = leaf + (n_levels - 1) * internal
        tree_proc.append(tp)

        # HotStuff: retransmit + 3 rounds of BLS agg
        hp = n_partial * 0.1 + 3 * (n * BLST_AGG_ADD_MS + BLST_AGG_VERIFY_MS)
        hs_proc.append(hp)

        ratios.append(fp / max(tp, 0.001))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    # Left: processing time (log scale)
    ax1.loglog(sizes, tree_proc, "D-", color=GREEN, lw=2.5, ms=6, label="Proxima Tree")
    ax1.loglog(sizes, flat_proc, "o-", color=BLUE, lw=2.5, ms=6, label="Proxima Flat")
    ax1.loglog(sizes, hs_proc, "s--", color=ORANGE, lw=2, ms=6, label="HotStuff")
    ax1.set_xlabel("Validators (N)")
    ax1.set_ylabel("Processing Time (ms)")
    ax1.set_title("Processing Time at the Aggregator", fontweight="bold")
    ax1.legend(fontsize=10)

    # Annotate endpoints
    ax1.annotate(f"{flat_proc[-1]:.0f}ms", xy=(sizes[-1], flat_proc[-1]),
                 xytext=(-60, 10), textcoords="offset points", fontsize=9,
                 color=BLUE, fontweight="bold")
    ax1.annotate(f"{tree_proc[-1]:.0f}ms", xy=(sizes[-1], tree_proc[-1]),
                 xytext=(-50, -18), textcoords="offset points", fontsize=9,
                 color=GREEN, fontweight="bold")

    # Right: speedup ratio
    ax2.semilogx(sizes, ratios, "D-", color=GREEN, lw=2.5, ms=7)
    ax2.set_xlabel("Validators (N)")
    ax2.set_ylabel("Flat / Tree Processing Time")
    ax2.set_title("Tree Speedup Factor", fontweight="bold")
    ax2.fill_between(sizes, ratios, alpha=0.1, color=GREEN)

    # Annotate key points
    for i, n in enumerate(sizes):
        if n in (1000, 100000):
            ax2.annotate(f"{ratios[i]:.0f}x at N={n:,}",
                         xy=(n, ratios[i]),
                         xytext=(15, -5), textcoords="offset points",
                         fontsize=9, fontweight="bold", color=GREEN)

    fig.suptitle("The Bottleneck: BLS Signature Aggregation",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "fig6_bls_bottleneck.png")


# =====================================================================
# Figure 7: Latency Model
# =====================================================================

def figure7():
    """Total finality latency: network RTT + BLS processing."""
    print("Figure 7: Latency model...")

    BLST_AGG_ADD_MS = 0.05
    BLST_AGG_VERIFY_MS = 1.5
    DIST_CHECK_MS = 0.002
    BLOOM_CHECK_MS = 0.01
    RTT_LOCAL = 1
    RTT_REGIONAL = 80
    RTT_GLOBAL = 200

    sizes = [100, 500, 1000, 5000, 10000, 50000, 100000]
    branching = 10
    byz_frac = 0.3
    partial_frac = 0.37

    lat_flat = []
    lat_tree = []
    lat_hs = []

    # For breakdown at N=100K
    breakdown_n = 100000

    for n in sizes:
        n_honest = int(n * (1 - byz_frac))
        n_partial = int(n_honest * partial_frac)

        # Flat: phase1 (global RTT + processing) + phase2 (global RTT + BLS agg) + finality broadcast
        flat_p1 = RTT_GLOBAL + n * DIST_CHECK_MS + n_partial * BLOOM_CHECK_MS
        flat_p2 = RTT_GLOBAL + n_honest * BLST_AGG_ADD_MS + BLST_AGG_VERIFY_MS
        flat_fin = RTT_GLOBAL
        flat_total = flat_p1 + flat_p2 + flat_fin
        lat_flat.append(flat_total)

        # Tree: leaf (local) + internal levels (regional) + cross-region + phase 2 same structure
        n_leaves = math.ceil(n / branching)
        n_levels = 1
        nodes = n_leaves
        while nodes > 1:
            nodes = math.ceil(nodes / branching)
            n_levels += 1
        honest_per_leaf = int(branching * (1 - byz_frac))
        partial_per_leaf = int(honest_per_leaf * partial_frac)

        leaf_proc = (branching * DIST_CHECK_MS + partial_per_leaf * BLOOM_CHECK_MS +
                     honest_per_leaf * BLST_AGG_ADD_MS + BLST_AGG_VERIFY_MS)
        internal_proc = branching * BLST_AGG_ADD_MS + BLST_AGG_VERIFY_MS
        # Phase 1 up: leaf (local RTT) + internal levels (regional) + root (global)
        tree_p1 = (RTT_LOCAL + leaf_proc +
                   max(0, n_levels - 2) * (RTT_REGIONAL + internal_proc) +
                   RTT_GLOBAL)
        # Phase 2 down then up: similar structure
        tree_p2 = (RTT_GLOBAL +
                   max(0, n_levels - 2) * (RTT_REGIONAL + internal_proc) +
                   RTT_LOCAL + honest_per_leaf * BLST_AGG_ADD_MS + BLST_AGG_VERIFY_MS)
        tree_total = tree_p1 + tree_p2
        lat_tree.append(tree_total)

        # HotStuff: retransmit RTT + 3 rounds * (global RTT + N agg)
        hs_retransmit = RTT_GLOBAL + n_partial * 0.1
        hs_per_round = RTT_GLOBAL + n * BLST_AGG_ADD_MS + BLST_AGG_VERIFY_MS
        hs_total = hs_retransmit + 3 * hs_per_round
        lat_hs.append(hs_total)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    # Left: total latency
    ax1.loglog(sizes, lat_tree, "D-", color=GREEN, lw=2.5, ms=6, label="Proxima Tree")
    ax1.loglog(sizes, lat_flat, "o-", color=BLUE, lw=2.5, ms=6, label="Proxima Flat")
    ax1.loglog(sizes, lat_hs, "s--", color=ORANGE, lw=2, ms=6, label="HotStuff")
    ax1.set_xlabel("Validators (N)")
    ax1.set_ylabel("Finality Latency (ms)")
    ax1.set_title("Finality Latency (3-region deployment)", fontweight="bold")
    ax1.legend(fontsize=10)

    # Mark crossover where tree clearly beats flat
    for i in range(1, len(sizes)):
        if lat_tree[i] < lat_flat[i] * 0.8 and lat_tree[i-1] >= lat_flat[i-1] * 0.8:
            ax1.axvline(x=sizes[i], color=GRAY, ls=":", alpha=0.5)
            ax1.text(sizes[i], ax1.get_ylim()[0] * 2,
                     f"tree wins\nN={sizes[i]:,}", fontsize=8, ha="center", color=GRAY)
            break

    # Annotate N=100K
    ax1.annotate(f"{lat_flat[-1]/1000:.1f}s", xy=(sizes[-1], lat_flat[-1]),
                 xytext=(-55, 10), textcoords="offset points", fontsize=9,
                 color=BLUE, fontweight="bold")
    ax1.annotate(f"{lat_tree[-1]:.0f}ms", xy=(sizes[-1], lat_tree[-1]),
                 xytext=(-55, -15), textcoords="offset points", fontsize=9,
                 color=GREEN, fontweight="bold")

    # Right: breakdown at N=100K
    n = breakdown_n
    n_honest = int(n * (1 - byz_frac))
    n_partial = int(n_honest * partial_frac)

    # Flat breakdown
    flat_network = 3 * RTT_GLOBAL
    flat_processing = (n * DIST_CHECK_MS + n_partial * BLOOM_CHECK_MS +
                       n_honest * BLST_AGG_ADD_MS + BLST_AGG_VERIFY_MS)

    # Tree breakdown
    n_leaves = math.ceil(n / branching)
    n_levels = 1
    nodes = n_leaves
    while nodes > 1:
        nodes = math.ceil(nodes / branching)
        n_levels += 1
    honest_per_leaf = int(branching * (1 - byz_frac))
    partial_per_leaf = int(honest_per_leaf * partial_frac)
    leaf_p = (branching * DIST_CHECK_MS + partial_per_leaf * BLOOM_CHECK_MS +
              honest_per_leaf * BLST_AGG_ADD_MS + BLST_AGG_VERIFY_MS)
    internal_p = branching * BLST_AGG_ADD_MS + BLST_AGG_VERIFY_MS
    tree_network = 2 * (RTT_LOCAL + max(0, n_levels - 2) * RTT_REGIONAL + RTT_GLOBAL)
    tree_processing = 2 * (leaf_p + max(0, n_levels - 2) * internal_p)

    # HotStuff breakdown
    hs_network = 4 * RTT_GLOBAL  # retransmit + 3 rounds
    hs_processing = (n_partial * 0.1 +
                     3 * (n * BLST_AGG_ADD_MS + BLST_AGG_VERIFY_MS))

    protocols = ["Proxima\nTree", "Proxima\nFlat", "HotStuff"]
    network_times = [tree_network, flat_network, hs_network]
    processing_times = [tree_processing, flat_processing, hs_processing]

    x = np.arange(3)
    w = 0.5
    bars_net = ax2.bar(x, network_times, w, label="Network RTT", color=GRAY, alpha=0.7)
    bars_proc = ax2.bar(x, processing_times, w, bottom=network_times,
                        label="BLS + Processing", color=[GREEN, BLUE, ORANGE], alpha=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(protocols)
    ax2.set_ylabel("Latency (ms)")
    ax2.set_title(f"Latency Breakdown at N={breakdown_n:,}", fontweight="bold")
    ax2.legend(fontsize=10)

    # Label totals on bars
    for i, (net, proc) in enumerate(zip(network_times, processing_times)):
        total = net + proc
        if total > 1000:
            label = f"{total/1000:.1f}s"
        else:
            label = f"{total:.0f}ms"
        ax2.text(i, total + max(processing_times) * 0.02, label,
                 ha="center", fontsize=10, fontweight="bold")

    fig.suptitle("Latency Model: Network RTT + BLS Processing (projected, blst)",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "fig7_latency_model.png")


# =====================================================================
# Figure 8: Cross-shard verification overhead
# =====================================================================

def figure8():
    """Cross-shard message overhead: digest comparison vs 2PC vs receipts."""
    print("Figure 8: Cross-shard verification overhead...")

    from cross_shard_sim import two_phase_commit_cost, receipt_cost, digest_cost

    n_txs = 1000
    n_val = 100
    prop_rates = np.arange(0.50, 1.005, 0.01)

    tpc = two_phase_commit_cost(n_txs, n_val)
    rec = receipt_cost(n_txs, n_val)

    digest_msgs = []
    digest_bytes = []
    for pr in prop_rates:
        d = digest_cost(n_txs, float(pr), n_val)
        digest_msgs.append(d["msgs"])
        digest_bytes.append(d["bytes"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    # Left: messages vs propagation rate
    ax1.axhline(y=tpc["msgs"], color=RED, lw=2, ls="--", label="2PC")
    ax1.axhline(y=rec["msgs"], color=ORANGE, lw=2, ls="--", label="Receipt (NEAR)")
    ax1.plot(prop_rates * 100, digest_msgs, "-", color=GREEN, lw=2.5, label="Digest (Proxima)")
    ax1.set_xlabel("Pre-deadline Propagation Rate (%)")
    ax1.set_ylabel("Total Messages")
    ax1.set_title("Cross-Shard Messages (1000 txs)", fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.set_xlim(50, 100)

    # Mark the 95% operating point
    d95 = digest_cost(n_txs, 0.95, n_val)
    ax1.plot(95, d95["msgs"], "D", color=GREEN, ms=10, zorder=5)
    ax1.annotate(f"{d95['msgs']:,} msgs\n({(1-d95['msgs']/tpc['msgs'])*100:.0f}% vs 2PC)",
                 xy=(95, d95["msgs"]), xytext=(-80, 30), textcoords="offset points",
                 fontsize=9, fontweight="bold", color=GREEN,
                 arrowprops=dict(arrowstyle="->", color=GREEN))

    # Right: bandwidth vs propagation rate
    ax2.axhline(y=tpc["bytes"]/1024, color=RED, lw=2, ls="--", label="2PC")
    ax2.axhline(y=rec["bytes"]/1024, color=ORANGE, lw=2, ls="--", label="Receipt (NEAR)")
    ax2.plot(prop_rates * 100, [b/1024 for b in digest_bytes], "-",
             color=GREEN, lw=2.5, label="Digest (Proxima)")
    ax2.set_xlabel("Pre-deadline Propagation Rate (%)")
    ax2.set_ylabel("Total Bandwidth (KB)")
    ax2.set_title("Cross-Shard Bandwidth (1000 txs)", fontweight="bold")
    ax2.legend(fontsize=10)
    ax2.set_xlim(50, 100)

    d95_kb = d95["bytes"] / 1024
    ax2.plot(95, d95_kb, "D", color=GREEN, ms=10, zorder=5)
    ax2.annotate(f"{d95_kb:.0f} KB",
                 xy=(95, d95_kb), xytext=(-60, 25), textcoords="offset points",
                 fontsize=9, fontweight="bold", color=GREEN,
                 arrowprops=dict(arrowstyle="->", color=GREEN))

    fig.suptitle("Cross-Shard Overhead: Digest Comparison vs 2PC vs Receipts",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "fig8_cross_shard.png")


# =====================================================================

def main():
    t0 = time.time()
    print("Generating figures...\n")
    figure1()
    figure2()
    figure3()
    figure4()
    figure5()
    figure6()
    figure7()
    figure8()
    print(f"\nDone in {time.time()-t0:.1f}s. Figures in {OUT}/")


if __name__ == "__main__":
    main()
