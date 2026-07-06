#!/usr/bin/env python3
"""
visualize_v2_paper.py -- Full figure suite for the v2 journal manuscript.

Writes figures to paper_v2/figures/ and the headline numbers to
paper_v2/results.json so the text can quote exactly what the plots show.

Figures (plan items in parentheses):
  F1 estimator collapse, misses vs substitutions          (E1)
  F2 exact small-d distributions vs tau^2 = 3n            (E2)
  F3 reference robustness under coordinated drag          (E3)
  F4 fast path: griefing fix and readiness sensing        (E4, C1)
  F5 sync cost: bloom vs IBLT vs true difference          (E6/C3)
  F6 conservation: verification bytes and localization    (D)
  F7 message/bandwidth scaling with HS2 and Kauri         (E8)
  F8 fork localization and partition healing              (C4, C5)
"""

import json
import math
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import blockchain as bc
bc.USE_REAL_BLS = False

import dpmh
import reconcile
import conservation
from blockchain import (
    BLSKeyPair, Blockchain, make_validators, make_partial_obs,
    calibrate_threshold, vector_consensus, tree_consensus, readiness_sense,
    heal_partition,
)
from hotstuff import (hotstuff_consensus, hotstuff2_consensus,
                      kauri_consensus, pbft_consensus)

OUT = "paper_v2/figures"
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.titlesize": 10,
    "figure.dpi": 200, "savefig.dpi": 200, "axes.grid": True,
    "grid.alpha": 0.3, "legend.fontsize": 7.5,
})
BLUE, GREEN, ORANGE, RED, GRAY, PURPLE = ("#2E75B6", "#27AE60", "#E67E22",
                                          "#C0392B", "#7F8C8D", "#6C5CE7")
N = dpmh.N_ROUND
RESULTS = {}


def save(fig, name):
    fig.tight_layout()
    fig.savefig(f"{OUT}/{name}", bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {OUT}/{name}")


def build(n_h, n_b, strategy="drop_half", n_txs=20, miss=0.37):
    BLSKeyPair._counter = 1
    sv, sh, sb = make_validators(n_h, n_b, strategy)
    sc = Blockchain(sv)
    for i in range(5):
        sc.register_account(f"S-{i}", 100000)
    for i in range(n_txs):
        tx = sc.make_tx(f"S-{i % 5}", f"S-{(i + 1) % 5}", 1.0)
        if tx:
            sc.submit_tx(tx)
    blk = sc.propose_block(sh[0])
    tau = calibrate_threshold()
    po = make_partial_obs(sv, len(blk.tx_data_strings), miss_prob=miss)
    return sc, sv, blk, tau, po


# ===========================================================================
def fig1_estimator():
    print("F1: estimator collapse...")
    rng = np.random.default_rng(42)

    def sample(dm, ds, trials=400):
        out = []
        for t in range(trials):
            txs = [f"tx-{t}-{i}" for i in range(40)]
            full = dpmh.digest(txs, height=t)
            other = list(txs)
            if dm:
                drop = rng.choice(len(other), size=dm, replace=False)
                other = [x for i, x in enumerate(other)
                         if i not in set(drop.tolist())]
            for s in range(ds):
                other[s] = f"SUB-{t}-{s}"
            out.append(dpmh.dist2(full, dpmh.digest(other, height=t)))
        return np.array(out)

    cases = [("miss 1", 1, 0, 1), ("miss 2", 2, 0, 2), ("miss 4", 4, 0, 4),
             ("subst 1", 0, 1, 2), ("subst 2", 0, 2, 4), ("subst 4", 0, 4, 8)]
    data, expected, labels = [], [], []
    for label, dm, ds, d_true in cases:
        data.append(sample(dm, ds) / N)
        expected.append(d_true)
        labels.append(f"{label}\n(d={d_true})")

    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, widths=0.5)
    for i, b in enumerate(bp["boxes"]):
        b.set_facecolor(BLUE if i < 3 else ORANGE)
        b.set_alpha(0.7)
    xs = np.arange(1, len(cases) + 1)
    ax.plot(xs, expected, "k--", lw=1.2, label=r"analytic $\|\Delta D\|^2/n = d$")
    ax.scatter(xs, expected, color="k", zorder=5, s=10)
    ax.set_ylabel(r"$\|\Delta D\|^2 / n$")
    ax.tick_params(axis="x", labelsize=6.5)
    ax.legend()
    save(fig, "fig1_estimator.png")
    RESULTS["estimator_miss1_variance"] = float(np.var(data[0]))


# ===========================================================================
def fig2_threshold():
    print("F2: threshold distributions...")
    rng = np.random.default_rng(43)
    fig, ax = plt.subplots(figsize=(3.5, 2.2))
    for d, c in zip((1, 2, 3, 4), (GREEN, BLUE, ORANGE, RED)):
        samples = []
        for t in range(600):
            txs = [f"tx-{t}-{i}" for i in range(40)]
            drop = rng.choice(len(txs), size=d, replace=False)
            sub = [x for i, x in enumerate(txs) if i not in set(drop.tolist())]
            samples.append(dpmh.dist2(dpmh.digest(txs, height=t),
                                      dpmh.digest(sub, height=t)) / N)
        ax.hist(samples, bins=40, density=True, alpha=0.55, color=c,
                label=f"d = {d}")
    ax.axvline(3.0, color="k", ls="--", lw=1.4, label=r"$\tau^2 = 3n$")
    ax.set_xlabel(r"$\|\Delta D\|^2 / n$")
    ax.set_ylabel("density")
    ax.set_yscale("log")
    ax.legend()
    save(fig, "fig2_threshold.png")


# ===========================================================================
def fig3_reference():
    print("F3: reference robustness under coordinated drag...")
    np.random.seed(44)
    n_total = 90
    fracs = [0.0, 0.1, 0.2, 0.3, 0.4, 0.45]
    trials = 8
    excl_median, excl_own = [], []

    for frac in fracs:
        n_b = int(n_total * frac)
        n_h = n_total - n_b
        med_rate, own_rate = [], []
        for t in range(trials):
            sc, sv, blk, tau, po = build(n_h, n_b, strategy="random_vector",
                                         miss=0.37)
            tau2 = dpmh.tau2()
            honest_full = dpmh.digest(blk.tx_data_strings, blk.height)
            # Coordinated drag: all Byzantine submit one extreme point.
            drag = honest_full + 40
            vecs = []
            for v in sv:
                if v.is_byzantine:
                    vecs.append(drag)
                else:
                    miss = po.get(v.id, set())
                    vecs.append(v.get_vector(blk, miss))
            # v2 reference: coordinate-wise median of all submissions.
            ref_med = dpmh.robust_reference(vecs)
            # v1-style reference: the aggregator's own state; with a
            # Byzantine aggregator this is the drag point itself.
            ref_own = drag
            h_ids = [i for i, v in enumerate(sv) if not v.is_byzantine]
            med_rate.append(np.mean([dpmh.dist2(vecs[i], ref_med) > tau2
                                     for i in h_ids]))
            own_rate.append(np.mean([dpmh.dist2(vecs[i], ref_own) > tau2
                                     for i in h_ids]))
        excl_median.append(np.mean(med_rate))
        excl_own.append(np.mean(own_rate))

    fig, ax = plt.subplots(figsize=(3.5, 2.2))
    ax.plot([f * 100 for f in fracs], [e * 100 for e in excl_own], "s--",
            color=RED, ms=4, label="v1: aggregator-chosen reference")
    ax.plot([f * 100 for f in fracs], [e * 100 for e in excl_median], "D-",
            color=GREEN, ms=4, label="v2: coordinate-wise median")
    ax.set_xlabel("Byzantine fraction (%), coordinated drag + Byz aggregator")
    ax.set_ylabel("honest validators excluded (%)")
    ax.set_ylim(-5, 105)
    ax.legend()
    save(fig, "fig3_reference.png")
    RESULTS["reference_drag"] = {
        "fracs": fracs, "median_excl": excl_median, "own_excl": excl_own}


# ===========================================================================
def fig4_fastpath():
    print("F4: griefing fix and readiness sensing...")
    np.random.seed(45)
    # Panel (a): fast-path rate vs number of in-cluster Byzantine (mimic).
    byz_counts = [0, 1, 2, 4, 8]
    trials = 12
    rate_v1, rate_v2 = [], []
    for nb in byz_counts:
        v1_hits = v2_hits = 0
        for t in range(trials):
            sc, sv, blk, tau, _ = build(60, nb, strategy="mimic_honest",
                                        miss=0.0)
            r = vector_consensus(sv, blk, tau, {})
            v2_hits += r["fast_path"]
            # v1 trigger: cluster variance ~ 0 (recomputed from distances).
            n_req = int(math.ceil(len(sv) * 2 / 3))
            ref = dpmh.digest(blk.tx_data_strings, blk.height)
            dists = [math.sqrt(dpmh.dist2(v.get_vector(blk, set()), ref))
                     for v in sv
                     if dpmh.dist2(v.get_vector(blk, set()), ref)
                     <= dpmh.tau2()]
            var = float(np.var(dists)) if dists else 999.0
            v1_hits += (var < 1e-6 and len(dists) >= n_req)
        rate_v1.append(v1_hits / trials * 100)
        rate_v2.append(v2_hits / trials * 100)

    # Panel (b): fast-path rate vs p_miss, fixed vs sensed proposal.
    miss_rates = [0.0, 0.1, 0.2, 0.4, 0.6, 0.8]
    trials_b = 16
    fixed_rate, sensed_rate, sensed_ticks = [], [], []
    for pm in miss_rates:
        f_hits = s_hits = 0
        ticks = []
        for t in range(trials_b):
            sc, sv, blk, tau, po = build(40, 12, miss=pm)
            f_hits += vector_consensus(sv, blk, tau, po)["fast_path"]
            sc2, sv2, blk2, tau2v, po2 = build(40, 12, miss=pm)
            rs = readiness_sense(sv2, blk2, tau2v, po2, fill_prob=0.5,
                                 max_ticks=8)
            s_hits += rs["fast_path"]
            ticks.append(rs["sense_ticks"])
        fixed_rate.append(f_hits / trials_b * 100)
        sensed_rate.append(s_hits / trials_b * 100)
        sensed_ticks.append(np.mean(ticks))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.3))
    ax1.plot(byz_counts, rate_v1, "s--", color=RED, ms=4,
             label="v1: variance trigger")
    ax1.plot(byz_counts, rate_v2, "D-", color=GREEN, ms=4,
             label="v2: commitment trigger")
    ax1.set_xlabel("in-cluster Byzantine validators (mimic)")
    ax1.set_ylabel("fast-path rate (%)")
    ax1.set_ylim(-5, 105)
    ax1.legend()
    ax2.plot([m * 100 for m in miss_rates], fixed_rate, "o--", color=GRAY,
             ms=4, label="fixed-interval proposal")
    ax2.plot([m * 100 for m in miss_rates], sensed_rate, "D-", color=PURPLE,
             ms=4, label="sensed proposal (8-tick cap)")
    ax2.set_xlabel(r"$p_{\mathrm{miss}}$ at first observation (%)")
    ax2.set_ylabel("fast-path rate (%)")
    ax2.set_ylim(-5, 105)
    ax2.legend()
    save(fig, "fig4_fastpath.png")
    RESULTS["griefing"] = {"byz": byz_counts, "v1": rate_v1, "v2": rate_v2}
    RESULTS["sensing"] = {"miss": miss_rates, "fixed": fixed_rate,
                          "sensed": sensed_rate, "ticks": sensed_ticks}


# ===========================================================================
def fig5_sync():
    print("F5: bloom vs IBLT sync cost...")
    np.random.seed(46)
    set_size = 500
    ds = [1, 2, 4, 8, 16, 32, 64]
    bloom_bytes = []
    iblt_bytes, iblt_rounds = [], []
    base = set(f"item-{i}" for i in range(set_size))
    for d in ds:
        other = set(list(base)[d // 2:]) | {f"n-{i}" for i in range(d - d // 2)}
        # v1 bloom: filter over the whole local set (1% per-lookup FP),
        # m = -|S| ln p / ln^2 2 bits, independent of d.
        m_bits = int(-set_size * math.log(0.01) / (math.log(2) ** 2))
        bloom_bytes.append(m_bits // 8)
        res = reconcile.reconcile(base, other, d_hint=d)
        assert res["ok"]
        iblt_bytes.append(res["bytes"])
        iblt_rounds.append(res["rounds"])

    fig, ax = plt.subplots(figsize=(3.5, 2.3))
    ax.plot(ds, bloom_bytes, "s--", color=RED, ms=4,
            label=f"v1 bloom (|S|={set_size}, FP risk)")
    ax.plot(ds, iblt_bytes, "D-", color=GREEN, ms=4,
            label="v2 IBLT (exact decode)")
    ax.set_xlabel("true symmetric difference d")
    ax.set_ylabel("sync sketch bytes")
    ax.set_xscale("log", base=2)
    ax.legend()
    save(fig, "fig5_sync.png")
    RESULTS["sync"] = {"d": ds, "bloom": bloom_bytes, "iblt": iblt_bytes,
                       "rounds": iblt_rounds}


# ===========================================================================
def fig6_conservation():
    print("F6: conservation verification and localization...")
    S = 16
    tx_range = [100, 500, 1000, 2000, 4000, 8000]
    cons = [conservation.conservation_verification_bytes(S) / 1024
            for _ in tx_range]
    rec = [conservation.receipts_verification_bytes(v, S) / 1024
           for v in tx_range]
    tpc = [conservation.two_pc_verification_bytes(v, S) / 1024
           for v in tx_range]

    C = 64
    fs = [1, 2, 4, 8, 16]
    probes, bounds = [], []
    for f in fs:
        r = conservation.simulate(n_shards=C, blocks=4, txs_per_corridor=30,
                                  n_faulty_corridors=f, faults_per_corridor=3,
                                  seed=100 + f)
        assert r["localization_exact"], f
        probes.append(r["probes"])
        bounds.append(2 * f * (math.log2(C / f) + 2))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.3))
    ax1.plot(tx_range, rec, "s--", color=ORANGE, ms=4, label="batched receipts")
    ax1.plot(tx_range, tpc, "^:", color=RED, ms=4, label="batched 2PC")
    ax1.plot(tx_range, cons, "D-", color=GREEN, ms=4,
             label="conservation (volume-independent)")
    ax1.set_xlabel(f"cross-shard txs per block (S={S})")
    ax1.set_ylabel("verification KB per block")
    ax1.set_xscale("log")
    ax1.legend()
    ax2.plot(fs, probes, "D-", color=GREEN, ms=4, label="measured probes")
    ax2.plot(fs, bounds, "k--", lw=1.2, label=r"$2f(\log_2(C/f)+2)$")
    ax2.set_xlabel(f"faulty corridors f (C={C})")
    ax2.set_ylabel("localization probes")
    ax2.legend()
    save(fig, "fig6_conservation.png")
    RESULTS["conservation"] = {
        "verification_kb": {"cons": cons[0], "receipts_at_2000": rec[3],
                            "crossover_txs": int(
                                (conservation.conservation_verification_bytes(S)
                                 - S * (32 + 96)) / 32)},
        "localization": {"f": fs, "probes": probes}}


# ===========================================================================
def fig7_scaling():
    print("F7: message/bandwidth scaling with all baselines...")
    np.random.seed(47)
    sizes = [100, 300, 1000, 3000]
    rows = {k: [] for k in ("tree", "flat", "hs", "hs2", "kauri", "pbft")}
    bw = {k: [] for k in ("tree", "flat", "hs", "hs2", "kauri", "pbft")}
    for Nv in sizes:
        nh, nb = int(Nv * 0.7), Nv - int(Nv * 0.7)
        sc, sv, blk, tau, po = build(nh, nb)
        rt = tree_consensus(sv, blk, tau, po, 10)
        rf = vector_consensus(sv, blk, tau, po)
        rh = hotstuff_consensus(Nv, nb, 20)
        r2 = hotstuff2_consensus(Nv, nb, 20)
        rk = kauri_consensus(Nv, nb, 20)
        rp = pbft_consensus(Nv, nb, 20)
        for k, r in zip(rows, (rt, rf, rh, r2, rk, rp)):
            rows[k].append(r["msgs"])
            bw[k].append(r["msg_bytes"] / 1024)
        print(f"    N={Nv}: tree={rt['msgs']:,} flat={rf['msgs']:,} "
              f"hs={rh['msgs']:,} hs2={r2['msgs']:,} kauri={rk['msgs']:,}")

    style = {"tree": ("D-", GREEN, "Proxima Tree"),
             "flat": ("o-", BLUE, "Proxima Flat"),
             "hs": ("s--", ORANGE, "HotStuff"),
             "hs2": ("v--", PURPLE, "HotStuff-2"),
             "kauri": ("x--", GRAY, "Kauri (msg model)"),
             "pbft": ("^:", RED, "PBFT")}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.4))
    for k, (fmt, c, lbl) in style.items():
        ax1.loglog(sizes, rows[k], fmt, color=c, ms=4, label=lbl)
        if k != "pbft":
            ax2.loglog(sizes, bw[k], fmt, color=c, ms=4, label=lbl)
    ax1.set_xlabel("validators N")
    ax1.set_ylabel("messages per block")
    ax1.legend(ncol=2, fontsize=6.2)
    ax2.set_xlabel("validators N")
    ax2.set_ylabel("KB per block")
    ax2.legend(ncol=2, fontsize=6.2)
    save(fig, "fig7_scaling.png")
    RESULTS["scaling"] = {"sizes": sizes, "msgs": rows,
                          "kb": {k: [round(x) for x in v]
                                 for k, v in bw.items()}}


# ===========================================================================
def fig8_fork_heal():
    print("F8: fork localization and partition healing...")
    np.random.seed(48)
    heights = [64, 256, 1024, 4096]
    probes = []
    for H in heights:
        A, B = dpmh.Accumulator(), dpmh.Accumulator()
        fork_at = int(H * 0.66)
        for h in range(H):
            blk = [f"b{h}-{i}" for i in range(4)]
            A.append_block(blk)
            B.append_block(blk if h < fork_at
                           else [f"ALT{h}-{i}" for i in range(4)])
        r = dpmh.find_fork(A, B)
        assert r["fork_height"] == fork_at
        probes.append(r["probes"])

    ds = [2, 4, 8, 16, 32, 64, 128]
    heal_bytes, heal_est = [], []
    base = set(f"t-{i}" for i in range(600))
    for d in ds:
        other = (base - set(list(base)[:d // 2])) | {
            f"p-{i}" for i in range(d - d // 2)}
        hp = heal_partition(base, other, height=5)
        assert hp["recovered_union"]
        heal_bytes.append(hp["bytes"] + hp["push_bytes"])
        heal_est.append(hp["d_hat"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.3))
    ax1.semilogx(heights, probes, "D-", color=GREEN, ms=4, base=2,
                 label="measured probes")
    ax1.semilogx(heights, [math.log2(h) + 1 for h in heights], "k--",
                 lw=1.2, base=2, label=r"$\log_2 H + 1$")
    ax1.set_xlabel("chain height H")
    ax1.set_ylabel("fork-localization probes")
    ax1.legend()
    ax2.loglog(ds, heal_bytes, "D-", color=PURPLE, ms=4, base=2,
               label="sketch + push bytes")
    ax2.loglog(ds, [d * 200 for d in ds], "k--", lw=1.2, base=2,
               label="200 B per divergent tx")
    ax2.set_xlabel("partition divergence d")
    ax2.set_ylabel("healing bytes")
    ax2.legend()
    save(fig, "fig8_fork_heal.png")
    RESULTS["fork"] = {"H": heights, "probes": probes}
    RESULTS["heal"] = {"d": ds, "bytes": heal_bytes,
                       "d_hat": [round(x, 1) for x in heal_est]}


# ===========================================================================
if __name__ == "__main__":
    import time
    t0 = time.time()
    fig1_estimator()
    fig2_threshold()
    fig3_reference()
    fig4_fastpath()
    fig5_sync()
    fig6_conservation()
    fig7_scaling()
    fig8_fork_heal()
    # conservation headline numbers for the text
    r = conservation.simulate()
    RESULTS["conservation_sim"] = {
        "faults": r["faults_injected"],
        "estimate": round(r["dangling_estimate"], 1),
        "probes": r["probes"],
        "localization_exact": r["localization_exact"],
        "decode_exact": r["decode_exact"],
        "verify_kb": round(r["verify_bytes"]["conservation"] / 1024, 1),
        "crossover": r["receipts_crossover_txs"],
    }
    with open("paper_v2/results.json", "w") as f:
        json.dump(RESULTS, f, indent=1)
    print(f"\nDone in {time.time() - t0:.1f}s; results in paper_v2/results.json")
