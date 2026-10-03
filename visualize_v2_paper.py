#!/usr/bin/env python3
"""
visualize_v2_paper.py -- Figure suite and results for the v2 manuscript.

Writes figures to paper_v2/figures/ and every number the text quotes to
paper_v2/results.json. Deterministic: all randomness is seeded, all set
iterations are sorted, and the ledger uses a logical clock.

  F1 estimator: misses and substitutions vs the analytic mean
  F2 exact distributions of ||dD||^2 / n for d = 1..4 vs tau^2 = 3n
  F3 deflation attack: d_hat vs true difference under greedy balancing
  F4 fast path: griefing at the safe quorum; sensing vs equal-mean delay
  F5 sync: block-level fetch-by-id vs mempool-level IBLT sizing
  F6 conservation: verification bytes vs S (ring, mesh, corridor roots);
     localization probes vs f
  F7 scaling: messages, bytes, and leader ingress against all baselines
  F8 fork localization and partition healing
  plus the safety fuzzing table (no figure)
Run: python visualize_v2_paper.py   (about 3 minutes)
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

import bft
import conservation
import dpmh
import reconcile
from blockchain import (
    Blockchain, make_validators, make_partial_obs, vector_consensus,
    tree_consensus, readiness_sense, fixed_delay_propose, heal_partition,
    fetch_bytes,
)
from hotstuff import (hotstuff_consensus, hotstuff2_consensus,
                      chained_hotstuff_consensus, kauri_consensus,
                      pbft_consensus)

OUT = "paper_v2/figures"
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.titlesize": 10,
    "figure.dpi": 200, "savefig.dpi": 200, "axes.grid": True,
    "grid.alpha": 0.3, "legend.fontsize": 7,
})
BLUE, GREEN, ORANGE, RED, GRAY, PURPLE, BROWN = (
    "#2E75B6", "#27AE60", "#E67E22", "#C0392B", "#7F8C8D", "#6C5CE7", "#8D6E63")
N = dpmh.N_ROUND
RESULTS = {}


def save(fig, name):
    fig.tight_layout()
    fig.savefig(f"{OUT}/{name}", bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {OUT}/{name}")


def build(n_h, n_b, strategy="abstain", n_txs=20, miss=0.37, seed=0):
    sv, sh, _ = make_validators(n_h, n_b, strategy, seed=seed)
    sc = Blockchain(sv)
    for i in range(5):
        sc.register_account(f"S-{i}", 100000)
    for i in range(n_txs):
        sc.submit_tx(sc.make_tx(f"S-{i % 5}", f"S-{(i + 1) % 5}", 1.0))
    blk = sc.propose_block(sh[0])
    po = make_partial_obs(sv, len(blk.tx_data_strings), miss_prob=miss,
                          rng=np.random.default_rng(seed))
    return sv, blk, po


def binom_pmf(n, p):
    k = np.arange(n + 1)
    logc = (np.vectorize(math.lgamma)(n + 1) - np.vectorize(math.lgamma)(k + 1)
            - np.vectorize(math.lgamma)(n - k + 1))
    return np.exp(logc + k * math.log(p) + (n - k) * math.log(1 - p))


# ===========================================================================
def fig1_estimator():
    print("F1: estimator")
    rng = np.random.default_rng(42)

    def sample(dm, ds, trials=400):
        out = []
        for t in range(trials):
            txs = [f"tx-{t}-{i}" for i in range(40)]
            other = list(txs)
            if dm:
                drop = set(rng.choice(40, size=dm, replace=False).tolist())
                other = [x for i, x in enumerate(other) if i not in drop]
            for s in range(ds):
                other[s] = f"SUB-{t}-{s}"
            out.append(dpmh.dist2(dpmh.digest(txs, t), dpmh.digest(other, t)) / N)
        return np.array(out)

    cases = [("miss 1", 1, 0, 1), ("miss 2", 2, 0, 2), ("miss 4", 4, 0, 4),
             ("subst 1", 0, 1, 2), ("subst 2", 0, 2, 4), ("subst 4", 0, 4, 8)]
    data = [sample(dm, ds) for _, dm, ds, _ in cases]
    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    bp = ax.boxplot(data, tick_labels=[f"{l}\n(d={d})" for l, _, _, d in cases],
                    patch_artist=True, widths=0.5)
    for i, b in enumerate(bp["boxes"]):
        b.set_facecolor(BLUE if i < 3 else ORANGE)
        b.set_alpha(0.7)
    xs = np.arange(1, len(cases) + 1)
    ax.plot(xs, [c[3] for c in cases], "k--", lw=1.2, label=r"$E\|\Delta D\|^2/n = d$")
    ax.set_ylabel(r"$\|\Delta D\|^2 / n$")
    ax.tick_params(axis="x", labelsize=6.5)
    ax.legend()
    save(fig, "fig1_estimator.png")
    RESULTS["estimator"] = {
        "means": {c[0]: round(float(d.mean()), 3) for c, d in zip(cases, data)},
        "miss1_variance": float(np.var(data[0]))}


# ===========================================================================
def fig2_threshold():
    print("F2: exact distributions")
    fig, ax = plt.subplots(figsize=(3.5, 2.2))
    tails = {}
    # d=1: point mass at n. d=2: 4 Bin(n,1/2). d=3: n + 8 Bin(n,1/4).
    # d=4: sum over coordinates of S^2, S in {-4,-2,0,2,4}; exact by DP.
    pm2 = binom_pmf(N, 0.5)
    x2 = 4 * np.arange(N + 1) / N
    pm3 = binom_pmf(N, 0.25)
    x3 = (N + 8 * np.arange(N + 1)) / N
    # d=4 per-coordinate S^2/4 in {0, 1, 4} w.p. {6/16, 8/16, 2/16}
    dist = np.zeros(4 * N + 1)
    dist[0] = 1.0
    for _ in range(N):
        nd = 6 / 16 * dist
        nd[1:] += 8 / 16 * dist[:-1]
        nd[4:] += 2 / 16 * dist[:-4]
        dist = nd
    x4 = 4 * np.arange(4 * N + 1) / N
    ax.axvline(1.0, color=GREEN, lw=2, label="d = 1 (exactly n)")
    for x, pm, c, lbl in ((x2, pm2, BLUE, "d = 2"), (x3, pm3, ORANGE, "d = 3"),
                          (x4, dist, RED, "d = 4")):
        m = pm > 1e-300
        ax.semilogy(x[m], pm[m], color=c, lw=1.2, label=lbl)
    ax.axvline(3.0, color="k", ls="--", lw=1.2, label=r"$\tau^2 = 3n$")
    ax.set_ylim(1e-40, 1)
    ax.set_xlim(0, 7)
    ax.set_xlabel(r"$\|\Delta D\|^2 / n$")
    ax.set_ylabel("probability")
    ax.legend(fontsize=6)
    save(fig, "fig2_threshold.png")
    tails["d2_exceeds"] = float(pm2[x2 > 3].sum())
    tails["d3_exceeds"] = float(pm3[x3 > 3].sum())
    tails["d4_below_or_eq"] = float(dist[x4 <= 3].sum())
    for nn in (128,):
        p2 = binom_pmf(nn, 0.5)
        tails[f"light_d2_exceeds"] = float(p2[4 * np.arange(nn + 1) > 3 * nn].sum())
        dd = np.zeros(4 * nn + 1)
        dd[0] = 1.0
        for _ in range(nn):
            nd = 6 / 16 * dd
            nd[1:] += 8 / 16 * dd[:-1]
            nd[4:] += 2 / 16 * dd[:-4]
            dd = nd
        tails["light_d4_below_or_eq"] = float(dd[4 * np.arange(4 * nn + 1) <= 3 * nn].sum())
    RESULTS["threshold_tails"] = tails


# ===========================================================================
def fig3_deflation():
    print("F3: deflation attack")
    ms = [50, 100, 200, 400, 800, 1600]
    pools = [1, 16, 256]
    curves = {p: [] for p in pools}
    for p in pools:
        for m in ms:
            curves[p].append(dpmh.adversarial_balance(m=m, pool=p, seed=7)["d_hat"])
    fig, ax = plt.subplots(figsize=(3.5, 2.3))
    ax.loglog(ms, ms, "k--", lw=1.2, label=r"honest: $\hat d = m$")
    for p, c in zip(pools, (ORANGE, PURPLE, RED)):
        ax.loglog(ms, curves[p], "o-", color=c, ms=3.5,
                  label=f"greedy, pool {p}" + (" (side choice only)" if p == 1 else ""))
    ax.axhline(3, color=GRAY, ls=":", lw=1, label=r"$\tau^2/n = 3$")
    ax.set_xlabel("true symmetric difference m")
    ax.set_ylabel(r"reported $\hat d$")
    ax.legend(fontsize=6)
    save(fig, "fig3_deflation.png")
    RESULTS["deflation"] = {"m": ms, **{f"pool_{p}": [round(x, 1) for x in curves[p]]
                                        for p in pools}}


# ===========================================================================
def fig4_fastpath():
    print("F4: fast path")
    n_total, f_cfg = 100, 20
    n_fast = bft.fast_quorum(n_total, f_cfg)
    ks = [0, 1, 2, 4, 8, 16, 19, 20]
    v1_mimic, v2_mimic, v2_abstain = [], [], []
    for k in ks:
        sv, blk, _ = build(n_total - k, k, "mimic_honest", miss=0.0, seed=k)
        r = vector_consensus(sv, blk, None, {}, f=f_cfg)
        v2_mimic.append(100.0 * r["fast_path"])
        # v1 trigger: fire only if every in-cluster sketch matches D(B)
        # (zero variance) and enough signatures arrived.
        d = [x for x in r["distances"].values() if x * x <= dpmh.tau2()]
        v1_mimic.append(100.0 * (len(d) >= n_fast and float(np.var(d)) < 1e-9))
        sv, blk, _ = build(n_total - k, k, "abstain", miss=0.0, seed=k)
        v2_abstain.append(100.0 * vector_consensus(sv, blk, None, {}, f=f_cfg)["fast_path"])

    miss_rates = [0.0, 0.1, 0.2, 0.4, 0.6, 0.8]
    trials = 40
    sensed, fixed, mean_ticks = [], [], []
    for pm in miss_rates:
        s_hits, ticks = 0, []
        for t in range(trials):
            sv, blk, po = build(n_total, 0, miss=pm, seed=1000 + t)
            rs = readiness_sense(sv, blk, po, fill_prob=0.5, max_ticks=8,
                                 f=f_cfg, rng=np.random.default_rng(t))
            s_hits += rs["fast_path"]
            ticks.append(rs["sense_ticks"])
        delay = int(round(np.mean(ticks)))
        f_hits = 0
        for t in range(trials):
            sv, blk, po = build(n_total, 0, miss=pm, seed=1000 + t)
            f_hits += fixed_delay_propose(sv, blk, po, delay, fill_prob=0.5,
                                          f=f_cfg, rng=np.random.default_rng(t))["fast_path"]
        sensed.append(100.0 * s_hits / trials)
        fixed.append(100.0 * f_hits / trials)
        mean_ticks.append(float(np.mean(ticks)))

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 2.3))
    a1.plot(ks, v1_mimic, "s--", color=RED, ms=4, label="v1 variance trigger, lying sketches")
    a1.plot(ks, v2_mimic, "D-", color=GREEN, ms=4, label="signature trigger, lying sketches")
    a1.plot(ks, v2_abstain, "x:", color=GRAY, ms=5, label="signature trigger, abstaining")
    a1.axvline(n_total - n_fast, color="k", lw=0.8, ls=":")
    a1.set_xlabel(f"Byzantine validators (N={n_total}, f={f_cfg}, " + r"$n_{fast}$" + f"={n_fast})")
    a1.set_ylabel("fast-path rate (%)")
    a1.set_ylim(-5, 105)
    a1.legend(fontsize=5.8)
    a2.plot([m * 100 for m in miss_rates], fixed, "o--", color=GRAY, ms=4,
            label="fixed delay = mean sensed delay")
    a2.plot([m * 100 for m in miss_rates], sensed, "D-", color=PURPLE, ms=4,
            label="sensed proposal (8-tick cap)")
    a2.set_xlabel(r"$p_{\mathrm{miss}}$ at first observation (%)")
    a2.set_ylabel("fast-path rate (%)")
    a2.set_ylim(-5, 105)
    a2.legend(fontsize=6)
    save(fig, "fig4_fastpath.png")
    RESULTS["griefing"] = {"N": n_total, "f": f_cfg, "n_fast": n_fast, "k": ks,
                           "v1_mimic": v1_mimic, "v2_mimic": v2_mimic,
                           "v2_abstain": v2_abstain}
    RESULTS["sensing"] = {"miss": miss_rates, "sensed": sensed, "fixed": fixed,
                          "mean_ticks": mean_ticks,
                          "gossip_bytes_per_tick": n_total * bc.DIGEST_WIRE}


# ===========================================================================
def fig5_sync():
    print("F5: sync")
    # (a) Block level: the validator holds the proposal's id list, so it
    # fetches by id. A sketch or bloom filter would only add bytes.
    ks = [1, 2, 4, 8]
    block_txs = 21
    by_id = [fetch_bytes(k) for k in ks]
    bloom_block = [int(-block_txs * math.log(0.01) / (math.log(2) ** 2)) // 8
                   + k * 200 for k in ks]
    iblt_block = [reconcile.sync_cost(k) + k * 200 for k in ks]
    # (b) Mempool level: two peers' sets with no shared id list.
    size = 5000
    base = sorted(f"item-{i}" for i in range(size))
    ds = [2, 8, 32, 128, 512]
    est_b, true_b, none_b, none_r, est_r, bloom_b = [], [], [], [], [], []
    for d in ds:
        other = set(base[d // 2:]) | {f"n-{i}" for i in range(d - d // 2)}
        d_hat = dpmh.est_symdiff(dpmh.digest(base, 3), dpmh.digest(sorted(other), 3))
        r_est = reconcile.reconcile(base, other, d_hint=max(1.0, d_hat), session=d)
        r_true = reconcile.reconcile(base, other, d_hint=d, session=d)
        r_none = reconcile.reconcile(base, other, d_hint=None, session=d)
        assert r_est["ok"] and r_true["ok"] and r_none["ok"]
        sk = lambda r: r["bytes"] - r["fetch_bytes"]
        est_b.append(sk(r_est) + N * 2)        # plus the sketch itself
        true_b.append(sk(r_true))
        none_b.append(sk(r_none))
        est_r.append(r_est["rounds"])
        none_r.append(r_none["rounds"])
        bloom_b.append(int(-size * math.log(0.01) / (math.log(2) ** 2)) // 8)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 2.3))
    a1.plot(ks, by_id, "D-", color=GREEN, ms=4, label="fetch by id (used)")
    a1.plot(ks, iblt_block, "o--", color=PURPLE, ms=4, label="IBLT + push")
    a1.plot(ks, bloom_block, "s:", color=RED, ms=4, label="bloom + push")
    a1.set_xlabel(f"missing txs (block of {block_txs})")
    a1.set_ylabel("bytes")
    a1.legend(fontsize=6)
    a2.loglog(ds, bloom_b, "s:", color=RED, ms=4, label=f"bloom (|S|={size})")
    a2.loglog(ds, est_b, "D-", color=GREEN, ms=4, label="IBLT sized by sketch (+1 KB)")
    a2.loglog(ds, true_b, "^--", color=BLUE, ms=4, label="IBLT sized by oracle d")
    a2.loglog(ds, none_b, "o:", color=GRAY, ms=4, label="IBLT, no estimate (doubling)")
    a2.set_xlabel("true symmetric difference d (mempools)")
    a2.set_ylabel("sketch bytes")
    a2.legend(fontsize=5.8)
    save(fig, "fig5_sync.png")
    RESULTS["sync"] = {"block": {"k": ks, "by_id": by_id, "iblt": iblt_block,
                                 "bloom": bloom_block},
                       "mempool": {"d": ds, "sketch_sized": est_b,
                                   "oracle_sized": true_b, "no_estimate": none_b,
                                   "rounds_sketch": est_r, "rounds_none": none_r,
                                   "bloom": bloom_b}}


# ===========================================================================
def fig6_conservation():
    print("F6: conservation")
    Ss = [4, 8, 16, 32, 64, 128]
    cons = [conservation.conservation_verification_bytes(s) / 1024 for s in Ss]
    roots_ring = [conservation.corridor_root_verification_bytes(s) / 1024 for s in Ss]
    roots_mesh = [conservation.corridor_root_verification_bytes(s * (s - 1)) / 1024
                  for s in Ss]
    C = 64
    fs = [1, 2, 4, 8, 16]
    probes = []
    for f in fs:
        rng = np.random.default_rng(f)
        picks = sorted(set(int(x) for x in rng.choice(C, size=f, replace=False)))
        faults = {(i, (i + 1) % C): [("drop", 0)] for i in picks}
        r = conservation.simulate(n_shards=C, blocks=5, txs_per_corridor=6,
                                  faults=faults, seed=f)
        assert r["localization_exact"] and r["decode_exact"], f
        probes.append(r["probes"])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 2.3))
    a1.loglog(Ss, cons, "D-", color=GREEN, ms=4, label="conservation (any topology)")
    a1.loglog(Ss, roots_ring, "s--", color=ORANGE, ms=4, label="corridor roots, ring")
    a1.loglog(Ss, roots_mesh, "^:", color=RED, ms=4, label="corridor roots, full mesh")
    a1.set_xlabel("shards S")
    a1.set_ylabel("verification KB per block")
    a1.legend(fontsize=6)
    a2.plot(fs, probes, "D-", color=GREEN, ms=4, label="bisection probes")
    a2.plot(fs, [2 * f * (math.ceil(math.log2(C / f))) + 1 for f in fs], "k--", lw=1,
            label=r"$2f\lceil\log_2(C/f)\rceil+1$")
    a2.axhline(C, color=GRAY, ls=":", lw=1, label=f"test each corridor (C={C})")
    a2.set_xlabel(f"faulty corridors f (C={C})")
    a2.set_ylabel("localization probes")
    a2.legend(fontsize=6)
    save(fig, "fig6_conservation.png")
    demo = conservation.simulate(faults={
        (0, 1): [("drop", 0), ("drop", 1)], (3, 4): [("replay", 2)],
        (7, 8): [("mint", 0)], (10, 11): [("misroute", 5)]})
    RESULTS["conservation"] = {
        "S": Ss, "cons_kb": cons, "roots_ring_kb": roots_ring,
        "roots_mesh_kb": roots_mesh,
        "mesh_crossover_S": conservation.crossover_shards_full_mesh(),
        "localization": {"C": C, "f": fs, "probes": probes},
        "demo": {k: demo[k] for k in ("total_txs", "in_flight_at_check",
                                      "faults_injected", "dangling_estimate",
                                      "truly_faulty", "alarms_exact",
                                      "localization_exact", "probes",
                                      "decode_exact")}}
    RESULTS["conservation"]["demo"]["truly_faulty"] = [
        list(k) for k in demo["truly_faulty"]]


# ===========================================================================
def fig7_scaling():
    print("F7: scaling")
    sizes = [100, 300, 1000, 3000]
    keys = ("flat", "tree", "hs", "hs2", "chained", "kauri", "pbft")
    msgs = {k: [] for k in keys}
    kb = {k: [] for k in keys}
    kb_thr = {k: [] for k in keys}
    lead = {k: [] for k in keys}
    flat_nosketch = []
    for Nv in sizes:
        nb = Nv - int(Nv * 0.7)
        sv, blk, po = build(Nv - nb, nb, seed=Nv)
        rows = {
            "flat": vector_consensus(sv, blk, None, po),
            "tree": tree_consensus(sv, blk, None, po, 10),
            "hs": hotstuff_consensus(Nv, nb, 20, seed=Nv),
            "hs2": hotstuff2_consensus(Nv, nb, 20, seed=Nv),
            "chained": chained_hotstuff_consensus(Nv, nb, 20, seed=Nv),
            "kauri": kauri_consensus(Nv, nb, 20, seed=Nv),
            "pbft": pbft_consensus(Nv, nb, 20),
        }
        thr = {
            "flat": vector_consensus(sv, blk, None, po, qc="threshold"),
            "tree": tree_consensus(sv, blk, None, po, 10, qc="threshold"),
            "hs": hotstuff_consensus(Nv, nb, 20, qc="threshold", seed=Nv),
            "hs2": hotstuff2_consensus(Nv, nb, 20, qc="threshold", seed=Nv),
            "chained": chained_hotstuff_consensus(Nv, nb, 20, qc="threshold", seed=Nv),
            "kauri": kauri_consensus(Nv, nb, 20, qc="threshold", seed=Nv),
            "pbft": rows["pbft"],
        }
        flat_nosketch.append(vector_consensus(sv, blk, None, po, observe=False)["msg_bytes"] / 1024)
        for k in keys:
            msgs[k].append(rows[k]["msgs"])
            kb[k].append(rows[k]["msg_bytes"] / 1024)
            kb_thr[k].append(thr[k]["msg_bytes"] / 1024)
            lead[k].append(rows[k]["leader_in_bytes"] / 1024)
        print(f"    N={Nv}: " + " ".join(f"{k}={msgs[k][-1]}" for k in keys))
    style = {"flat": ("o-", BLUE, "Proxima flat"), "tree": ("D-", GREEN, "Proxima tree"),
             "hs": ("s--", ORANGE, "HotStuff"), "hs2": ("v--", PURPLE, "HotStuff-2"),
             "chained": ("<-.", BROWN, "Chained HotStuff (amortized)"),
             "kauri": ("x--", GRAY, "Kauri (msg model)"), "pbft": ("^:", RED, "PBFT")}
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.4))
    for k, (fmt, c, lbl) in style.items():
        axes[0].loglog(sizes, msgs[k], fmt, color=c, ms=3.5, label=lbl)
        if k != "pbft":
            axes[1].loglog(sizes, kb_thr[k], fmt, color=c, ms=3.5, label=lbl)
            axes[2].loglog(sizes, lead[k], fmt, color=c, ms=3.5, label=lbl)
    axes[1].loglog(sizes, flat_nosketch, "o:", color=BLUE, ms=3, alpha=0.6,
                   label="Proxima flat, no sketches")
    axes[0].set_ylabel("messages per block")
    axes[1].set_ylabel("KB per block (threshold QCs)")
    axes[2].set_ylabel("KB into the leader")
    for a in axes:
        a.set_xlabel("validators N")
    axes[0].legend(fontsize=5.2, ncol=1)
    axes[1].legend(fontsize=5.2, ncol=1)
    save(fig, "fig7_scaling.png")
    RESULTS["scaling"] = {"sizes": sizes, "msgs": msgs,
                          "kb_bitmap": {k: [round(x) for x in v] for k, v in kb.items()},
                          "kb_threshold": {k: [round(x) for x in v] for k, v in kb_thr.items()},
                          "leader_in_kb": {k: [round(x, 1) for x in v] for k, v in lead.items()},
                          "flat_no_sketch_kb": [round(x) for x in flat_nosketch]}


# ===========================================================================
def fig8_fork_heal():
    print("F8: fork localization and healing")
    heights = [64, 256, 1024, 4096]
    probes = []
    for H in heights:
        A, B = dpmh.Accumulator(), dpmh.Accumulator()
        fork_at = int(H * 0.66)
        for h in range(H):
            blk = [f"b{h}-{i}" for i in range(4)]
            A.append_block(blk)
            B.append_block(blk if h < fork_at else [f"ALT{h}-{i}" for i in range(4)])
        r = dpmh.find_fork(A, B)
        assert r["fork_height"] == fork_at
        probes.append(r["probes"])
    ds = [2, 4, 8, 16, 32, 64, 128]
    heal_bytes, heal_est, heal_rounds = [], [], []
    base = sorted(f"t-{i}" for i in range(600))
    for d in ds:
        other = set(base[d // 2:]) | {f"p-{i}" for i in range(d - d // 2)}
        hp = heal_partition(set(base), other, salt=5, session=d)
        assert hp["recovered_union"]
        heal_bytes.append(hp["bytes"] + hp["push_bytes"])
        heal_est.append(hp["d_hat"])
        heal_rounds.append(hp["rounds"])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 2.3))
    a1.semilogx(heights, probes, "D-", color=GREEN, ms=4, base=2, label="probes")
    a1.semilogx(heights, [math.log2(h) + 1 for h in heights], "k--", lw=1.2, base=2,
                label=r"$\log_2 H + 1$")
    a1.set_xlabel("chain height H")
    a1.set_ylabel("fork-localization probes")
    a1.legend()
    a2.loglog(ds, heal_bytes, "D-", color=PURPLE, ms=4, base=2,
              label="sketch + fetch + push bytes")
    a2.loglog(ds, [d * 232 for d in ds], "k--", lw=1.2, base=2,
              label="232 B per divergent tx")
    a2.set_xlabel("partition divergence d")
    a2.set_ylabel("healing bytes")
    a2.legend()
    save(fig, "fig8_fork_heal.png")
    RESULTS["fork"] = {"H": heights, "probes": probes}
    RESULTS["heal"] = {"d": ds, "bytes": heal_bytes,
                       "d_hat": [round(x, 1) for x in heal_est],
                       "rounds": heal_rounds}


# ===========================================================================
def safety_table():
    print("safety fuzzing")
    rows = []
    trials = 10000
    for n in (4, 6, 7, 9, 10):            # n = 6, 9: fast quorum below n
        r = bft.fuzz(n, trials, seed=n)
        rows.append({"n": n, "f": r["f"], "n_fast": r["n_fast"], "trials": trials,
                     "unsafe": r["unsafe"], "not_live": r["not_live"],
                     "variant": "safe fast quorum"})
    for n in (4, 7, 10):
        r = bft.fuzz(n, trials, n_fast=n - bft.max_faults(n), seed=100 + n)
        rows.append({"n": n, "f": r["f"], "n_fast": r["n_fast"], "trials": trials,
                     "unsafe": r["unsafe"], "not_live": r["not_live"],
                     "variant": "v1 fast quorum n - f"})
    RESULTS["safety_fuzz"] = rows
    for row in rows:
        print("   ", row)


# ===========================================================================
if __name__ == "__main__":
    import time
    t0 = time.time()
    np.random.seed(0)
    fig1_estimator()
    fig2_threshold()
    fig3_deflation()
    fig4_fastpath()
    fig5_sync()
    fig6_conservation()
    fig7_scaling()
    fig8_fork_heal()
    safety_table()
    with open("paper_v2/results.json", "w") as fh:
        json.dump(RESULTS, fh, indent=1)
    print(f"\nDone in {time.time() - t0:.1f}s; results in paper_v2/results.json")
