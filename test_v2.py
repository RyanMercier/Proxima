#!/usr/bin/env python3
"""
test_v2.py -- Executable checks for the Proxima v2 primitive and protocol.

Covers the implementation test vectors from docs/v2_formal_foundations.md
(Sections 2 and 11) plus end-to-end checks of every v2 protocol feature:
robust reference, honest-sufficient fast path (griefing fix), IBLT sync,
readiness sensing, fork localization, partition healing, and cross-shard
conservation. Run: python test_v2.py
"""

import math

import numpy as np

np.random.seed(7)

import blockchain as bc
bc.USE_REAL_BLS = False

import dpmh
import reconcile
import conservation
from blockchain import (
    BLSKeyPair, Blockchain, Validator, make_validators, make_partial_obs,
    calibrate_threshold, validate_threshold, vector_consensus, tree_consensus,
    readiness_sense, heal_partition, split_bimodal,
)

PASS = 0
FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok   {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name}  {detail}")


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


# ---------------------------------------------------------------------------
print("[1] dpmh test vectors (v2_formal_foundations Sections 2, 11)")

n = dpmh.N_ROUND
txs = [f"tx-{i}" for i in range(30)]
D = dpmh.digest(txs, height=5)

check("d=1: dist2 == n exactly, zero variance",
      all(dpmh.dist2(D, dpmh.digest(txs[:i] + txs[i + 1:], height=5)) == n
          for i in range(5)))

d2_samples = []
for t in range(200):
    salted = [f"s{t}-{x}" for x in txs]
    full = dpmh.digest(salted, height=t)
    less2 = dpmh.digest(salted[:-2], height=t)
    d2_samples.append(dpmh.dist2(full, less2))
d2_samples = np.array(d2_samples)
check("d=2: mean matches 2n within 5%",
      abs(d2_samples.mean() - 2 * n) < 0.05 * 2 * n,
      f"mean={d2_samples.mean():.1f}")
check("d=2: values are 4*Bin(n,1/2) (all divisible by 4)",
      bool(np.all(d2_samples % 4 == 0)))
check("d=2: variance matches 4n within 25%",
      abs(d2_samples.var() - 4 * n) < n,
      f"var={d2_samples.var():.1f} expect {4 * n}")

sub = txs[:-3] + ["A", "B", "C"]     # 3 substitutions = symdiff 6
est = dpmh.est_symdiff(D, dpmh.digest(sub, height=5))
check("substitution of j reads as 2j (v1 blind spot closed)",
      abs(est - 6) < 1.5, f"est={est:.2f}")

v = dpmh.validate_estimator(d=3, trials=300)
check("estimator unbiased at d=3 (within 5%)",
      abs(v["mean"] - 3) < 0.15, str(v))

check("homomorphism D(A)+D(B) == D(A u B)",
      bool(np.array_equal(dpmh.digest(txs[:10], 5) + dpmh.digest(txs[10:], 5), D)))
w = dpmh.to_wire(D, "round")
check("wire roundtrip lossless within block bound",
      bool(np.array_equal(dpmh.from_wire(w, "round"), D))
      and len(w) == dpmh.wire_bytes("round"))

# ---------------------------------------------------------------------------
print("[2] closed-form threshold replaces Monte Carlo")

tau = calibrate_threshold()
check("tau^2 == 3n", abs(tau * tau - 3 * n) < 1e-6)
val = validate_threshold(txs, trials=500)
check("honest exclusion rate is 0 in 500 trials (bound exp(-n/8))",
      val["exceed"] == 0, str(val))

# ---------------------------------------------------------------------------
print("[3] accumulators: range, fork localization, prune accounting")

A, B = dpmh.Accumulator(), dpmh.Accumulator()
blocks = [[f"b{h}-t{i}" for i in range(6)] for h in range(32)]
for h, blk in enumerate(blocks):
    A.append_block(blk)
    B.append_block(blk if h < 21 else [f"ALT{h}-{i}" for i in range(6)])
r = dpmh.find_fork(A, B)
check("fork found at 21", r["fork_height"] == 21, str(r))
check("probes are O(log H)", r["probes"] <= math.ceil(math.log2(32)) + 1,
      f"probes={r['probes']}")
check("probe trace carries divergence estimates", len(r["trace"]) > 0)

rng_d = A.range_digest(4, 9)
manual = dpmh.ledger_digest([t for blk in blocks[4:9] for t in blk])
check("range digest == digest of range, O(1)",
      dpmh.wire_equal(rng_d, manual, "ledger"))

pruned = [t for blk in blocks[:3] for t in blk]
after = A.head() - dpmh.ledger_digest(pruned)
check("prune accounting verifies", dpmh.prove_prune(A.head(), after, pruned))
check("prune accounting rejects a lie",
      not dpmh.prove_prune(A.head(), after, pruned[:-1]))

# ---------------------------------------------------------------------------
print("[4] IBLT reconciliation")

base = set(f"item-{i}" for i in range(500))
for d_true in (1, 3, 8, 25):
    other = set(list(base)[d_true:]) | {f"new-{i}" for i in range(d_true)}
    res = reconcile.reconcile(base, other, d_hint=d_true)
    check(f"exact decode at d={2 * d_true}",
          res["ok"] and len(res["only_a"]) == d_true
          and len(res["only_b"]) == d_true,
          str({k: res[k] for k in ('ok', 'rounds', 'bytes')}))

low_hint = reconcile.reconcile(base, set(list(base)[40:]), d_hint=2)
check("undersized hint recovers by doubling",
      low_hint["ok"] and low_hint["rounds"] > 1, str(low_hint["rounds"]))

# ---------------------------------------------------------------------------
print("[5] v2 flat consensus: robust reference, exclusion, fast path")

sc, sv, blk, tau, po = build(70, 30)
r = vector_consensus(sv, blk, tau, po)
check("finalizes at 30% Byzantine", r["finalized"])
check("all Byzantine excluded (drop_half)",
      all(is_b for (_nm, is_b, _s, _d) in r["excluded"]),
      str(r["excluded"][:3]))
check("no honest excluded", len(r["excluded"]) == 30)
check("slow path when 37% miss", not r["fast_path"])

sc2, sv2, blk2, tau2v, _ = build(70, 0, miss=0.0)
r2 = vector_consensus(sv2, blk2, tau2v, {})
check("fast path engages with complete views", r2["fast_path"])

# griefing fix: one in-cluster Byzantine (mimic) cannot block the trigger
sc3, sv3, blk3, tau3, _ = build(70, 1, strategy="mimic_honest", miss=0.0)
r3 = vector_consensus(sv3, blk3, tau3, {})
check("fast path survives in-cluster Byzantine (griefing fix)",
      r3["fast_path"] and r3["finalized"])

# robust reference: byzantine digests cannot move the median off the honest
# cluster, so honest validators keep distance ~0
sc4, sv4, blk4, tau4, _ = build(60, 29, strategy="random_vector", miss=0.0)
r4 = vector_consensus(sv4, blk4, tau4, {})
honest_far = [d for (_nm, is_b, _s, d) in r4["excluded"] if not is_b]
check("robust reference: zero honest exclusions at 33% random-vector Byz",
      not honest_far and r4["finalized"])

# ---------------------------------------------------------------------------
print("[6] tree consensus")

rt = tree_consensus(sv, blk, tau, po, 10)
check("tree finalizes at 30% Byzantine", rt["finalized"])
check("tree summaries use exact-sum size",
      rt["msg_breakdown"].get("L0_summary", 0) > 0)

# ---------------------------------------------------------------------------
print("[7] readiness sensing")

trials = 30
fast_fixed = fast_sensed = 0
for t in range(trials):
    scx, svx, blkx, taux, pox = build(40, 10, miss=0.6)
    rf = vector_consensus(svx, blkx, taux, pox)
    fast_fixed += rf["fast_path"]
    scy, svy, blky, tauy, poy = build(40, 10, miss=0.6)
    rs = readiness_sense(svy, blky, tauy, poy, fill_prob=0.5, max_ticks=10)
    fast_sensed += rs["fast_path"]
check("sensing lifts fast-path rate at 60% miss",
      fast_sensed > fast_fixed,
      f"sensed {fast_sensed}/{trials} vs fixed {fast_fixed}/{trials}")

# ---------------------------------------------------------------------------
print("[8] partition healing")

set_a = set(f"t-{i}" for i in range(80))
set_b = (set_a - {f"t-{i}" for i in range(4)}) | {"p1", "p2", "p3"}
hp = heal_partition(set_a, set_b, height=9)
check("partition healed exactly", hp["recovered_union"], str(hp))
check("d_hat close to true divergence",
      abs(hp["d_hat"] - hp["true_d"]) < 0.35 * hp["true_d"] + 1,
      f"d_hat={hp['d_hat']:.1f} true={hp['true_d']}")

digs = {i: dpmh.digest(sorted(set_a), 9) for i in range(6)}
digs.update({10 + i: dpmh.digest(sorted(set_b), 9) for i in range(5)})
ca, cbb = split_bimodal(digs)
check("bimodal split separates the camps",
      {frozenset(ca), frozenset(cbb)} ==
      {frozenset(range(6)), frozenset(range(10, 15))})

# ---------------------------------------------------------------------------
print("[9] cross-shard conservation")

r9 = conservation.simulate()
check("faults detected via global invariant",
      r9["invariant_violated"] and r9["detected"])
check("dangling estimate within 25% of truth",
      abs(r9["dangling_estimate"] - r9["faults_injected"])
      <= max(2, 0.25 * r9["faults_injected"]),
      f"est={r9['dangling_estimate']:.1f} true={r9['faults_injected']}")
check("aged alarms name exactly the faulty corridors",
      r9["aged_alarms"] == r9["truly_faulty"])
check("bisection localization exact", r9["localization_exact"])
check("localization probes near O(f log C) bound",
      r9["probes"] <= 2 * r9["probe_bound"],
      f"probes={r9['probes']} bound={r9['probe_bound']}")
check("corridor decode recovers exact danglers (drops + mints)",
      r9["decode_ok"] and r9["decode_exact"])

clean = conservation.simulate(n_faulty_corridors=0, faults_per_corridor=0)
check("no faults -> invariant holds, no alarms",
      not clean["invariant_violated"] and not clean["aged_alarms"])

# ---------------------------------------------------------------------------
print()
print(f"{PASS} passed, {FAIL} failed")
raise SystemExit(1 if FAIL else 0)
