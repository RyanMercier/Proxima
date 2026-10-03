#!/usr/bin/env python3
"""
test_v2.py -- Executable checks for the Proxima v2 primitive and protocol.

Every claim the manuscript makes about behaviour is checked here: sketch
test vectors and exact tails, LtHash binding and accumulators, the
deflation attack, IBLT reconciliation, safety of the base protocol under
adversarial fuzzing (and the counterexample for the unsafe fast quorum),
consensus accounting with equivocation and Byzantine leaf leaders, real
BLS at small N, sensing against a fair baseline, partition healing, and
cross-shard conservation including misroutes and honest lag.
Run: python test_v2.py
"""

import math
import random

import numpy as np

np.random.seed(7)

import blockchain as bc
bc.USE_REAL_BLS = False

import bft
import conservation
import dpmh
import reconcile
from blockchain import (
    BLSKeyPair, Blockchain, make_validators, make_partial_obs,
    validate_threshold, vector_consensus, tree_consensus, readiness_sense,
    fixed_delay_propose, heal_partition, split_bimodal,
)

PASS = FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok   {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name}  {detail}")


def build(n_h, n_b, strategy="abstain", n_txs=20, miss=0.37, seed=0):
    sv, sh, _sb = make_validators(n_h, n_b, strategy, seed=seed)
    sc = Blockchain(sv)
    for i in range(5):
        sc.register_account(f"S-{i}", 100000)
    for i in range(n_txs):
        sc.submit_tx(sc.make_tx(f"S-{i % 5}", f"S-{(i + 1) % 5}", 1.0))
    blk = sc.propose_block(sh[0])
    po = make_partial_obs(sv, len(blk.tx_data_strings), miss_prob=miss,
                          rng=np.random.default_rng(seed))
    return sc, sv, blk, po


# ---------------------------------------------------------------------------
print("[1] sketch test vectors and estimator")

n = dpmh.N_ROUND
salt = dpmh.round_salt(5, "ab" * 32)
txs = [f"tx-{i}" for i in range(30)]
D = dpmh.digest(txs, salt)
check("d=1: dist2 == n exactly",
      all(dpmh.dist2(D, dpmh.digest(txs[:i] + txs[i + 1:], salt)) == n
          for i in range(5)))
d2 = np.array([dpmh.dist2(dpmh.digest([f"s{t}-{x}" for x in txs], t),
                          dpmh.digest([f"s{t}-{x}" for x in txs[:-2]], t))
               for t in range(200)])
check("d=2: mean 2n within 5%", abs(d2.mean() - 2 * n) < 0.1 * n, str(d2.mean()))
check("d=2: values are 4 Bin(n,1/2)", bool(np.all(d2 % 4 == 0)))
check("d=2: variance 4n within 25%", abs(d2.var() - 4 * n) < n, str(d2.var()))
est = dpmh.est_symdiff(D, dpmh.digest(txs[:-3] + ["A", "B", "C"], salt))
check("substituting 3 reads as symmetric difference 6", abs(est - 6) < 1.5, str(est))
v = dpmh.validate_estimator(d=3, trials=300)
check("unbiased at d=3", abs(v["mean"] - 3) < 0.15, str(v))
mult = np.mean([dpmh.est_symdiff(dpmh.digest([f"m{t}a", f"m{t}a", f"m{t}b", f"m{t}b"], t),
                                 dpmh.digest([], t)) for t in range(300)])
check("multiset remark: c=(2,2) estimates ||c||_2^2 = 8, not 4",
      abs(mult - 8) < 0.6, str(mult))
check("homomorphism", bool(np.array_equal(
    dpmh.digest(txs[:10], salt) + dpmh.digest(txs[10:], salt), D)))
w = dpmh.to_wire(D, "round")
check("wire roundtrip", bool(np.array_equal(dpmh.from_wire(w, "round"), D)))
big_a = np.full(n, 40000, dtype=np.int64)
big_b = big_a - 3
wa = dpmh.from_wire(dpmh.to_wire(big_a, "round"), "round")
wb = dpmh.from_wire(dpmh.to_wire(big_b, "round"), "round")
check("centered_diff recovers differences after wraparound",
      int(np.sum(dpmh.centered_diff(wa, wb, 2 ** 16))) == 3 * n)
check("salt changes every vector",
      not np.array_equal(dpmh.tx_vector("x", dpmh.round_salt(1, "00" * 32)),
                         dpmh.tx_vector("x", dpmh.round_salt(1, "01" + "00" * 31))))
try:
    dpmh.assert_set(["a", "b", "a"])
    check("duplicate items rejected (MU_MAX = 1)", False)
except ValueError:
    check("duplicate items rejected (MU_MAX = 1)", True)

# ---------------------------------------------------------------------------
print("[2] threshold")

val = validate_threshold(txs, trials=500)
check("no honest d<=2 view exceeds 3n in 500 trials", val["exceed"] == 0, str(val))

# ---------------------------------------------------------------------------
print("[3] LtHash ledger digests and accumulators")

L1 = dpmh.LedgerDigest.of(["a", "b", "c"])
L2 = dpmh.LedgerDigest.of(["a", "b"]) + dpmh.LedgerDigest.item("c")
check("LtHash homomorphism", L1 == L2)
check("LtHash distinguishes sets", not (L1 == dpmh.LedgerDigest.of(["a", "b", "d"])))
check("LtHash lanes are uniform mod 2^16 (no distance leakage)",
      0.45 < np.mean(dpmh.lthash_vector("z") > 2 ** 15) < 0.55)
check("ledger sketch estimates difference", abs(L1.est(dpmh.LedgerDigest.of(["a"])) - 2) < 0.6)
A, B = dpmh.Accumulator(), dpmh.Accumulator()
blocks = [[f"b{h}-t{i}" for i in range(6)] for h in range(32)]
for h, blk in enumerate(blocks):
    A.append_block(blk)
    B.append_block(blk if h < 21 else [f"ALT{h}-{i}" for i in range(6)])
r = dpmh.find_fork(A, B)
check("fork found at 21", r["fork_height"] == 21, str(r))
check("probes <= log2 H + 1", r["probes"] <= math.ceil(math.log2(32)) + 1, str(r["probes"]))
check("range digest == digest of range",
      A.range_digest(4, 9) == dpmh.LedgerDigest.of([t for b in blocks[4:9] for t in b]))
pruned = [t for b in blocks[:3] for t in b]
after = A.head() - dpmh.LedgerDigest.of(pruned)
check("prune accounting verifies", dpmh.prove_prune(A.head(), after, pruned))
check("prune accounting rejects a lie", not dpmh.prove_prune(A.head(), after, pruned[:-1]))

# ---------------------------------------------------------------------------
print("[4] adversarial deflation (why the sketch is advisory)")

honest = dpmh.est_symdiff(dpmh.digest([f"h-{i}" for i in range(200)], 1),
                         np.zeros(n, dtype=np.int64))
side_only = dpmh.adversarial_balance(m=200, pool=1, seed=1)
greedy = dpmh.adversarial_balance(m=200, pool=256, seed=1)
check("honest 200-item difference reads near 200", abs(honest - 200) < 40, str(honest))
check("choosing sides alone (pool 1) already halves d_hat",
      side_only["d_hat"] < 0.7 * 200, str(side_only))
check("greedy balancing deflates a 200-item difference below 30",
      greedy["d_hat"] < 30, str(greedy))

# ---------------------------------------------------------------------------
print("[5] IBLT reconciliation")

base = set(f"item-{i}" for i in range(500))
for d_true in (1, 3, 8, 25):
    other = set(sorted(base)[d_true:]) | {f"new-{i}" for i in range(d_true)}
    res = reconcile.reconcile(base, other, d_hint=2 * d_true)
    check(f"exact decode at d={2 * d_true}, fetch counted",
          res["ok"] and len(res["only_a"]) == d_true and len(res["only_b"]) == d_true
          and res["fetch_bytes"] == d_true * (reconcile.KEY_BYTES + reconcile.ITEM_BYTES),
          str({k: res[k] for k in ("ok", "rounds", "bytes")}))
low = reconcile.reconcile(base, set(sorted(base)[40:]), d_hint=2)
check("undersized hint recovers by doubling", low["ok"] and low["rounds"] > 1)
check("short ids differ across sessions",
      reconcile._key64("tx", 1) != reconcile._key64("tx", 2))

# ---------------------------------------------------------------------------
print("[6] base protocol safety (bft.py)")

for nn in (4, 7, 10):
    fz = bft.fuzz(nn, 600, seed=nn)
    check(f"n={nn}: safe and live under 600 adversarial schedules",
          fz["unsafe"] == 0 and fz["not_live"] == 0, str(fz))
fz = bft.fuzz(6, 600, seed=3, n_byz=1)
check("n=6, f=1: fast quorum 5 < n is safe", fz["unsafe"] == 0 and fz["n_fast"] == 5, str(fz))
bad = bft.fuzz(4, 600, n_fast=3, seed=4)
check("v1-style fast quorum 2f+1 is unsafe (fuzzer finds conflicts)", bad["unsafe"] > 0)
ce = bft.unsafe_fast_path_counterexample(2)
check("explicit counterexample for the 2f+1 fast path", ce["conflict"]
      and ce["view0_fast_decided_under_safe_quorum"] is None, str(ce))

# ---------------------------------------------------------------------------
print("[7] consensus accounting on a chain")

sc, sv, blk, po = build(70, 30)
r = vector_consensus(sv, blk, None, po)
check("flat finalizes at 30% abstaining Byzantine", r["finalized"] and r["rounds"] == 2)
check("no fast path at n=3f+1 with abstainers", not r["fast_path"])
check("stragglers fetch by id and rejoin",
      r["n_commits"] == 70 and r["sync_pushed"] > 0)
check("all honest validators receive the certificate",
      r["msg_breakdown"].get("commit_qc") == 100)

sc, sv, blk, _ = build(95, 5, "mimic_honest", miss=0.0)
r = vector_consensus(sv, blk, None, {}, f=20)
check("fast path at n_fast = 81 with lying-sketch Byzantine signing",
      r["fast_path"] and r["n_fast"] == 81)
sc, sv, blk, _ = build(70, 30, "equivocate", miss=0.0)
r = vector_consensus(sv, blk, None, {})
check("equivocating signatures are rejected at verification",
      r["finalized"] and r["n_commits"] == 70)

sc, sv, blk, po = build(70, 30, seed=3)
rt = tree_consensus(sv, blk, None, po, 10)
check("Byzantine leaf leaders occur under random placement", rt["fallback_leaves"] > 0)
check("tree finalizes through the fallback path", rt["finalized"])

bc.USE_REAL_BLS = True
if bc.BLS_AVAILABLE:
    sc, sv, blk, _ = build(3, 1, "equivocate", miss=0.0, n_txs=3)
    r = vector_consensus(sv, blk, None, {})
    check("real BLS12-381: aggregate verifies, equivocation rejected",
          r["finalized"] and r["n_commits"] == 3)
bc.USE_REAL_BLS = False

# ---------------------------------------------------------------------------
print("[8] readiness sensing vs a fixed delay of the same mean")

sens_fast = fixed_fast = 0
delays = []
trials = 20
for t in range(trials):
    sc, sv, blk, po = build(100, 0, miss=0.6, seed=100 + t)
    rs = readiness_sense(sv, blk, po, fill_prob=0.5, max_ticks=8, f=20,
                         rng=np.random.default_rng(t))
    sens_fast += rs["fast_path"]
    delays.append(rs["sense_ticks"])
mean_delay = int(round(np.mean(delays)))
for t in range(trials):
    sc, sv, blk, po = build(100, 0, miss=0.6, seed=100 + t)
    rf = fixed_delay_propose(sv, blk, po, mean_delay, fill_prob=0.5, f=20,
                             rng=np.random.default_rng(t))
    fixed_fast += rf["fast_path"]
check("sensing at least matches a fixed delay of equal mean",
      sens_fast >= fixed_fast, f"sensed {sens_fast} fixed {fixed_fast} delay {mean_delay}")
check("sensing gossip bytes are charged", rs["sensing_bytes"] > 0)

# ---------------------------------------------------------------------------
print("[9] partition healing")

set_a = set(f"t-{i}" for i in range(80))
set_b = (set_a - {f"t-{i}" for i in range(4)}) | {"p1", "p2", "p3"}
hp = heal_partition(set_a, set_b, salt=9)
check("both camps reach the union", hp["recovered_union"], str(hp))
check("d_hat near true divergence", abs(hp["d_hat"] - hp["true_d"]) < 0.35 * hp["true_d"] + 1)
digs = {i: dpmh.digest(sorted(set_a), 9) for i in range(6)}
digs.update({10 + i: dpmh.digest(sorted(set_b), 9) for i in range(5)})
ca, cb = split_bimodal(digs)
check("bimodal split separates the camps",
      {frozenset(ca), frozenset(cb)} == {frozenset(range(6)), frozenset(range(10, 15))})

# ---------------------------------------------------------------------------
print("[10] cross-shard conservation")

faults = {(0, 1): [("drop", 0), ("drop", 1)], (3, 4): [("replay", 2)],
          (7, 8): [("mint", 0)], (10, 11): [("misroute", 5)]}
r = conservation.simulate(faults=faults)
check("honest lag leaves transactions in flight at check time",
      r["in_flight_at_check"] > 0)
check("invariant violated", r["invariant_violated"])
check("alarms name exactly the faulty corridors (misroute hits both ends)",
      r["alarms_exact"] and (11, 12) in r["aged_alarms"], str(r["aged_alarms"]))
check("bisection localizes exactly", r["localization_exact"])
check("decode recovers drops, replay, mint, misroute exactly", r["decode_exact"])
clean = conservation.simulate(faults={}, lag=1)
check("no faults + lag: invariant holds, no alarms",
      not clean["invariant_violated"] and not clean["aged_alarms"])
mesh = conservation.simulate(n_shards=6, topology="mesh", faults={(1, 4): [("drop", 3)]})
check("full mesh: exact detection and localization",
      mesh["localization_exact"] and mesh["decode_exact"])
check("crossover vs per-corridor roots on a full mesh is reported",
      conservation.crossover_shards_full_mesh() > 2)

# ---------------------------------------------------------------------------
print()
print(f"{PASS} passed, {FAIL} failed")
raise SystemExit(1 if FAIL else 0)
