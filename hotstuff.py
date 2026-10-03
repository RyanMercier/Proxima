"""
hotstuff.py -- Baseline protocols, accounted exactly like Proxima.

Every protocol here and in blockchain.py shares one cost model:
- the proposal (64 B header + 32 B per tx id) reaches all N validators;
- a vote is 128 B (BLS signature + block hash / view metadata);
- a certificate is a 96 B aggregate + 64 B metadata, plus an N/8-byte
  signer bitmap when qc="bitmap" (multi-signatures) or nothing when
  qc="threshold" (threshold signatures, as the HotStuff paper assumes);
- a straggler fetches its missing transactions by id at 32 + 200 B each;
- Byzantine validators abstain (the worst case for message-count
  liveness), and every certificate is delivered to all N validators;
- leader_in_bytes counts bytes arriving at the leader, the per-node
  bottleneck of leader-based protocols.

Single-block costs (one view, honest leader) are compared alongside the
amortized per-block cost of chained HotStuff, which is what pipelined
deployments (DiemBFT / Jolteon / HotStuff-2 chained) actually pay.
"""

import math
import time

import numpy as np

from blockchain import MessageCounter, fetch_bytes, VOTE_BYTES, _cert_bytes


def _proposal_bytes(n_txs: int) -> int:
    return 64 + 32 * (n_txs + 1)


def _stragglers(msgs, n_honest: int, rate: float, max_miss: int, rng) -> int:
    k = 0
    for _ in range(int(round(n_honest * rate))):
        m = int(rng.integers(1, max_miss + 1))
        msgs.send("fetch_by_id", fetch_bytes(m))
        k += m
    return k


def _result(name, msgs, n, n_byz, rounds, t0, **extra):
    n_req = n - (n - 1) // 3
    return {"protocol": name, "finalized": n - n_byz >= n_req,
            "msgs": msgs.count, "msg_bytes": msgs.bytes,
            "leader_in_bytes": msgs.leader_in,
            "msg_breakdown": dict(msgs.by_type), "rounds": rounds,
            "total_time": time.time() - t0, "n_validators": n,
            "n_byzantine": n_byz, **extra}


def _linear_phases(name, n, n_byz, n_txs, rate, max_miss, phases, qc, seed):
    t0 = time.time()
    rng = np.random.default_rng(seed)
    msgs = MessageCounter()
    voters = n - n_byz
    msgs.send("propose", _proposal_bytes(n_txs), n)
    _stragglers(msgs, voters, rate, max_miss, rng)
    cert = _cert_bytes(n, qc)
    for p in phases:
        msgs.send(f"{p}_vote", VOTE_BYTES, voters, to_leader=True)
        msgs.send(f"{p}_cert", cert, n)
    return _result(name, msgs, n, n_byz, len(phases), t0)


def hotstuff_consensus(n_validators, n_byzantine, n_txs=20,
                       partial_obs_rate=0.37, max_miss=2, qc="bitmap", seed=0):
    """Basic HotStuff: prepare, pre-commit, commit votes; the commit
    certificate is the decide message."""
    return _linear_phases("HotStuff", n_validators, n_byzantine, n_txs,
                          partial_obs_rate, max_miss,
                          ["prepare", "precommit", "commit"], qc, seed)


def hotstuff2_consensus(n_validators, n_byzantine, n_txs=20,
                        partial_obs_rate=0.37, max_miss=2, qc="bitmap", seed=0):
    """HotStuff-2 (Malkhi, Nayak 2023): two voting phases. Same phase
    structure as Proxima's slow path, which is built on it."""
    return _linear_phases("HotStuff-2", n_validators, n_byzantine, n_txs,
                          partial_obs_rate, max_miss, ["prepare", "commit"],
                          qc, seed)


def chained_hotstuff_consensus(n_validators, n_byzantine, n_txs=20,
                               partial_obs_rate=0.37, max_miss=2,
                               qc="bitmap", seed=0):
    """Chained HotStuff, amortized per block in steady state.

    Each view the leader broadcasts one proposal carrying the previous QC
    and collects one round of votes; a block commits after its successors'
    QCs, so the per-block cost is one proposal broadcast plus one vote
    round. Latency is still several rounds per block; only cost amortizes.
    """
    t0 = time.time()
    rng = np.random.default_rng(seed)
    msgs = MessageCounter()
    voters = n_validators - n_byzantine
    msgs.send("propose_with_qc",
              _proposal_bytes(n_txs) + _cert_bytes(n_validators, qc),
              n_validators)
    _stragglers(msgs, voters, partial_obs_rate, max_miss, rng)
    msgs.send("vote", VOTE_BYTES, voters, to_leader=True)
    return _result("Chained HotStuff", msgs, n_validators, n_byzantine, 1, t0,
                   amortized=True)


def kauri_consensus(n_validators, n_byzantine, n_txs=20, partial_obs_rate=0.37,
                    fanout=10, max_miss=2, qc="bitmap", seed=0):
    """Kauri (Neiheiser, Matos, Rodrigues, SOSP 2021): HotStuff phases
    disseminated and aggregated over a fanout-m tree.

    Each phase sends the proposal or certificate down every tree edge and
    aggregates votes up (internal nodes forward one aggregate). Not
    charged: tree reconfiguration after relay failures, and the pipelining
    that is Kauri's real throughput win; a faithful comparison needs
    Kauri's own harness.
    """
    t0 = time.time()
    rng = np.random.default_rng(seed)
    msgs = MessageCounter()
    n = n_validators
    voters = n - n_byzantine
    _stragglers(msgs, voters, partial_obs_rate, max_miss, rng)
    cert = _cert_bytes(n, qc)
    agg = 96 + (math.ceil(n / 8) if qc == "bitmap" else 0)
    # Internal relays below the root, and how many of them feed the root.
    levels, remaining = [], n
    while remaining > fanout:
        remaining = math.ceil(remaining / fanout)
        levels.append(remaining)
    internal = sum(levels)
    into_root = levels[-1] if levels else 0
    msgs.send("propose_down", _proposal_bytes(n_txs), n - 1)
    for i, p in enumerate(["prepare", "precommit", "commit"]):
        if i:
            msgs.send(f"{p}_cert_down", cert, n - 1)
        msgs.send(f"{p}_vote_up", VOTE_BYTES, n - internal - 1,
                  to_leader=not levels)
        msgs.send(f"{p}_agg_up", agg, internal - into_root)
        msgs.send(f"{p}_agg_into_root", agg, into_root, to_leader=True)
    msgs.send("decide_down", cert, n - 1)
    return _result("Kauri", msgs, n, n_byzantine, 3, t0, fanout=fanout)


def pbft_consensus(n_validators, n_byzantine, n_txs=20):
    """Classic PBFT message counts (all-to-all), for scale context only."""
    t0 = time.time()
    msgs = MessageCounter()
    n = n_validators
    msgs.send("pre_prepare", _proposal_bytes(n_txs), n)
    msgs.send("prepare", 128, n * (n - 1))
    msgs.send("commit", 128, n * (n - 1))
    return _result("PBFT", msgs, n, n_byzantine, 3, t0)
