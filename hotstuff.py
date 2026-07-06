"""
hotstuff.py -- HotStuff BFT simulation for fair comparison.

Leader-based, BLS aggregate sigs, O(N) per phase, 3 phases always.
Uses the same MessageCounter as our protocol so counts are comparable.
Block proposal is shared overhead so we only count consensus messages.
"""

import math
import time
import numpy as np
from blockchain import MessageCounter


def hotstuff_consensus(n_validators: int, n_byzantine: int,
                       n_txs: int = 20, partial_obs_rate: float = 0.37,
                       max_miss: int = 2) -> dict:
    """
    Simulate HotStuff for n_validators with n_byzantine adversaries.

    Three phases: prepare, pre-commit, commit.
    Leader aggregates BLS signatures at each phase.
    Validators with incomplete state need retransmission before they can vote.
    """
    t0 = time.time()
    msgs = MessageCounter()
    n = n_validators
    n_honest = n - n_byzantine

    # Retransmission round: validators with missing txs request them from leader.
    # Request-response: 2 messages per incomplete validator.
    n_partial = int(n_honest * partial_obs_rate)
    for _ in range(n_partial):
        n_miss = np.random.randint(1, max_miss + 1)
        msgs.send("retransmit_request", 64)
        msgs.send("retransmit_response", n_miss * 250)

    # BLS aggregate sig (96) + signer bitmap (N/8) + block hash (32) + round metadata (32)
    agg_cert_size = 96 + math.ceil(n / 8) + 64

    # 3 identical voting rounds: prepare, pre-commit, commit
    # Each round: N validators send vote to leader (96 bytes BLS sig + 32 block hash)
    #             Leader broadcasts aggregate cert to N validators
    for phase in ["prepare", "precommit", "commit"]:
        for _ in range(n):
            msgs.send(f"{phase}_vote", 128)
        for _ in range(n):
            msgs.send(f"{phase}_cert", agg_cert_size)

    total_time = time.time() - t0
    n_req = int(math.ceil(n * 2 / 3))
    finalized = n_honest >= n_req

    return {
        "finalized": finalized,
        "msgs": msgs.count,
        "msg_bytes": msgs.bytes,
        "msg_breakdown": dict(msgs.by_type),
        "rounds": 3,
        "retransmit_msgs": n_partial * 2,
        "total_time": total_time,
        "n_validators": n,
        "n_byzantine": n_byzantine,
    }


def hotstuff2_consensus(n_validators: int, n_byzantine: int,
                        n_txs: int = 20, partial_obs_rate: float = 0.37,
                        max_miss: int = 2) -> dict:
    """
    HotStuff-2 (Malkhi, Nayak 2023): two voting phases instead of three,
    keeping linear message complexity and optimistic responsiveness.

    The modern two-phase baseline the v2 evaluation compares against:
    Proxima's "two phases vs three" advantage halves against this variant,
    so it must appear in the message/latency tables. Same MessageCounter
    and byte constants as the other protocols.
    """
    t0 = time.time()
    msgs = MessageCounter()
    n = n_validators
    n_honest = n - n_byzantine

    # Retransmission for incomplete validators, as in hotstuff_consensus.
    n_partial = int(n_honest * partial_obs_rate)
    for _ in range(n_partial):
        n_miss = np.random.randint(1, max_miss + 1)
        msgs.send("retransmit_request", 64)
        msgs.send("retransmit_response", n_miss * 250)

    agg_cert_size = 96 + math.ceil(n / 8) + 64

    # Two voting phases (prepare, commit), then the decide broadcast.
    for phase in ["prepare", "commit"]:
        for _ in range(n):
            msgs.send(f"{phase}_vote", 128)
        for _ in range(n):
            msgs.send(f"{phase}_cert", agg_cert_size)
    for _ in range(n):
        msgs.send("decide", agg_cert_size)

    total_time = time.time() - t0
    n_req = int(math.ceil(n * 2 / 3))
    return {
        "finalized": n_honest >= n_req,
        "msgs": msgs.count,
        "msg_bytes": msgs.bytes,
        "msg_breakdown": dict(msgs.by_type),
        "rounds": 2,
        "retransmit_msgs": n_partial * 2,
        "total_time": total_time,
        "n_validators": n,
        "n_byzantine": n_byzantine,
    }


def kauri_consensus(n_validators: int, n_byzantine: int,
                    n_txs: int = 20, partial_obs_rate: float = 0.37,
                    fanout: int = 10, max_miss: int = 2) -> dict:
    """
    Kauri (Neiheiser, Matos, Rodrigues, SOSP 2021): HotStuff with pipelined
    tree-based dissemination and aggregation.

    Model: a fanout-m tree over all N validators. Each HotStuff phase costs
    one dissemination pass down the tree (N - 1 messages) and one
    aggregation pass up (N - 1 messages, votes combined at internal nodes),
    so raw message count stays close to flat HotStuff; Kauri's wins are
    load distribution and latency via pipelining, which this counter-level
    model does not credit. What it also does not charge for: Kauri needs
    tree reconfiguration when internal relays fail, and its internal nodes
    are trusted for aggregation availability, costs outside this happy-path
    count (documented so the comparison stays honest). Byte sizes match the
    other protocols: 128 B votes, 96 B + N/8 aggregates at internal nodes.
    """
    t0 = time.time()
    msgs = MessageCounter()
    n = n_validators
    n_honest = n - n_byzantine

    n_partial = int(n_honest * partial_obs_rate)
    for _ in range(n_partial):
        n_miss = np.random.randint(1, max_miss + 1)
        msgs.send("retransmit_request", 64)
        msgs.send("retransmit_response", n_miss * 250)

    agg_size = 96 + math.ceil(n / 8)

    # Internal (non-leaf) node count of a fanout-m tree over n nodes.
    internal = 0
    level = 1
    remaining = n
    while remaining > fanout:
        parents = math.ceil(remaining / fanout)
        internal += parents
        remaining = parents
        level += 1

    for phase in ["prepare", "precommit", "commit"]:
        # Dissemination down the tree: every edge carries the proposal/cert.
        msgs.send(f"{phase}_disseminate", 128, n - 1)
        # Aggregation up: leaves send votes, internal nodes send aggregates.
        msgs.send(f"{phase}_vote_up", 128, n - internal - 1)
        msgs.send(f"{phase}_agg_up", agg_size, internal)

    # Decide broadcast down the tree.
    msgs.send("decide", agg_size, n - 1)

    total_time = time.time() - t0
    n_req = int(math.ceil(n * 2 / 3))
    return {
        "finalized": n_honest >= n_req,
        "msgs": msgs.count,
        "msg_bytes": msgs.bytes,
        "msg_breakdown": dict(msgs.by_type),
        "rounds": 3,
        "fanout": fanout,
        "retransmit_msgs": n_partial * 2,
        "total_time": total_time,
        "n_validators": n,
        "n_byzantine": n_byzantine,
    }


def pbft_consensus(n_validators: int, n_byzantine: int, n_txs: int = 20) -> dict:
    """
    Classic PBFT message counts for reference. O(N^2) all-to-all voting.
    """
    msgs = MessageCounter()
    n = n_validators
    n_honest = n - n_byzantine

    # Pre-prepare: leader to all (N messages)
    for _ in range(n):
        msgs.send("pre_prepare", 128)

    # Prepare: each validator broadcasts to all others. N * (N-1)
    msgs.send("prepare", 128, n * (n - 1))

    # Commit: each validator broadcasts to all others. N * (N-1)
    msgs.send("commit", 128, n * (n - 1))

    return {
        "finalized": True,
        "msgs": msgs.count,
        "msg_bytes": msgs.bytes,
        "msg_breakdown": dict(msgs.by_type),
        "rounds": 3,
        "n_validators": n,
        "n_byzantine": n_byzantine,
    }
