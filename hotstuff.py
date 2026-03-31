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
