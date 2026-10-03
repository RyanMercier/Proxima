"""
blockchain.py -- Ledger, validators, and consensus accounting (v2).

Consensus is the base protocol in bft.py (two voting phases with locking
and view change) plus Proxima's augmentations; this module accounts its
messages and bytes on a concrete chain and hosts the observability tools:

- Sketches (dpmh.py) on prepare votes report who is behind and by how
  much against D(B), the sketch of the proposal. They never gate votes.
- Speculative fast path at the FaB/SBFT quorum ceil((n + 3f + 1) / 2):
  one voting round when that many validators are ready. Its trigger is a
  signature count, so a Byzantine sketch cannot suppress it; Byzantine
  abstention can, as for every one-round protocol.
- Stragglers fetch missing transactions by id; they rejoin the vote.
- Readiness sensing: gossiped sketches tell the proposer when the
  mempool has converged on a candidate block.
- Partition healing: bimodal sketch clouds, estimate-then-decode repair.
- Ledger accumulators (LtHash + sketch) for range digests, O(log H) fork
  localization, and prune accounting.
"""

import hashlib
import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Tuple

import numpy as np

import dpmh
import reconcile

N_DIMS = dpmh.N_ROUND          # digest dimension (v2: 512, was 8)
DIGEST_WIRE = dpmh.wire_bytes("round")   # 1024 B round digest on the wire
SUMMARY_BYTES = DIGEST_WIRE + 4 + 8      # exact sum + count + sumsq dist
MAX_SUPPLY = 21_000_000.0
INITIAL_REWARD = 50.0
HALVING_INTERVAL = 210_000


# ---------------------------------------------------------------------------
# BLS signatures (py-ecc or fallback)
# ---------------------------------------------------------------------------

try:
    from py_ecc.bls import G2ProofOfPossession as bls
    BLS_AVAILABLE = True
except ImportError:
    BLS_AVAILABLE = False

# Real BLS is slow (py-ecc is pure Python). For benchmarks with many validators,
# set USE_REAL_BLS = False to use hash-based mocks that preserve message sizes.
# The demo (node.py with < 10 validators) uses real BLS by default.

USE_REAL_BLS = True

class BLSKeyPair:
    """BLS12-381 keys via py-ecc, or a hash-based mock of the same sizes.

    The mock is verifiable: a signature is SHA-384(pubkey || msg) and an
    aggregate is SHA-384 over the signatures in signer order, so verify()
    and verify_aggregate() recompute from public keys and reject a
    signature on any other message (equivocation). It is not unforgeable;
    it exists to keep large simulations fast while still exercising the
    verification path. tests run real BLS at small N.

    Private keys are sequential small integers: a demo, not key management.
    """
    _counter = 1

    def __init__(self, name=None):
        self.privkey = BLSKeyPair._counter
        BLSKeyPair._counter += 1
        if BLS_AVAILABLE and USE_REAL_BLS:
            self.pubkey = bls.SkToPk(self.privkey)
        else:
            self.pubkey = hashlib.sha256(f"pub:{self.privkey}".encode()).digest()[:48]
        self.name = name or f"v{self.privkey}"

    def sign(self, msg: bytes) -> bytes:
        if BLS_AVAILABLE and USE_REAL_BLS:
            return bls.Sign(self.privkey, msg)
        return hashlib.sha384(self.pubkey + msg).digest()

    @staticmethod
    def verify(pubkey: bytes, msg: bytes, sig: bytes) -> bool:
        if BLS_AVAILABLE and USE_REAL_BLS:
            return bls.Verify(pubkey, msg, sig)
        return sig == hashlib.sha384(pubkey + msg).digest()

    @staticmethod
    def aggregate(signatures: list) -> bytes:
        if BLS_AVAILABLE and USE_REAL_BLS:
            return bls.Aggregate(signatures)
        return hashlib.sha384(b"agg:" + b"".join(signatures)).digest()

    @staticmethod
    def verify_aggregate(pubkeys: list, msg: bytes, agg_sig: bytes,
                         sigs: list = None) -> bool:
        if BLS_AVAILABLE and USE_REAL_BLS:
            return bls.FastAggregateVerify(pubkeys, msg, agg_sig)
        expect = [hashlib.sha384(pk + msg).digest() for pk in pubkeys]
        return agg_sig == hashlib.sha384(b"agg:" + b"".join(expect)).digest()


# ---------------------------------------------------------------------------
# Transaction sketches
# ---------------------------------------------------------------------------

def tx_to_vector(tx_data: str, salt=0) -> np.ndarray:
    """Rademacher vector of one transaction (round domain)."""
    return dpmh.tx_vector(tx_data, salt)


def compute_vector(tx_list: list, salt=0) -> np.ndarray:
    """Integer sketch of a transaction set (round domain)."""
    return dpmh.digest(tx_list, salt)


def calibrate_threshold(txs: list = None, *args, **kwargs) -> float:
    """tau = sqrt(3n), closed form (arguments kept for old call sites)."""
    return dpmh.tau()


def validate_threshold(txs: list, max_miss: int = 2, trials: int = 2000,
                       salt=0, seed: int = 0) -> dict:
    """Monte Carlo check of the closed-form threshold (not protocol)."""
    rng = np.random.default_rng(seed)
    honest = compute_vector(txs, salt)
    exceed = 0
    for _ in range(trials):
        k = int(rng.integers(1, max_miss + 1))
        drop = set(rng.choice(len(txs), size=min(k, len(txs)), replace=False).tolist())
        partial = [tx for j, tx in enumerate(txs) if j not in drop]
        exceed += dpmh.dist2(compute_vector(partial, salt), honest) > dpmh.tau2()
    return {"trials": trials, "exceed": exceed, "rate": exceed / trials,
            "bound": math.exp(-dpmh.N_ROUND / 8)}


# ---------------------------------------------------------------------------
# Message counter
# ---------------------------------------------------------------------------

class MessageCounter:
    """Counts messages and bytes; leader_in tracks bytes arriving at the
    leader or aggregator, the per-node bottleneck of leader-based BFT."""

    def __init__(self):
        self.count = 0
        self.bytes = 0
        self.leader_in = 0
        self.by_type: Dict[str, int] = {}

    def send(self, msg_type: str, size_bytes: int, n: int = 1,
             to_leader: bool = False):
        self.count += n
        self.bytes += size_bytes * n
        if to_leader:
            self.leader_in += size_bytes * n
        self.by_type[msg_type] = self.by_type.get(msg_type, 0) + n


# ---------------------------------------------------------------------------
# Transactions
# ---------------------------------------------------------------------------

@dataclass
class Transaction:
    sender: str
    receiver: str
    amount: float
    nonce: int
    fee: float = 0.01
    timestamp: float = 0.0
    tx_hash: str = ""
    sender_name: str = ""
    receiver_name: str = ""

    def compute_hash(self) -> str:
        data = f"{self.sender}:{self.receiver}:{self.amount:.2f}:{self.fee:.2f}:{self.nonce}:{self.timestamp}"
        self.tx_hash = hashlib.sha256(data.encode()).hexdigest()
        return self.tx_hash

    @property
    def data_str(self) -> str:
        if not self.tx_hash:
            self.compute_hash()
        return self.tx_hash

    def to_dict(self) -> dict:
        return {
            "sender": self.sender_name or self.sender[:8],
            "receiver": self.receiver_name or self.receiver[:8],
            "amount": self.amount,
            "fee": self.fee,
            "nonce": self.nonce,
            "tx_hash": self.tx_hash[:12] + "...",
        }

    def __repr__(self):
        s = self.sender_name or self.sender[:6]
        r = self.receiver_name or self.receiver[:6]
        return f"{s}->{r}:{self.amount:.2f}"


@dataclass
class CoinbaseTx:
    receiver: str
    amount: float
    height: int
    tx_hash: str = ""
    receiver_name: str = ""

    def compute_hash(self) -> str:
        self.tx_hash = hashlib.sha256(
            f"cb:{self.receiver}:{self.amount:.2f}:{self.height}".encode()
        ).hexdigest()
        return self.tx_hash

    @property
    def data_str(self) -> str:
        if not self.tx_hash:
            self.compute_hash()
        return self.tx_hash

    def to_dict(self) -> dict:
        return {
            "type": "coinbase",
            "receiver": self.receiver_name or self.receiver[:8],
            "amount": self.amount,
            "tx_hash": self.tx_hash[:12] + "...",
        }


# ---------------------------------------------------------------------------
# Block
# ---------------------------------------------------------------------------

@dataclass
class Block:
    height: int
    prev_hash: str
    transactions: list
    timestamp: float
    proposer: str
    proposer_name: str = ""
    merkle_root: str = ""
    block_hash: str = ""

    def __post_init__(self):
        hashes = [bytes.fromhex(tx.compute_hash()) for tx in self.transactions]
        if not hashes:
            hashes = [hashlib.sha256(b"empty").digest()]
        while len(hashes) > 1:
            if len(hashes) % 2:
                hashes.append(hashes[-1])
            hashes = [
                hashlib.sha256(hashes[i] + hashes[i + 1]).digest()
                for i in range(0, len(hashes), 2)
            ]
        self.merkle_root = hashes[0].hex()
        self.block_hash = hashlib.sha256(
            f"{self.height}:{self.prev_hash}:{self.merkle_root}:{self.timestamp}:{self.proposer}".encode()
        ).hexdigest()

    @property
    def tx_data_strings(self) -> list:
        return [tx.data_str for tx in self.transactions]

    def to_dict(self) -> dict:
        return {
            "height": self.height,
            "block_hash": self.block_hash[:16] + "...",
            "prev_hash": self.prev_hash[:16] + "...",
            "merkle_root": self.merkle_root[:16] + "...",
            "proposer": self.proposer_name or self.proposer[:8],
            "timestamp": self.timestamp,
            "n_transactions": len(self.transactions),
            "transactions": [tx.to_dict() for tx in self.transactions],
        }


# ---------------------------------------------------------------------------
# Chain state (account-based)
# ---------------------------------------------------------------------------

class State:
    def __init__(self):
        self.balances: Dict[str, float] = {}
        self.nonces: Dict[str, int] = {}
        self.names: Dict[str, str] = {}  # addr -> human name
        self.addr_by_name: Dict[str, str] = {}  # human name -> addr
        self.supply: float = 0.0

    def bal(self, addr: str) -> float:
        return self.balances.get(addr, 0.0)

    def nonce(self, addr: str) -> int:
        return self.nonces.get(addr, 0)

    def apply_tx(self, tx: Transaction) -> bool:
        if self.bal(tx.sender) < tx.amount + tx.fee:
            return False
        if tx.nonce != self.nonce(tx.sender):
            return False
        self.balances[tx.sender] -= tx.amount + tx.fee
        self.balances[tx.receiver] = self.bal(tx.receiver) + tx.amount
        self.nonces[tx.sender] = tx.nonce + 1
        return True

    def apply_coinbase(self, cb: CoinbaseTx):
        self.balances[cb.receiver] = self.bal(cb.receiver) + cb.amount
        self.supply += cb.amount

    def snapshot(self) -> tuple:
        return dict(self.balances), dict(self.nonces), self.supply

    def restore(self, snap: tuple):
        self.balances, self.nonces, self.supply = dict(snap[0]), dict(snap[1]), snap[2]

    def register(self, name: str, balance: float = 0.0) -> str:
        """Register a named account. Returns address."""
        kp = BLSKeyPair(name=name)
        addr = kp.pubkey.hex() if isinstance(kp.pubkey, bytes) else str(kp.pubkey)
        self.names[addr] = name
        self.addr_by_name[name] = addr
        if balance > 0:
            self.balances[addr] = balance
            self.supply += balance
        return addr


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------

BYZANTINE_STRATEGIES = ["drop_half", "random_vector", "replace_one_tx",
                        "mimic_honest", "coalition", "equivocate", "abstain"]


class Validator:
    def __init__(self, vid: int, keypair: BLSKeyPair,
                 is_byzantine: bool = False, strategy: str = "drop_half"):
        self.id = vid
        self.kp = keypair
        self.is_byzantine = is_byzantine
        self.strategy = strategy

    @property
    def name(self) -> str:
        return self.kp.name

    def get_vector(self, block: Block, missing: Optional[Set[int]] = None) -> np.ndarray:
        """Sketch of this validator's view of the block (round domain)."""
        salt = dpmh.round_salt(block.height, block.prev_hash)
        if self.is_byzantine:
            return self._byzantine_vector(block, salt)
        strs = block.tx_data_strings
        if missing:
            strs = [s for i, s in enumerate(strs) if i not in missing]
        return compute_vector(strs, salt)

    def _byzantine_vector(self, block: Block, salt) -> np.ndarray:
        strs = block.tx_data_strings
        rng = np.random.default_rng(self.id * 7919 + block.height)
        if not strs or self.strategy == "random_vector":
            scale = max(len(strs), 1)
            return rng.integers(-scale, scale + 1, size=N_DIMS).astype(np.int64)
        if self.strategy == "drop_half":
            return compute_vector(strs[::2], salt)
        if self.strategy == "replace_one_tx":
            m = list(strs)
            m[min(1, len(m) - 1)] = hashlib.sha256(b"FRAUD").hexdigest()
            return compute_vector(m, salt)
        if self.strategy == "mimic_honest":
            m = list(strs)
            m[0] = hashlib.sha256(b"SLIGHT").hexdigest()
            return compute_vector(m, salt)
        if self.strategy == "coalition":
            return compute_vector(strs[:len(strs) // 2], salt)
        return compute_vector(strs, salt)       # equivocate, abstain

    def to_dict(self) -> dict:
        d = {"id": self.id, "name": self.name, "byzantine": self.is_byzantine}
        if self.is_byzantine:
            d["strategy"] = self.strategy
        return d


# ---------------------------------------------------------------------------
# Consensus accounting (one height, one view with an honest leader)
# ---------------------------------------------------------------------------
#
# The protocol is bft.py's: two voting phases with locking (prepare QC,
# commit QC) and view change, plus the speculative fast path at the
# FaB/SBFT quorum n_fast = ceil((n + 3f + 1) / 2). bft.py fuzzes safety
# under equivocation and adversarial scheduling; the functions below
# account messages and bytes for the common case of an honest leader after
# GST, which is what the scaling comparison measures.
#
# What the sketch does here. Inside consensus the reference is D(B), the
# sketch of the proposal, which anyone can compute from the block, so no
# party has discretion over it. A validator that holds the proposal knows
# exactly which transaction ids it lacks and fetches them by id. The sketch
# on each prepare vote is observability: it tells the aggregator who was
# behind, by roughly how much, and whether the cloud is bimodal (partition),
# without shipping id lists. It never gates a vote or a certificate.

FETCH_REQ_BYTES = 32          # per missing transaction id requested
TX_BYTES = 200                # transaction body
VOTE_BYTES = 96 + 32          # BLS signature + block hash / view metadata
SAFETY_F = None               # default: bft.max_faults(n)
# Byzantine strategies that put a signature on the wire: equivocate (on a
# conflicting message) or sign correctly while lying in the sketch.
SIGNING_STRATEGIES = {"equivocate", "mimic_honest", "replace_one_tx"}


def fetch_bytes(k: int) -> int:
    """Request k ids and receive k bodies (shared with the baselines)."""
    return k * (FETCH_REQ_BYTES + TX_BYTES)


def _cert_bytes(n: int, qc: str = "bitmap") -> int:
    """Certificate: aggregate signature plus signer bitmap, or a constant
    threshold signature."""
    return 96 + (math.ceil(n / 8) if qc == "bitmap" else 0) + 64


def _sign_or_none(v, msg: bytes, phase: str):
    """A validator's signature on msg for this phase, or None.

    Byzantine behaviours: "equivocate" signs a different message (rejected
    at verification), "mimic_honest" and "replace_one_tx" sign correctly
    (only their sketches lie), every other strategy abstains.
    """
    if not v.is_byzantine:
        return v.kp.sign(msg)
    if v.strategy == "equivocate":
        return v.kp.sign(msg + b"|conflict")
    if v.strategy in SIGNING_STRATEGIES:
        return v.kp.sign(msg)
    return None


def _n_senders(validators) -> int:
    """Validators that put a vote on the wire (valid or not)."""
    return sum(1 for v in validators
               if not v.is_byzantine or v.strategy in SIGNING_STRATEGIES)


def _collect(validators, msg: bytes, voters, phase: str):
    """Collect and verify signatures; return (valid signer ids, agg sig)."""
    sigs, pubs, ids = [], [], []
    for v in voters:
        s = _sign_or_none(v, msg, phase)
        if s is None:
            continue
        if not BLSKeyPair.verify(v.kp.pubkey, msg, s):
            continue                      # equivocation or garbage: dropped
        sigs.append(s)
        pubs.append(v.kp.pubkey)
        ids.append(v.id)
    agg = BLSKeyPair.aggregate(sigs) if sigs else None
    if agg is not None:
        assert BLSKeyPair.verify_aggregate(pubs, msg, agg, sigs)
    return ids, agg


def vector_consensus(validators: list, block: Block, threshold: float = None,
                     partial_obs: Optional[dict] = None, f: int = SAFETY_F,
                     qc: str = "bitmap", observe: bool = True) -> dict:
    """Flat Proxima on the base protocol: one view, honest leader.

    Round 1 (PREPARE): the leader multicasts the proposal (tx ids); each
    validator replies with a prepare signature on H(B) and, if observe,
    the sketch of its own view of B. Validators with the full block vote
    at once; stragglers fetch their missing ids from the leader first and
    vote late. If n_fast valid prepare signatures arrive on time, they are
    the decision certificate (fast path). Otherwise the leader waits for
    stragglers, forms the prepare QC (n - f), and the commit phase follows
    (commit QC, n - f). The certificate is delivered to all N validators.
    """
    import bft
    t0 = time.time()
    msgs = MessageCounter()
    n = len(validators)
    f = bft.max_faults(n) if f is None else f
    q, n_fast = bft.quorum(n, f), bft.fast_quorum(n, f)
    partial_obs = partial_obs or {}
    salt = dpmh.round_salt(block.height, block.prev_hash)
    ref = dpmh.digest(block.tx_data_strings, salt)
    tau2 = dpmh.tau2()
    h = block.block_hash.encode()

    # Proposal to everyone.
    msgs.send("propose", 64 + 32 * len(block.transactions), n)

    # Stragglers fetch missing ids before voting.
    stragglers = [v for v in validators
                  if not v.is_byzantine and partial_obs.get(v.id)]
    sync_details = []
    for v in stragglers:
        k = len(partial_obs[v.id])
        msgs.send("fetch_by_id", fetch_bytes(k))
        sync_details.append((v.name, k))

    # Observability sketches ride on prepare votes (pre-fetch view).
    distances, flagged = {}, []
    if observe:
        for v in validators:
            d2 = dpmh.dist2(v.get_vector(block, partial_obs.get(v.id)), ref)
            distances[v.id] = math.sqrt(d2)
            if d2 > tau2:
                flagged.append(v)
    sketch = DIGEST_WIRE if observe else 0

    on_time = [v for v in validators if v not in stragglers]
    p_on_time, agg = _collect(validators, b"prepare:" + h, on_time, "prepare")
    p_late, _ = _collect(validators, b"prepare:" + h, stragglers, "prepare")
    msgs.send("prepare_vote", VOTE_BYTES + sketch, _n_senders(validators),
              to_leader=True)

    fast = len(p_on_time) >= n_fast
    cert = _cert_bytes(n, qc)
    if fast:
        msgs.send("fast_certificate", cert, n)
        n_commits, rounds = len(p_on_time), 1
        finalized = True
    else:
        prepared = p_on_time + p_late
        if len(prepared) < q:
            finalized, n_commits, rounds = False, len(prepared), 1
        else:
            msgs.send("prepare_qc", cert, n)
            c_ids, _ = _collect(validators, b"commit:" + h, validators, "commit")
            msgs.send("commit_vote", VOTE_BYTES, _n_senders(validators),
                      to_leader=True)
            finalized = len(c_ids) >= q
            if finalized:
                msgs.send("commit_qc", cert, n)
            n_commits, rounds = len(c_ids), 2

    return {
        "finalized": finalized,
        "fast_path": fast,
        "rounds": rounds,
        "f": f, "n_fast": n_fast, "quorum": q,
        "cluster_size": n - len(flagged),
        "excluded": [(v.name, v.is_byzantine,
                      v.strategy if v.is_byzantine else None,
                      distances.get(v.id, 0.0)) for v in flagged],
        "cluster_variance": float(np.var(list(distances.values())))
                            if distances else 0.0,
        "total_time": time.time() - t0,
        "msgs": msgs.count,
        "msg_bytes": msgs.bytes,
        "leader_in_bytes": msgs.leader_in,
        "msg_breakdown": dict(msgs.by_type),
        "sync_pushed": sum(k for _, k in sync_details),
        "sync_details": sync_details,
        "n_commits": n_commits,
        "n_required": n_fast if fast else q,
        "finality_proof_bytes": cert,
        "distances": {v.name: distances.get(v.id, 0.0) for v in validators},
    }


def _tree_shape(n_leaves: int, branching: int) -> list:
    """Node counts per level, leaves first, root last: [c0, c1, ..., 1]."""
    counts = [n_leaves]
    while counts[-1] > 1:
        counts.append(math.ceil(counts[-1] / branching))
    return counts


def _tree_up(msgs, phase: str, leaves: list, counts: list, vote_bytes: int,
             agg_bytes: int) -> int:
    """Account one aggregation pass up the tree; returns fallback leaves.

    Members send votes to their leaf leader; an honest leader forwards one
    aggregate, a Byzantine leader drops the leaf and its members resend to
    the root after a timeout. Level-i nodes send one aggregate each to
    their parent; only the last hop lands at the root (leader ingress).
    """
    single = len(counts) == 1
    fallbacks = 0
    for leaf in leaves:
        msgs.send(f"{phase}_L0_vote", vote_bytes, len(leaf) - 1,
                  to_leader=single)
        if leaf[0].is_byzantine and not single:
            fallbacks += 1
            msgs.send(f"{phase}_fallback", vote_bytes, len(leaf) - 1,
                      to_leader=True)
    for i in range(len(counts) - 1):
        sending = counts[i] - (fallbacks if i == 0 else 0)
        msgs.send(f"{phase}_agg_L{i}", agg_bytes, sending,
                  to_leader=(i == len(counts) - 2))
    return fallbacks


def tree_consensus(validators: list, block: Block, threshold: float = None,
                   partial_obs: Optional[dict] = None, branching: int = 10,
                   f: int = SAFETY_F, qc: str = "bitmap",
                   observe: bool = True) -> dict:
    """Tree-routed Proxima: the same two phases, aggregated up a tree.

    Leaves of size `branching`; each leaf's first member is its leader, so
    leaders inherit the random placement of Byzantine validators. BLS
    aggregation is associative, so each node forwards one aggregate. A
    Byzantine leaf leader drops its leaf; members resend to the root after
    a timeout (Kauri-style fallback, counted). With observe, leaf leaders
    also forward exact integer summaries: the sum of member sketches, the
    member count, and the sum of squared distances to D(B).
    """
    import bft
    t0 = time.time()
    msgs = MessageCounter()
    n = len(validators)
    f = bft.max_faults(n) if f is None else f
    q, n_fast = bft.quorum(n, f), bft.fast_quorum(n, f)
    partial_obs = partial_obs or {}
    salt = dpmh.round_salt(block.height, block.prev_hash)
    ref = dpmh.digest(block.tx_data_strings, salt)
    tau2 = dpmh.tau2()
    h = block.block_hash.encode()
    cert = _cert_bytes(n, qc)
    agg = 96 + math.ceil(n / 8)
    sketch = DIGEST_WIRE if observe else 0

    leaves = [validators[i:i + branching] for i in range(0, n, branching)]
    counts = _tree_shape(len(leaves), branching)

    msgs.send("propose_down", 64 + 32 * len(block.transactions), n - 1)
    stragglers = {v.id for v in validators
                  if not v.is_byzantine and partial_obs.get(v.id)}
    sync = 0
    for vid in stragglers:
        k = len(partial_obs[vid])
        msgs.send("fetch_by_id", fetch_bytes(k))
        sync += k

    flagged, ssq = 0, 0
    if observe:
        for v in validators:
            d2 = dpmh.dist2(v.get_vector(block, partial_obs.get(v.id)), ref)
            ssq += d2
            flagged += d2 > tau2
    fb = _tree_up(msgs, "prepare", leaves, counts, VOTE_BYTES + sketch,
                  agg + (SUMMARY_BYTES if observe else 0))

    on_time = [v for v in validators if v.id not in stragglers]
    late = [v for v in validators if v.id in stragglers]
    p_on, _ = _collect(validators, b"prepare:" + h, on_time, "prepare")
    p_late, _ = _collect(validators, b"prepare:" + h, late, "prepare")
    fast = len(p_on) >= n_fast
    if fast:
        msgs.send("fast_certificate_down", cert, n - 1)
        finalized, n_commits, rounds = True, len(p_on), 1
    elif len(p_on) + len(p_late) < q:
        finalized, n_commits, rounds = False, len(p_on) + len(p_late), 1
    else:
        msgs.send("prepare_qc_down", cert, n - 1)
        c_ids, _ = _collect(validators, b"commit:" + h, validators, "commit")
        _tree_up(msgs, "commit", leaves, counts, VOTE_BYTES, agg)
        finalized = len(c_ids) >= q
        if finalized:
            msgs.send("commit_qc_down", cert, n - 1)
        n_commits, rounds = len(c_ids), 2

    level_stats = [{"level": 0, "groups": len(leaves), "excluded": flagged,
                    "passed": n - flagged, "fallback_leaves": fb}]
    level_stats += [{"level": i, "groups": c, "msgs_this_level": c}
                    for i, c in enumerate(counts[1:], start=1)]
    return {
        "finalized": finalized, "fast_path": fast, "rounds": rounds,
        "tree_mode": True, "f": f, "n_fast": n_fast, "quorum": q,
        "cluster_size": n - flagged, "excluded": [],
        "cluster_variance": ssq / n / N_DIMS,
        "total_time": time.time() - t0,
        "msgs": msgs.count, "msg_bytes": msgs.bytes,
        "leader_in_bytes": msgs.leader_in,
        "msg_breakdown": dict(msgs.by_type),
        "sync_pushed": sync, "sync_details": [],
        "n_commits": n_commits, "n_required": n_fast if fast else q,
        "finality_proof_bytes": cert,
        "n_levels": len(counts), "n_leaves": len(leaves),
        "branching": branching, "fallback_leaves": fb,
        "level_stats": level_stats, "distances": {},
    }


# ---------------------------------------------------------------------------
# Block reward
# ---------------------------------------------------------------------------

def block_reward(height: int) -> float:
    halvings = height // HALVING_INTERVAL
    if halvings >= 64:
        return 0.0
    return INITIAL_REWARD / (2 ** halvings)


# ---------------------------------------------------------------------------
# Blockchain
# ---------------------------------------------------------------------------

class Blockchain:
    def __init__(self, validators: list, clock=None):
        """clock: callable returning a timestamp for txs and blocks. The
        default is a logical counter so every hash, digest, and result is
        reproducible run to run; the live node passes time.time."""
        self.validators = validators
        self._ticks = 0
        self.clock = clock or self._logical_clock
        self.state = State()
        self.chain: List[Block] = []
        self.mempool: List[Transaction] = []
        self.consensus_log: list = []
        # v2: cumulative ledger-domain accumulator, P[b] per block height.
        # Enables O(1) range digests and O(log H) fork localization.
        self.ledger_acc = dpmh.Accumulator()

    def _logical_clock(self) -> float:
        self._ticks += 1
        return float(self._ticks)

    @property
    def height(self) -> int:
        return len(self.chain)

    @property
    def tip(self) -> str:
        return self.chain[-1].block_hash if self.chain else "0" * 64

    def register_account(self, name: str, balance: float = 0.0) -> str:
        return self.state.register(name, balance)

    def addr_for(self, name: str) -> Optional[str]:
        return self.state.addr_by_name.get(name)

    def name_for(self, addr: str) -> str:
        return self.state.names.get(addr, addr[:8])

    def make_tx(self, sender_name: str, receiver_name: str, amount: float,
                fee: float = 0.01) -> Optional[Transaction]:
        s_addr = self.state.addr_by_name.get(sender_name)
        r_addr = self.state.addr_by_name.get(receiver_name)
        if not s_addr or not r_addr:
            return None
        pending_nonce = sum(1 for t in self.mempool if t.sender == s_addr)
        tx = Transaction(
            sender=s_addr,
            receiver=r_addr,
            amount=amount,
            nonce=self.state.nonce(s_addr) + pending_nonce,
            fee=fee,
            timestamp=self.clock(),
            sender_name=sender_name,
            receiver_name=receiver_name,
        )
        tx.compute_hash()
        return tx

    def submit_tx(self, tx: Transaction) -> bool:
        self.mempool.append(tx)
        return True

    def propose_block(self, proposer: Validator) -> Block:
        addr = proposer.kp.pubkey.hex() if isinstance(proposer.kp.pubkey, bytes) else str(proposer.kp.pubkey)
        cb = CoinbaseTx(addr, block_reward(self.height), self.height,
                        receiver_name=proposer.name)
        cb.compute_hash()
        seen, txs = set(), [cb]
        for tx in self.mempool:                 # a block is a set (MU_MAX)
            if tx.data_str not in seen:
                seen.add(tx.data_str)
                txs.append(tx)
        return Block(self.height, self.tip, txs, self.clock(), addr,
                     proposer_name=proposer.name)

    def finalize_block(self, block: Block) -> bool:
        try:
            dpmh.assert_set(block.tx_data_strings)
        except ValueError:
            return False
        snap = self.state.snapshot()
        fees = 0.0
        for tx in block.transactions:
            if isinstance(tx, CoinbaseTx):
                self.state.apply_coinbase(tx)
            else:
                if not self.state.apply_tx(tx):
                    self.state.restore(snap)
                    return False
                fees += tx.fee
        # Miner gets fees
        self.state.balances[block.proposer] = self.state.bal(block.proposer) + fees
        self.chain.append(block)
        self.ledger_acc.append_block(block.tx_data_strings)
        # Clear finalized txs from mempool
        done = {tx.tx_hash for tx in block.transactions if isinstance(tx, Transaction)}
        self.mempool = [tx for tx in self.mempool if tx.tx_hash not in done]
        return True

    def range_digest(self, a: int, b: int):
        """Ledger digest of all transactions in blocks (a, b], O(1)."""
        return self.ledger_acc.range_digest(a, b)

    def find_fork_with(self, other: "Blockchain") -> dict:
        """Locate the fork point against another chain in O(log H) probes."""
        return dpmh.find_fork(self.ledger_acc, other.ledger_acc)

    def mine_block(self, proposer: Validator, threshold: float,
                   partial_obs: Optional[dict] = None) -> Tuple[bool, dict]:
        block = self.propose_block(proposer)
        result = vector_consensus(self.validators, block, threshold, partial_obs)
        if result["finalized"]:
            if not self.finalize_block(block):
                result["finalized"] = False
        result["block"] = block
        self.consensus_log.append(result)
        return result["finalized"], result

    def get_history(self, name: str) -> list:
        addr = self.state.addr_by_name.get(name)
        if not addr:
            return []
        history = []
        for block in self.chain:
            for tx in block.transactions:
                if isinstance(tx, Transaction):
                    if tx.sender == addr or tx.receiver == addr:
                        direction = "sent" if tx.sender == addr else "received"
                        other = tx.receiver_name if direction == "sent" else tx.sender_name
                        history.append({
                            "block": block.height,
                            "direction": direction,
                            "other": other,
                            "amount": tx.amount,
                            "tx_hash": tx.tx_hash[:12],
                        })
                elif isinstance(tx, CoinbaseTx) and tx.receiver == addr:
                    history.append({
                        "block": block.height,
                        "direction": "mined",
                        "other": "coinbase",
                        "amount": tx.amount,
                        "tx_hash": tx.tx_hash[:12],
                    })
        return history


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_validators(n_honest: int, n_byz: int = 0, strategy: str = "drop_half",
                    seed: int = 0, shuffle: bool = True) -> Tuple[list, list, list]:
    """Validators with Byzantine members at random positions.

    Position matters: tree leaves are contiguous slices and a leaf's first
    member is its leader, so placing Byzantine validators at the end (as
    v1 did) would hide Byzantine leaf leaders entirely. ids equal list
    positions. The first honest validator in the returned honest list is
    the default proposer.
    """
    BLSKeyPair._counter = 1
    roles = [False] * n_honest + [True] * n_byz
    if shuffle:
        np.random.default_rng(seed).shuffle(roles)
    out, h_i, b_i = [], 0, 0
    for pos, is_b in enumerate(roles):
        if is_b:
            out.append(Validator(pos, BLSKeyPair(f"Byz-{b_i}"), True, strategy))
            b_i += 1
        else:
            out.append(Validator(pos, BLSKeyPair(f"Miner-{h_i}")))
            h_i += 1
    honest = [v for v in out if not v.is_byzantine]
    byzantine = [v for v in out if v.is_byzantine]
    return out, honest, byzantine


def make_partial_obs(validators: list, n_txs: int, max_miss: int = 2,
                     miss_prob: float = 0.37, rng=None) -> dict:
    """Some honest validators have not yet received 1..max_miss of the
    block's transactions when the proposal arrives. miss_prob is a model
    parameter (swept in the evaluation), not a measured constant."""
    # Default draws from the global numpy seed so np.random.seed() pins it.
    rng = rng or np.random.default_rng(np.random.randint(2 ** 31))
    obs = {}
    for v in validators:
        if v.is_byzantine:
            continue
        if rng.random() < miss_prob:
            k = int(rng.integers(1, max_miss + 1))
            obs[v.id] = set(rng.choice(n_txs, size=min(k, n_txs),
                                       replace=False).tolist())
    return obs


# ---------------------------------------------------------------------------
# v2: readiness sensing (sense, then propose)
# ---------------------------------------------------------------------------

def _gossip_tick(obs: dict, fill_prob: float, rng) -> dict:
    """One gossip interval: each missing tx arrives with prob fill_prob."""
    out = {}
    for vid, miss in obs.items():
        rest = {i for i in miss if rng.random() >= fill_prob}
        if rest:
            out[vid] = rest
    return out


def readiness_sense(validators: list, block: Block, initial_obs: dict,
                    fill_prob: float = 0.5, max_ticks: int = 8,
                    f: int = SAFETY_F, rng=None) -> dict:
    """Propose when the gossiped sketches say the fast path will fire.

    Each tick every validator gossips the sketch of its view of the
    candidate block to the proposer (DIGEST_WIRE bytes each, charged). The
    proposer counts sketches exactly equal to D(B) and proposes once that
    count reaches the fast quorum, or at max_ticks. Byzantine validators
    can report D(B) without holding it, which only makes the proposer
    propose early: a latency cost, never a safety one. Compare against
    fixed_delay_propose at the same mean delay, not against delay zero.
    """
    import bft
    rng = rng or np.random.default_rng(0)
    n = len(validators)
    f = bft.max_faults(n) if f is None else f
    need = bft.fast_quorum(n, f)
    salt = dpmh.round_salt(block.height, block.prev_hash)
    ref = dpmh.digest(block.tx_data_strings, salt)
    obs = {vid: set(m) for vid, m in initial_obs.items()}
    ticks, gossip_bytes = 0, 0
    while True:
        gossip_bytes += n * DIGEST_WIRE
        matches = sum(1 for v in validators
                      if dpmh.dist2(v.get_vector(block, obs.get(v.id)), ref) == 0)
        if matches >= need or ticks >= max_ticks:
            break
        obs = _gossip_tick(obs, fill_prob, rng)
        ticks += 1
    result = vector_consensus(validators, block, None, obs, f=f)
    result.update(sense_ticks=ticks, sensed_matches=matches,
                  sensing_bytes=gossip_bytes)
    return result


def fixed_delay_propose(validators: list, block: Block, initial_obs: dict,
                        delay: int, fill_prob: float = 0.5, f: int = SAFETY_F,
                        rng=None) -> dict:
    """Baseline: wait a fixed number of gossip ticks, then propose."""
    rng = rng or np.random.default_rng(0)
    obs = {vid: set(m) for vid, m in initial_obs.items()}
    for _ in range(delay):
        obs = _gossip_tick(obs, fill_prob, rng)
    result = vector_consensus(validators, block, None, obs, f=f)
    result["sense_ticks"] = delay
    return result


# ---------------------------------------------------------------------------
# v2: partition detection and healing
# ---------------------------------------------------------------------------

def split_bimodal(digests: dict) -> Tuple[list, list]:
    """Split a digest cloud into two camps (farthest-pair seeding).

    Returns two lists of ids. Exact centroid arithmetic (integer sums,
    verifier-side division) means any observer of the signed digests
    computes the same split.
    """
    ids = list(digests.keys())
    if len(ids) < 2:
        return ids, []
    # Seed with the farthest pair (O(k^2) over unique digests is fine at
    # simulation scale; production would seed from cluster stats).
    best = (ids[0], ids[1])
    best_d = -1
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            d = dpmh.dist2(digests[ids[i]], digests[ids[j]])
            if d > best_d:
                best_d = d
                best = (ids[i], ids[j])
    seed_a, seed_b = best
    camp_a, camp_b = [], []
    for vid in ids:
        da = dpmh.dist2(digests[vid], digests[seed_a])
        db = dpmh.dist2(digests[vid], digests[seed_b])
        (camp_a if da <= db else camp_b).append(vid)
    return camp_a, camp_b


def heal_partition(set_a: set, set_b: set, salt=0, session: int = 0) -> dict:
    """Reconcile two camps after a partition.

    Camp centroids are exact by linearity, so the inter-camp divergence
    d_hat = ||mu_a - mu_b||^2 / n is visible to anyone holding the camps'
    sketches before reconciliation starts. One estimate-then-decode
    exchange between camp representatives recovers the divergent
    transactions: A learns B's extras (fetched by short id, counted in
    rec["bytes"]) and pushes its own extras to B (push_bytes). With an
    adversary who knows the salt, d_hat can be deflated (dpmh.
    adversarial_balance); the IBLT then doubles, costing rounds, not
    correctness.
    """
    d_a = dpmh.digest(sorted(set_a), salt)
    d_b = dpmh.digest(sorted(set_b), salt)
    d_hat = dpmh.est_symdiff(d_a, d_b)
    rec = reconcile.reconcile(set_a, set_b, d_hint=max(1.0, d_hat),
                              session=session)
    union = set(set_a) | set(set_b)
    return {
        "d_hat": d_hat,
        "true_d": rec["true_d"],
        "recovered_union": rec["ok"] and (set(set_a) | rec["only_b"]) == union
                           and (set(set_b) | rec["only_a"]) == union,
        "bytes": rec["bytes"],
        "rounds": rec["rounds"],
        "push_bytes": len(rec["only_a"]) * reconcile.ITEM_BYTES,
    }
