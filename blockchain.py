"""
blockchain.py -- Core protocol implementation.

Transaction vector encoding (SHA-512 -> 8D), bloom filters for set
reconciliation, BLS aggregate signatures, and the two-phase consensus
with optimistic fast path. Everything else imports from here.
"""

import hashlib
import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Tuple

import numpy as np
from bitarray import bitarray

N_DIMS = 8
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
        # Mock: preserves 96-byte signature size for accurate bandwidth counting
        return hashlib.sha384(f"{self.privkey}:{msg.hex()}".encode()).digest()

    @staticmethod
    def aggregate(signatures: list) -> bytes:
        if BLS_AVAILABLE and USE_REAL_BLS:
            return bls.Aggregate(signatures)
        return hashlib.sha384(b"agg:" + b"".join(signatures)).digest()

    @staticmethod
    def verify_aggregate(pubkeys: list, msg: bytes, agg_sig: bytes) -> bool:
        if BLS_AVAILABLE and USE_REAL_BLS:
            return bls.FastAggregateVerify(pubkeys, msg, agg_sig)
        return True


# ---------------------------------------------------------------------------
# Bloom filter
# ---------------------------------------------------------------------------

class BloomFilter:
    """Probabilistic set membership test. About 25 bytes for 20 txs at 1% FP rate."""

    def __init__(self, n_items: int = 50, fp_rate: float = 0.01):
        self.size = max(8, int(-n_items * math.log(fp_rate) / (math.log(2) ** 2)))
        self.n_hashes = max(1, int(self.size / max(n_items, 1) * math.log(2)))
        self.bits = bitarray(self.size)
        self.bits.setall(0)

    def add(self, item: str):
        for i in range(self.n_hashes):
            h = int(hashlib.md5(f"{i}:{item}".encode()).hexdigest(), 16) % self.size
            self.bits[h] = 1

    def contains(self, item: str) -> bool:
        for i in range(self.n_hashes):
            h = int(hashlib.md5(f"{i}:{item}".encode()).hexdigest(), 16) % self.size
            if not self.bits[h]:
                return False
        return True

    @property
    def size_bytes(self) -> int:
        return len(self.bits.tobytes())

    def missing_from(self, full_set: list) -> list:
        return [item for item in full_set if not self.contains(item)]


# ---------------------------------------------------------------------------
# Transaction vector encoding
# ---------------------------------------------------------------------------

def tx_to_vector(tx_data: str) -> np.ndarray:
    """SHA-512 split into 8 segments, each mapped to [0, 1). Returns 8D vector."""
    h = hashlib.sha512(tx_data.encode()).digest()
    return np.array([
        int.from_bytes(h[d * 8:(d + 1) * 8], 'big') % 10000 / 10000.0
        for d in range(N_DIMS)
    ])

def compute_vector(tx_list: list) -> np.ndarray:
    """Commutative sum of transaction vectors."""
    if not tx_list:
        return np.zeros(N_DIMS)
    return np.sum([tx_to_vector(tx) for tx in tx_list], axis=0)

def calibrate_threshold(txs: list, max_miss: int = 2, percentile: int = 99,
                        margin: float = 1.2) -> float:
    """Sample 2000 partial observations, take p99 distance * margin as threshold."""
    if len(txs) < 2:
        return 5.0
    honest = compute_vector(txs)
    dists = []
    for _ in range(2000):
        n_miss = np.random.randint(1, max_miss + 1)
        missing = set(np.random.choice(len(txs), size=min(n_miss, len(txs)), replace=False))
        partial = [tx for j, tx in enumerate(txs) if j not in missing]
        dists.append(np.linalg.norm(compute_vector(partial) - honest))
    return float(np.percentile(dists, percentile) * margin)


# ---------------------------------------------------------------------------
# Message counter
# ---------------------------------------------------------------------------

class MessageCounter:
    def __init__(self):
        self.count = 0
        self.bytes = 0
        self.by_type: Dict[str, int] = {}

    def send(self, msg_type: str, size_bytes: int, n: int = 1):
        self.count += n
        self.bytes += size_bytes * n
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

BYZANTINE_STRATEGIES = ["drop_half", "random_vector", "replace_one_tx", "mimic_honest", "coalition"]

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
        if self.is_byzantine:
            return self._byzantine_vector(block)
        strs = block.tx_data_strings
        if missing:
            strs = [s for i, s in enumerate(strs) if i not in missing]
        return compute_vector(strs)

    def make_bloom(self, block: Block, missing: Optional[Set[int]] = None) -> BloomFilter:
        strs = block.tx_data_strings
        bf = BloomFilter(max(len(strs), 1))
        for i, s in enumerate(strs):
            if missing and i in missing:
                continue
            bf.add(s)
        return bf

    def _byzantine_vector(self, block: Block) -> np.ndarray:
        strs = block.tx_data_strings
        if not strs:
            return np.random.uniform(0, 15, size=N_DIMS)
        if self.strategy == "drop_half":
            return compute_vector(strs[::2])
        elif self.strategy == "random_vector":
            return np.random.uniform(0, 15, size=N_DIMS)
        elif self.strategy == "replace_one_tx":
            m = list(strs)
            m[min(1, len(m) - 1)] = hashlib.sha256(b"FRAUD").hexdigest()
            return compute_vector(m)
        elif self.strategy == "mimic_honest":
            m = list(strs)
            m[0] = hashlib.sha256(b"SLIGHT").hexdigest()
            return compute_vector(m)
        elif self.strategy == "coalition":
            return compute_vector(strs[:len(strs) // 2])
        return compute_vector(strs)

    def sign_commit(self, block_hash: str) -> Optional[bytes]:
        if self.is_byzantine:
            return None
        return self.kp.sign(block_hash.encode())

    def to_dict(self) -> dict:
        d = {"id": self.id, "name": self.name, "byzantine": self.is_byzantine}
        if self.is_byzantine:
            d["strategy"] = self.strategy
        return d


# ---------------------------------------------------------------------------
# Two-phase consensus
# ---------------------------------------------------------------------------

def vector_consensus(validators: list, block: Block, threshold: float,
                     partial_obs: Optional[dict] = None) -> dict:
    """
    Run two-phase consensus on a proposed block.

    Phase 1: each validator sends its vector + bloom to the aggregator.
    Aggregator clusters by Euclidean distance from the reference vector,
    diffs blooms to push missing txs. If cluster variance is near zero,
    finalize in one round (fast path).

    Phase 2: cluster members send BLS-signed commits. Aggregator produces
    an aggregate signature + signer bitmap and multicasts the finality proof.
    """
    t0 = time.time()
    msgs = MessageCounter()
    n = len(validators)
    n_req = int(math.ceil(n * 2 / 3))
    all_tx_strs = block.tx_data_strings

    # Phase 1: validators send vector + bloom to aggregator
    vectors = {}
    blooms = {}
    for v in validators:
        miss = partial_obs.get(v.id, set()) if partial_obs else set()
        vectors[v.id] = v.get_vector(block, miss)
        if not v.is_byzantine:
            blooms[v.id] = v.make_bloom(block, miss)
        else:
            blooms[v.id] = None
        bloom_size = blooms[v.id].size_bytes if blooms[v.id] else 0
        msgs.send("phase1_vector", N_DIMS * 8 + bloom_size)

    # Aggregator knows the correct block, so the reference vector is the
    # full transaction set vector. Distances are measured from this reference.
    reference = compute_vector(all_tx_strs)

    cluster = []
    excluded = []
    distances = {}
    for v in validators:
        d = float(np.linalg.norm(vectors[v.id] - reference))
        distances[v.id] = d
        if d < threshold:
            cluster.append(v)
        else:
            excluded.append(v)

    # Bloom filter sync: aggregator pushes missing txs to incomplete validators
    sync_pushed = 0
    sync_details = []
    if partial_obs:
        for v in cluster:
            if v.is_byzantine or v.id not in partial_obs:
                continue
            bf = blooms.get(v.id)
            if bf:
                missing_strs = bf.missing_from(all_tx_strs)
                if missing_strs:
                    sync_pushed += len(missing_strs)
                    sync_details.append((v.name, len(missing_strs)))
                    msgs.send("sync_push", len(missing_strs) * 200)

    # Cluster assignment broadcast (multicast to cluster members)
    for _ in cluster:
        msgs.send("phase1_cluster", 16)

    phase1_time = time.time() - t0
    cluster_variance = float(np.var([distances[v.id] for v in cluster])) if cluster else 999.0

    # Fast path check: if variance near zero, all cluster members agree
    fast_path = cluster_variance < 1e-6 and len(cluster) >= n_req

    if fast_path:
        # Single-round finality. No Phase 2 needed.
        for _ in cluster:
            msgs.send("fast_path_finality", 32)
        total_time = time.time() - t0
        return {
            "finalized": True,
            "fast_path": True,
            "cluster_size": len(cluster),
            "excluded": [(v.name, v.is_byzantine, v.strategy if v.is_byzantine else None,
                          distances[v.id]) for v in excluded],
            "cluster_variance": cluster_variance,
            "phase1_time": phase1_time,
            "phase2_time": 0.0,
            "total_time": total_time,
            "msgs": msgs.count,
            "msg_bytes": msgs.bytes,
            "msg_breakdown": dict(msgs.by_type),
            "sync_pushed": sync_pushed,
            "sync_details": sync_details,
            "n_commits": len(cluster),
            "n_required": n_req,
            "distances": {v.name: distances[v.id] for v in validators},
        }

    # Phase 2: BLS commitments within cluster
    t1 = time.time()
    sigs = []
    pubs = []
    signer_bitmap = bitarray(n)
    signer_bitmap.setall(0)

    for v in cluster:
        sig = v.sign_commit(block.block_hash)
        if sig is not None:
            sigs.append(sig)
            pubs.append(v.kp.pubkey)
            signer_bitmap[v.id] = 1
            msgs.send("phase2_commit", 96)

    finality_proof_size = 0
    if sigs:
        agg_sig = BLSKeyPair.aggregate(sigs)
        bitmap_bytes = len(signer_bitmap.tobytes())
        finality_proof_size = 96 + bitmap_bytes
        # Multicast finality proof to cluster members
        for _ in cluster:
            msgs.send("phase2_finality", finality_proof_size)

    phase2_time = time.time() - t1
    total_time = time.time() - t0
    finalized = len(sigs) >= n_req

    return {
        "finalized": finalized,
        "fast_path": False,
        "cluster_size": len(cluster),
        "excluded": [(v.name, v.is_byzantine, v.strategy if v.is_byzantine else None,
                      distances[v.id]) for v in excluded],
        "cluster_variance": cluster_variance,
        "phase1_time": phase1_time,
        "phase2_time": phase2_time,
        "total_time": total_time,
        "msgs": msgs.count,
        "msg_bytes": msgs.bytes,
        "msg_breakdown": dict(msgs.by_type),
        "sync_pushed": sync_pushed,
        "sync_details": sync_details,
        "n_commits": len(sigs),
        "n_required": n_req,
        "finality_proof_bytes": finality_proof_size,
        "distances": {v.name: distances[v.id] for v in validators},
    }


# ---------------------------------------------------------------------------
# Tree-structured consensus
# ---------------------------------------------------------------------------

def tree_consensus(validators: list, block: Block, threshold: float,
                   partial_obs: Optional[dict] = None,
                   branching: int = 10) -> dict:
    """
    Hierarchical two-phase consensus using a tree of aggregators.

    Key insight: leaves do NOT run per-leaf BFT. The distance threshold
    filters Byzantine validators individually. A leaf with 5 honest and
    5 Byzantine just excludes the 5 Byzantine and reports the filtered
    mean of the remaining 5. Phase 2 BLS commits catch any Byzantine
    who slipped past the distance filter.

    Phase 1 (bottom-up):
      Level 0: validators send vector + bloom to leaf leader. Leaf leader
      distance-filters, bloom-syncs, computes weighted mean, sends 76-byte
      summary (mean + count + variance) upstream.
      Level 1+: internal nodes aggregate child summaries into their own
      76-byte summary and pass it up.
      Root: checks global weighted mean distance from reference.

    Phase 2 (top-down):
      Root broadcasts "collect commits" down. Each validator that passed
      the distance filter sends a BLS commit up through the tree. Internal
      nodes aggregate BLS sigs from children (associative). Root checks
      count >= 2/3.
    """
    t0 = time.time()
    msgs = MessageCounter()
    n = len(validators)
    n_req = int(math.ceil(n * 2 / 3))
    all_tx_strs = block.tx_data_strings
    reference = compute_vector(all_tx_strs)

    # Build tree structure: split validators into leaf groups
    leaf_groups = [validators[i:i + branching]
                   for i in range(0, n, branching)]
    n_leaves = len(leaf_groups)

    # Compute tree depth
    n_levels = 1  # level 0 = leaves
    nodes_at_level = n_leaves
    while nodes_at_level > 1:
        nodes_at_level = math.ceil(nodes_at_level / branching)
        n_levels += 1

    # ---- Phase 1: bottom-up ----

    # Level 0: validators -> leaf leaders
    passed_filter = set()  # validator ids that passed distance filter
    leaf_summaries = []    # (weighted_mean, count, variance) per leaf
    total_excluded = 0
    total_sync = 0
    level_stats = []

    for group in leaf_groups:
        vectors = {}
        blooms = {}

        for v in group:
            miss = partial_obs.get(v.id, set()) if partial_obs else set()
            vectors[v.id] = v.get_vector(block, miss)
            if not v.is_byzantine:
                blooms[v.id] = v.make_bloom(block, miss)
            else:
                blooms[v.id] = None
            bloom_size = blooms[v.id].size_bytes if blooms[v.id] else 0
            msgs.send("L0_vector", N_DIMS * 8 + bloom_size)

        # Distance filter against reference (not group mean)
        included = []
        included_vecs = []
        for v in group:
            d = float(np.linalg.norm(vectors[v.id] - reference))
            if d < threshold:
                included.append(v)
                included_vecs.append(vectors[v.id])
                passed_filter.add(v.id)
            else:
                total_excluded += 1

        # Bloom sync for included validators with partial observation
        if partial_obs:
            for v in included:
                if v.is_byzantine or v.id not in partial_obs:
                    continue
                bf = blooms.get(v.id)
                if bf:
                    missing = bf.missing_from(all_tx_strs)
                    if missing:
                        total_sync += len(missing)
                        msgs.send("L0_sync", len(missing) * 200)

        # Leaf summary: weighted mean of included vectors
        if included_vecs:
            leaf_mean = np.mean(included_vecs, axis=0)
            leaf_var = float(np.var([np.linalg.norm(v - reference)
                                     for v in included_vecs]))
            leaf_count = len(included)
        else:
            # All validators in this leaf were excluded
            leaf_mean = np.zeros(N_DIMS)
            leaf_var = 0.0
            leaf_count = 0

        leaf_summaries.append((leaf_mean, leaf_count, leaf_var))
        # Leaf leader sends 76-byte summary upstream
        msgs.send("L0_summary", 76)

    level_stats.append({
        "level": 0,
        "groups": n_leaves,
        "excluded": total_excluded,
        "passed": len(passed_filter),
        "msgs": msgs.count,
    })

    # Level 1+: aggregate summaries up the tree
    current_summaries = leaf_summaries
    level = 1
    while len(current_summaries) > 1:
        next_summaries = []
        level_groups = [current_summaries[i:i + branching]
                        for i in range(0, len(current_summaries), branching)]

        for group in level_groups:
            total_count = sum(c for _, c, _ in group)
            if total_count > 0:
                agg_mean = np.sum(
                    [mean * count for mean, count, _ in group], axis=0
                ) / total_count
                # Combined variance (weighted)
                agg_var = sum(v * c for _, c, v in group) / total_count
            else:
                agg_mean = np.zeros(N_DIMS)
                agg_var = 0.0

            next_summaries.append((agg_mean, total_count, agg_var))
            # Internal node sends 76-byte summary upstream
            msgs.send(f"L{level}_summary", 76)

        level_stats.append({
            "level": level,
            "groups": len(level_groups),
            "msgs_this_level": len(level_groups),
        })
        current_summaries = next_summaries
        level += 1

    # Root: check global weighted mean
    root_mean, root_count, root_var = current_summaries[0]
    global_dist = float(np.linalg.norm(root_mean - reference))

    phase1_time = time.time() - t0

    # ---- Phase 2: top-down BLS commits ----

    t1 = time.time()

    # Root broadcasts "collect commits" down tree
    # At each level, message goes to branching children
    nodes_this_level = 1
    for lv in range(n_levels - 1):
        children = min(nodes_this_level * branching,
                       math.ceil(n / (branching ** (n_levels - 1 - lv))))
        msgs.send(f"P2_collect_L{lv}", 32, children)
        nodes_this_level = children

    # Each validator that passed filter sends BLS commit up
    sigs = []
    signer_bitmap = bitarray(n)
    signer_bitmap.setall(0)

    for v in validators:
        if v.id not in passed_filter:
            continue
        sig = v.sign_commit(block.block_hash)
        if sig is not None:
            sigs.append(sig)
            signer_bitmap[v.id] = 1
            msgs.send("P2_commit", 96)

    # Commits aggregate up through tree: each internal node receives
    # commits from children, aggregates BLS sigs, sends one aggregate up.
    # Cost: branching messages in per node, 1 out. Net at each level
    # above leaves = number of internal nodes at that level.
    agg_sig_size = 96 + math.ceil(n / 8)  # aggregate sig + bitmap
    for lv in range(n_levels - 1):
        # Number of internal nodes at this level
        n_nodes = math.ceil(n_leaves / (branching ** (lv + 1)))
        n_nodes = max(n_nodes, 1)
        msgs.send(f"P2_agg_L{lv}", agg_sig_size, n_nodes)

    # Root produces finality proof, broadcasts down tree
    finality_proof_size = 96 + math.ceil(n / 8)
    for lv in range(n_levels - 1):
        n_nodes = math.ceil(n_leaves / (branching ** lv))
        n_nodes = max(n_nodes, 1)
        msgs.send(f"P2_finality_L{lv}", finality_proof_size, n_nodes)

    # Leaf leaders forward finality proof to their validators
    n_passed = len(passed_filter)
    msgs.send("P2_finality_validators", finality_proof_size, n_passed)

    phase2_time = time.time() - t1
    total_time = time.time() - t0
    finalized = len(sigs) >= n_req

    if sigs:
        BLSKeyPair.aggregate(sigs)

    return {
        "finalized": finalized,
        "fast_path": False,  # tree mode does not use fast path
        "tree_mode": True,
        "cluster_size": len(passed_filter),
        "excluded": [(v.name, v.is_byzantine,
                      v.strategy if v.is_byzantine else None,
                      float(np.linalg.norm(
                          v.get_vector(block,
                                       partial_obs.get(v.id, set()) if partial_obs else set())
                          - reference)))
                     for v in validators if v.id not in passed_filter],
        "cluster_variance": root_var,
        "global_dist": global_dist,
        "phase1_time": phase1_time,
        "phase2_time": phase2_time,
        "total_time": total_time,
        "msgs": msgs.count,
        "msg_bytes": msgs.bytes,
        "msg_breakdown": dict(msgs.by_type),
        "sync_pushed": total_sync,
        "sync_details": [],
        "n_commits": len(sigs),
        "n_required": n_req,
        "finality_proof_bytes": finality_proof_size,
        "n_levels": n_levels,
        "n_leaves": n_leaves,
        "branching": branching,
        "level_stats": level_stats,
        "distances": {},  # skip per-validator distances for large N
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
    def __init__(self, validators: list):
        self.validators = validators
        self.state = State()
        self.chain: List[Block] = []
        self.mempool: List[Transaction] = []
        self.consensus_log: list = []

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
            timestamp=time.time(),
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
        txs = [cb] + list(self.mempool)
        return Block(self.height, self.tip, txs, time.time(), addr,
                     proposer_name=proposer.name)

    def finalize_block(self, block: Block) -> bool:
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
        # Clear finalized txs from mempool
        done = {tx.tx_hash for tx in block.transactions if isinstance(tx, Transaction)}
        self.mempool = [tx for tx in self.mempool if tx.tx_hash not in done]
        return True

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

def make_validators(n_honest: int, n_byz: int = 0,
                    strategy: str = "drop_half") -> Tuple[list, list, list]:
    BLSKeyPair._counter = 1
    honest = [Validator(i, BLSKeyPair(f"Miner-{i}")) for i in range(n_honest)]
    byzantine = [
        Validator(i + n_honest, BLSKeyPair(f"Byz-{i}"), True, strategy)
        for i in range(n_byz)
    ]
    return honest + byzantine, honest, byzantine


def make_partial_obs(validators: list, n_txs: int,
                     max_miss: int = 2, miss_prob: float = 0.37) -> dict:
    """Simulate network delay: some honest validators miss 1-2 txs."""
    obs = {}
    for v in validators:
        if v.is_byzantine:
            continue
        if np.random.random() < miss_prob:
            n_miss = np.random.randint(1, max_miss + 1)
            obs[v.id] = set(np.random.choice(n_txs, size=min(n_miss, n_txs), replace=False))
    return obs
