"""
dpmh.py -- Distance-Preserving Multiset Hash (Proxima v2 primitive).

Rademacher construction (docs/v2_formal_foundations.md, Sections 1-4):

    v_h(tx) = sgn(SHA-512(domain || height || tx))  in {-1, +1}^n

The digest of a multiset T is the coordinate-wise integer sum
D_h(T) = sum v_h(tx). Two domains:

- Round domain ("PXM/round"), salted by height, n = 512, wire q = 2^16.
  Feeds clustering, readiness sensing, and sync sizing. Salting forces any
  grinding table to be rebuilt every block.
- Ledger domain ("PXM/ledger"), fixed salt, n = 512, wire q = 2^32.
  Feeds cumulative accumulators, conservation invariants, and fork
  localization, where equality binding (a SIS-shaped problem) matters and
  cross-height algebra is required.

Key properties (with d = |A delta B| the multiset symmetric difference):

- Homomorphism: D(A + B) = D(A) + D(B); D(A) - D(B) is meaningful.
- Unbiased estimator: E||D(A) - D(B)||^2 = n d, so dist2 / n estimates d
  with relative standard error <= sqrt(2/n) (6.25% at n = 512).
- Closed-form threshold: tau^2 = 3n separates honest partial observation
  (d <= 2) from fabricated state; honest exclusion probability <= exp(-n/8).
- All consensus decisions are integer comparisons (dist2 <= 3n); floats
  appear only in reporting.

Exact small-d distributions (implementation test vectors, see test_v2.py):
  d = 1: dist2 == n exactly (zero variance).
  d = 2: dist2 ~ 4 * Bin(n, 1/2), mean 2n.
  d = 3: dist2 ~ n + 8 * Bin(n, 1/4), mean 3n.
"""

import hashlib
import math

import numpy as np

# ---------------------------------------------------------------------------
# Parameter profiles
# ---------------------------------------------------------------------------

# name -> (dimension n, wire modulus q, bytes per coordinate)
PROFILES = {
    "round": (512, 2 ** 16, 2),    # default: clustering, sync, fast path
    "light": (128, 2 ** 16, 2),    # bandwidth-constrained leaves
    "ledger": (512, 2 ** 32, 4),   # accumulators: conservation, history, forks
}

N_ROUND = PROFILES["round"][0]
N_LEDGER = PROFILES["ledger"][0]

ROUND_DOMAIN = b"PXM/round"
LEDGER_DOMAIN = b"PXM/ledger"

# Multiplicity bound within a block (a block is a set); enforce at validation.
MU_MAX = 1


def wire_bytes(profile: str = "round") -> int:
    """Serialized digest size in bytes for a profile."""
    n, _q, cbytes = PROFILES[profile]
    return n * cbytes


# ---------------------------------------------------------------------------
# Per-transaction vectors
# ---------------------------------------------------------------------------

def _sign_bits(data: bytes, n: int) -> np.ndarray:
    """First n hash output bits of SHA-512(data), mapped to {-1, +1} int8.

    For n <= 512 one SHA-512 call suffices; larger n draws further blocks
    with a counter suffix.
    """
    out = bytearray()
    counter = 0
    while len(out) * 8 < n:
        h = hashlib.sha512(data + (counter.to_bytes(4, "big") if counter else b""))
        out.extend(h.digest())
        counter += 1
    bits = np.unpackbits(np.frombuffer(bytes(out), dtype=np.uint8))[:n]
    return (bits.astype(np.int8) * 2 - 1)


def tx_vector(tx_data: str, height: int = 0, domain: bytes = ROUND_DOMAIN,
              n: int = N_ROUND) -> np.ndarray:
    """Rademacher vector of one transaction in the given domain."""
    payload = domain + height.to_bytes(8, "big") + tx_data.encode()
    return _sign_bits(payload, n)


# ---------------------------------------------------------------------------
# Digests
# ---------------------------------------------------------------------------

def digest(tx_list, height: int = 0, domain: bytes = ROUND_DOMAIN,
           n: int = N_ROUND) -> np.ndarray:
    """Integer digest of a transaction multiset (int64, exact)."""
    out = np.zeros(n, dtype=np.int64)
    for tx in tx_list:
        out += tx_vector(tx, height, domain, n)
    return out


def ledger_digest(tx_list, n: int = N_LEDGER) -> np.ndarray:
    """Fixed-domain digest for accumulators (no height salt)."""
    return digest(tx_list, height=0, domain=LEDGER_DOMAIN, n=n)


def dist2(d1: np.ndarray, d2: np.ndarray) -> int:
    """Exact squared Euclidean distance between two digests (integer)."""
    delta = d1.astype(np.int64) - d2.astype(np.int64)
    return int(np.dot(delta, delta))


def est_symdiff(d1: np.ndarray, d2: np.ndarray) -> float:
    """Unbiased estimate of |A delta B| from two digests."""
    n = len(d1)
    return dist2(d1, d2) / n


def tau2(n: int = N_ROUND) -> int:
    """Closed-form squared clustering threshold (v2 replaces Monte Carlo).

    tau^2 = 3n admits honest stragglers with d <= 2 missing transactions
    (exclusion probability <= exp(-n/8), i.e. exp(-64) at n = 512) and
    excludes d >= 4 fabrications with ~4.6 sigma of margin.
    """
    return 3 * n


def tau(n: int = N_ROUND) -> float:
    """Euclidean threshold, for float-comparison call sites."""
    return math.sqrt(tau2(n))


# ---------------------------------------------------------------------------
# Wire encoding (centered mod-q representatives)
# ---------------------------------------------------------------------------

def to_wire(d: np.ndarray, profile: str = "round") -> bytes:
    """Serialize a digest as centered mod-q coordinates.

    Within a block (|T| < q/2) this is lossless. Accumulators are only
    ever subtracted and compared for equality, so wraparound in absolute
    coordinates is irrelevant (see v2_formal_foundations.md Section 4).
    """
    n, q, cbytes = PROFILES[profile]
    assert len(d) == n, "digest dimension does not match profile"
    reduced = np.mod(d.astype(object), q)  # exact arithmetic mod q
    reduced = np.array([int(x) for x in reduced], dtype=np.int64)
    dtype = {2: np.uint16, 4: np.uint32}[cbytes]
    return reduced.astype(dtype).tobytes()


def from_wire(raw: bytes, profile: str = "round") -> np.ndarray:
    """Deserialize to centered representatives in (-q/2, q/2]."""
    n, q, cbytes = PROFILES[profile]
    dtype = {2: np.uint16, 4: np.uint32}[cbytes]
    vals = np.frombuffer(raw, dtype=dtype).astype(np.int64)
    assert len(vals) == n, "wire length does not match profile"
    half = q // 2
    vals = np.where(vals > half, vals - q, vals)
    return vals


def wire_equal(d1: np.ndarray, d2: np.ndarray, profile: str = "ledger") -> bool:
    """Equality test mod q (the binding notion for accumulators)."""
    _n, q, _c = PROFILES[profile]
    return bool(np.all(np.mod(d1 - d2, q) == 0))


# ---------------------------------------------------------------------------
# Robust reference selection (v2_formal_foundations.md Section 7)
# ---------------------------------------------------------------------------

def robust_reference(digests: list) -> np.ndarray:
    """Coordinate-wise median of submitted digests.

    Breakdown point 1/2 per coordinate: with f < N/3 Byzantine, no
    coalition can move any coordinate of the reference outside the range
    of honest values. Verifiable by any validator from the signed Phase 1
    digests; removes the aggregator's own state as a manipulation point.
    """
    stack = np.stack([d.astype(np.int64) for d in digests])
    med = np.median(stack, axis=0)
    # Round to integers so downstream arithmetic stays exact.
    return np.round(med).astype(np.int64)


# ---------------------------------------------------------------------------
# Cumulative accumulators (fork localization, prune accounting)
# ---------------------------------------------------------------------------

class Accumulator:
    """Cumulative ledger-domain digest chain: P[b] = sum of block digests."""

    def __init__(self, n: int = N_LEDGER):
        self.n = n
        self.prefix = [np.zeros(n, dtype=np.int64)]  # P[0] = 0 (genesis)

    def append_block(self, tx_list) -> None:
        block_d = ledger_digest(tx_list, self.n)
        self.prefix.append(self.prefix[-1] + block_d)

    @property
    def height(self) -> int:
        return len(self.prefix) - 1

    def range_digest(self, a: int, b: int) -> np.ndarray:
        """Digest of all transactions in blocks (a, b], O(1)."""
        return self.prefix[b] - self.prefix[a]

    def head(self) -> np.ndarray:
        return self.prefix[-1]


def find_fork(acc_a: Accumulator, acc_b: Accumulator) -> dict:
    """Locate the fork point of two chains by binary search on prefixes.

    O(log H) probes, one accumulator comparison each; every probe also
    yields an estimate of how much diverged in the probed range.
    Returns the last common height, the probe count, and per-probe
    divergence estimates.
    """
    hi = min(acc_a.height, acc_b.height)
    lo = 0
    probes = 0
    divergence_trace = []
    if wire_equal(acc_a.prefix[hi], acc_b.prefix[hi]):
        return {"fork_height": hi, "probes": 1, "diverged": False,
                "trace": []}
    while lo < hi:
        mid = (lo + hi + 1) // 2
        probes += 1
        da, db = acc_a.prefix[mid], acc_b.prefix[mid]
        d_hat = est_symdiff(da, db)
        divergence_trace.append((mid, d_hat))
        if wire_equal(da, db):
            lo = mid
        else:
            hi = mid - 1
    return {"fork_height": lo, "probes": probes + 1, "diverged": True,
            "trace": divergence_trace}


def prove_prune(before: np.ndarray, after: np.ndarray, pruned_txs) -> bool:
    """Verify a prune claim: D(before) - D(after) == D(pruned set) mod q."""
    claimed = before - after
    actual = ledger_digest(pruned_txs, n=len(before))
    return wire_equal(claimed, actual, "ledger")


# ---------------------------------------------------------------------------
# Monte Carlo validation of the closed forms (not used by the protocol)
# ---------------------------------------------------------------------------

def validate_estimator(n: int = N_ROUND, d: int = 2, trials: int = 2000,
                       seed: int = 42) -> dict:
    """Empirical check that dist2/n is an unbiased estimator of d."""
    rng = np.random.default_rng(seed)
    base = [f"tx-{i}" for i in range(64)]
    ests = []
    for t in range(trials):
        salt = rng.integers(0, 2 ** 31)
        txs = [f"{s}-{salt}" for s in base]
        drop = rng.choice(len(txs), size=d, replace=False)
        sub = [tx for i, tx in enumerate(txs) if i not in set(drop.tolist())]
        d_full = digest(txs, height=t, n=n)
        d_sub = digest(sub, height=t, n=n)
        ests.append(est_symdiff(d_full, d_sub))
    ests = np.array(ests)
    return {
        "n": n, "d": d, "trials": trials,
        "mean": float(ests.mean()),
        "rse": float(ests.std() / max(ests.mean(), 1e-12)),
        "rse_bound": math.sqrt(2 * max(d - 1, 0) / (d * n)) if d else 0.0,
    }
