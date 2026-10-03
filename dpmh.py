"""
dpmh.py -- Distance-preserving multiset digests (Proxima v2 primitive).

Two objects, each used only for the property it actually has.

1. Rademacher sketch (estimation). Each item maps to

       v_s(x) = sgn(SHA-512(domain || salt || x))  in {-1, +1}^n,  n = 512,

   and a set digests to the coordinate-wise integer sum. This is the AMS
   tug-of-war sketch with random-oracle signs: for two SETS A, B fixed
   before the salt is revealed, ||D(A) - D(B)||^2 / n is an unbiased
   estimate of d = |A delta B| with relative standard error <= sqrt(2/n).
   For multisets with multiplicity difference c it estimates ||c||_2^2,
   not ||c||_1. The sketch is NOT binding: a +-1 matrix has no modulus to
   hide behind (|coordinate| <= item count, so mod-q and integer collisions
   coincide), and greedy sign balancing lets an adversary who knows the
   salt make a large difference read small at polynomial cost (see
   adversarial_balance below). The sketch therefore feeds only advisory
   decisions: scheduling, sync sizing, monitoring.

2. LtHash accumulator (binding). Each item maps to a uniform vector in
   Z_q^1024, q = 2^16, expanded from SHAKE-256 (Bellare-Micciancio LtHash
   with the parameters of Lewi et al., ePrint 2019/227, whose analysis
   applies verbatim to sets). Equality of accumulators is the binding
   notion behind conservation, prefix/range digests, fork localization,
   and prune accounting. Uniform coordinates destroy distance, which is
   why the sketch rides alongside.

Salts. The round domain is salted with the previous block hash and height
(round_salt), which nobody can predict before the previous block is final,
so grinding tables cannot be built ahead of time. The stronger pattern for
adversarial settings is commit-then-salt: parties commit to their sets,
then a salt (beacon or next-block randomness) is revealed and only then
are sketches computed; the sets are then fixed before the oracle is known
and the unbiasedness lemma holds against any adversary.

All consensus-facing arithmetic is integer; floats appear only in reports.

Exact small-d distributions of dist2 for sets (test vectors, test_v2.py):
  d = 1: dist2 == n exactly.
  d = 2: dist2 ~ 4 Bin(n, 1/2), mean 2n.
  d = 3: dist2 ~ n + 8 Bin(n, 1/4), mean 3n.
"""

import hashlib
import math

import numpy as np

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

# Sketch profiles: name -> (dimension n, wire modulus q, bytes per coordinate)
PROFILES = {
    "round": (512, 2 ** 16, 2),    # per-round sketch: sensing, sync sizing
    "light": (128, 2 ** 16, 2),    # bandwidth-constrained profile
    "ledger": (512, 2 ** 32, 4),   # cumulative sketch riding on accumulators
}

N_ROUND = PROFILES["round"][0]
N_LEDGER = PROFILES["ledger"][0]

# LtHash parameters (Lewi, Kim, Maykov, Weis 2019): n = 1024 lanes of 16 bits.
LT_N = 1024
LT_Q = 2 ** 16
LT_BYTES = LT_N * 2

ROUND_DOMAIN = b"PXM/round"
LEDGER_DOMAIN = b"PXM/ledger"
LT_DOMAIN = b"PXM/lthash"

# Multiplicity bound within a block: a block is a set.
MU_MAX = 1


def wire_bytes(profile: str = "round") -> int:
    """Serialized sketch size in bytes for a profile."""
    n, _q, cbytes = PROFILES[profile]
    return n * cbytes


def ledger_wire_bytes() -> int:
    """One ledger digest on the wire: LtHash (binding) + sketch (estimate)."""
    return LT_BYTES + wire_bytes("ledger")


def assert_set(items) -> None:
    """Enforce MU_MAX = 1: a block or batch may not repeat an item."""
    items = list(items)
    if len(items) != len(set(items)):
        raise ValueError("multiplicity bound violated: duplicate item")


# ---------------------------------------------------------------------------
# Salts
# ---------------------------------------------------------------------------

def _salt_bytes(salt) -> bytes:
    if isinstance(salt, (bytes, bytearray)):
        return bytes(salt)
    return int(salt).to_bytes(8, "big")


def round_salt(height: int, prev_hash: str) -> bytes:
    """Unpredictable per-round salt: height plus the previous block hash."""
    return height.to_bytes(8, "big") + bytes.fromhex(prev_hash)


# ---------------------------------------------------------------------------
# Rademacher sketch
# ---------------------------------------------------------------------------

def _sign_bits(data: bytes, n: int) -> np.ndarray:
    """First n bits of SHA-512(data) (counter-extended), as {-1,+1} int8."""
    out = bytearray()
    counter = 0
    while len(out) * 8 < n:
        h = hashlib.sha512(data + (counter.to_bytes(4, "big") if counter else b""))
        out.extend(h.digest())
        counter += 1
    bits = np.unpackbits(np.frombuffer(bytes(out), dtype=np.uint8))[:n]
    return bits.astype(np.int8) * 2 - 1


def tx_vector(tx_data: str, salt=0, domain: bytes = ROUND_DOMAIN,
              n: int = N_ROUND) -> np.ndarray:
    """Rademacher vector of one item. salt: bytes, or an int (height)."""
    return _sign_bits(domain + _salt_bytes(salt) + tx_data.encode(), n)


def digest(tx_list, salt=0, domain: bytes = ROUND_DOMAIN,
           n: int = N_ROUND) -> np.ndarray:
    """Integer sketch of a multiset (int64, exact)."""
    out = np.zeros(n, dtype=np.int64)
    for tx in tx_list:
        out += tx_vector(tx, salt, domain, n)
    return out


def dist2(d1: np.ndarray, d2: np.ndarray) -> int:
    """Exact squared Euclidean distance between two sketches (integer)."""
    delta = d1.astype(np.int64) - d2.astype(np.int64)
    return int(np.dot(delta, delta))


def est_symdiff(d1: np.ndarray, d2: np.ndarray) -> float:
    """Estimate of |A delta B| (sets) or ||m_A - m_B||_2^2 (multisets)."""
    return dist2(d1, d2) / len(d1)


def tau2(n: int = N_ROUND) -> int:
    """Squared straggler threshold: d <= 2 admitted, d >= 4 flagged.

    Exact tails at n = 512: an honest d = 2 view exceeds 3n with
    probability Pr[Bin(512, 1/2) > 384] = 1.6e-31 (Hoeffding: e^{-n/8});
    an honest-but-divergent d = 4 view stays below it with probability
    4.9e-7; d = 3 is genuinely ambiguous (0.52). The threshold classifies
    honest divergence only: a Byzantine party can submit any vector.
    """
    return 3 * n


def tau(n: int = N_ROUND) -> float:
    return math.sqrt(tau2(n))


# ---------------------------------------------------------------------------
# Wire encoding of sketches (centered mod-q representatives)
# ---------------------------------------------------------------------------

def to_wire(d: np.ndarray, profile: str = "round") -> bytes:
    n, q, cbytes = PROFILES[profile]
    assert len(d) == n, "digest dimension does not match profile"
    reduced = np.array([int(x) % q for x in d.tolist()], dtype=np.int64)
    dtype = {2: np.uint16, 4: np.uint32}[cbytes]
    return reduced.astype(dtype).tobytes()


def from_wire(raw: bytes, profile: str = "round") -> np.ndarray:
    """Deserialize to centered representatives in (-q/2, q/2]."""
    n, q, cbytes = PROFILES[profile]
    dtype = {2: np.uint16, 4: np.uint32}[cbytes]
    vals = np.frombuffer(raw, dtype=dtype).astype(np.int64)
    assert len(vals) == n, "wire length does not match profile"
    return np.where(vals > q // 2, vals - q, vals)


def centered_diff(a: np.ndarray, b: np.ndarray, q: int) -> np.ndarray:
    """(a - b) mod q in centered form. Use before dist2 on wire-decoded
    absolute values, which may have wrapped even when their difference
    has not."""
    d = np.mod(a.astype(np.int64) - b.astype(np.int64), q)
    return np.where(d > q // 2, d - q, d)


# ---------------------------------------------------------------------------
# LtHash (binding accumulator)
# ---------------------------------------------------------------------------

def lthash_vector(item: str, domain: bytes = LT_DOMAIN) -> np.ndarray:
    """Uniform vector in Z_q^1024 from SHAKE-256 (2 bytes per lane)."""
    raw = hashlib.shake_256(domain + item.encode()).digest(LT_BYTES)
    return np.frombuffer(raw, dtype="<u2").astype(np.int64)


def lthash(items, domain: bytes = LT_DOMAIN) -> np.ndarray:
    out = np.zeros(LT_N, dtype=np.int64)
    for it in items:
        out += lthash_vector(it, domain)
    return np.mod(out, LT_Q)


def lt_equal(a: np.ndarray, b: np.ndarray) -> bool:
    return bool(np.all(np.mod(a - b, LT_Q) == 0))


class LedgerDigest:
    """Binding LtHash plus an estimation sketch over the same multiset.

    Both components are linear, so digests add, subtract, and aggregate
    across shards. Equality is decided by the LtHash component only;
    est() reports the sketch estimate of how much two digests differ.
    """

    __slots__ = ("lt", "sk")

    def __init__(self, lt=None, sk=None):
        self.lt = np.zeros(LT_N, dtype=np.int64) if lt is None else lt
        self.sk = np.zeros(N_LEDGER, dtype=np.int64) if sk is None else sk

    @classmethod
    def of(cls, items, domain: bytes = LEDGER_DOMAIN) -> "LedgerDigest":
        items = list(items)
        return cls(lthash(items, domain + b"/lt"),
                   digest(items, 0, domain, N_LEDGER))

    @classmethod
    def item(cls, x: str, domain: bytes = LEDGER_DOMAIN) -> "LedgerDigest":
        return cls(lthash_vector(x, domain + b"/lt"),
                   tx_vector(x, 0, domain, N_LEDGER).astype(np.int64))

    def __add__(self, o):
        return LedgerDigest(np.mod(self.lt + o.lt, LT_Q), self.sk + o.sk)

    def __sub__(self, o):
        return LedgerDigest(np.mod(self.lt - o.lt, LT_Q), self.sk - o.sk)

    def __eq__(self, o):
        return lt_equal(self.lt, o.lt)

    def is_zero(self) -> bool:
        return bool(np.all(np.mod(self.lt, LT_Q) == 0))

    def est(self, o=None) -> float:
        """Sketch estimate of the size of the difference (self - o)."""
        sk = self.sk if o is None else self.sk - o.sk
        return float(np.dot(sk, sk)) / N_LEDGER

    def copy(self):
        return LedgerDigest(self.lt.copy(), self.sk.copy())


# ---------------------------------------------------------------------------
# Robust location (pre-proposal mempool reference)
# ---------------------------------------------------------------------------

def robust_reference(digests: list) -> np.ndarray:
    """Coordinate-wise median of a set of sketches (integer).

    Used where no proposal exists yet (mempool sensing): the median of
    gossiped sketches is a location estimate no minority coalition can
    drag outside the honest range coordinate-wise. It is not a set digest
    unless more than half of the inputs are identical, and its holder
    still chooses which inputs to count, so it is advisory. Inside
    consensus the reference is D(B) of the proposal, computable by anyone.
    Ties round half to even (numpy), deterministically.
    """
    stack = np.stack([d.astype(np.int64) for d in digests])
    return np.round(np.median(stack, axis=0)).astype(np.int64)


# ---------------------------------------------------------------------------
# Cumulative accumulators (prefix digests, fork localization, pruning)
# ---------------------------------------------------------------------------

class Accumulator:
    """P[b] = sum of block ledger digests; range digests in O(1)."""

    def __init__(self):
        self.prefix = [LedgerDigest()]

    def append_block(self, tx_list) -> None:
        tx_list = list(tx_list)
        assert_set(tx_list)
        self.prefix.append(self.prefix[-1] + LedgerDigest.of(tx_list))

    @property
    def height(self) -> int:
        return len(self.prefix) - 1

    def range_digest(self, a: int, b: int) -> LedgerDigest:
        """Digest of all transactions in blocks (a, b]."""
        return self.prefix[b] - self.prefix[a]

    def head(self) -> LedgerDigest:
        return self.prefix[-1]


def find_fork(acc_a: Accumulator, acc_b: Accumulator) -> dict:
    """Last common height by binary search on prefix equality (LtHash).

    O(log H) probes; each probe also reports the sketch estimate of how
    many transactions differ up to the probed height.
    """
    hi = min(acc_a.height, acc_b.height)
    lo = 0
    if acc_a.prefix[hi] == acc_b.prefix[hi]:
        return {"fork_height": hi, "probes": 1, "diverged": False, "trace": []}
    probes, trace = 1, []
    while lo < hi:
        mid = (lo + hi + 1) // 2
        probes += 1
        da, db = acc_a.prefix[mid], acc_b.prefix[mid]
        trace.append((mid, da.est(db)))
        if da == db:
            lo = mid
        else:
            hi = mid - 1
    return {"fork_height": lo, "probes": probes, "diverged": True,
            "trace": trace}


def prove_prune(before: LedgerDigest, after: LedgerDigest, pruned) -> bool:
    """Verify a prune claim: D(before) - D(after) == D(pruned)."""
    return (before - after) == LedgerDigest.of(pruned)


# ---------------------------------------------------------------------------
# Validation and attack demonstrations (not used by the protocol)
# ---------------------------------------------------------------------------

def validate_estimator(n: int = N_ROUND, d: int = 2, trials: int = 2000,
                       seed: int = 42) -> dict:
    """Empirical check that dist2 / n is unbiased for |A delta B|."""
    rng = np.random.default_rng(seed)
    base = [f"tx-{i}" for i in range(64)]
    ests = []
    for t in range(trials):
        salt = int(rng.integers(0, 2 ** 31))
        txs = [f"{s}-{salt}" for s in base]
        drop = set(rng.choice(len(txs), size=d, replace=False).tolist())
        sub = [tx for i, tx in enumerate(txs) if i not in drop]
        ests.append(est_symdiff(digest(txs, t, n=n), digest(sub, t, n=n)))
    ests = np.array(ests)
    return {
        "n": n, "d": d, "trials": trials,
        "mean": float(ests.mean()),
        "rse": float(ests.std() / max(ests.mean(), 1e-12)),
        "rse_bound": math.sqrt(2 * max(d - 1, 0) / (d * n)) if d else 0.0,
    }


def adversarial_balance(m: int, pool: int, salt=0, n: int = N_ROUND,
                        seed: int = 0) -> dict:
    """Greedy sign balancing: build X, Y with |X delta Y| = m whose sketches
    are close, given knowledge of the salt.

    The adversary keeps a running difference S = D(X) - D(Y). For each of
    m steps it draws `pool` fresh candidate items and puts the one (and the
    side) that most reduces ||S|| into X or Y. The final d_hat stays near a
    steady state set by n and pool, independent of m, so distance can be
    deflated without bound at polynomial cost. Measured at n = 512 for
    m = 200, 1000, 3000: pool 1 (no grinding, only the choice of side)
    plateaus near 200, pool 16 near 30, pool 256 near 14. This is why the sketch is
    advisory and why salts must be unpredictable or revealed after sets
    are committed.
    """
    rng = np.random.default_rng(seed)
    S = np.zeros(n, dtype=np.int64)
    for step in range(m):
        cands = np.stack([tx_vector(f"adv-{seed}-{step}-{k}-{int(rng.integers(1 << 30))}",
                                    salt, n=n).astype(np.int64)
                          for k in range(pool)])
        dots = cands @ S
        k = int(np.argmax(np.abs(dots)))
        S = S - np.sign(dots[k] or 1) * cands[k]
    return {"m": m, "pool": pool, "d_hat": float(np.dot(S, S)) / n,
            "honest_expectation": float(m)}
