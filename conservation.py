"""
conservation.py -- Cross-shard conservation for Proxima v2.

Double-entry bookkeeping in vector space (docs/v2_formal_foundations.md,
Section 8). Every cross-shard transaction t from shard i to shard j
contributes +v(t) (ledger domain) to i's outbound corridor accumulator
O[i][j] when the debit applies and +v(t) to j's inbound accumulator
I[i][j] when the credit applies. The system invariant is

    sum_i O_i  ==  sum_j I_j   (mod q, over a settlement window)

so any minted, lost, or replayed cross-shard effect breaks a signed,
collision-binding equality. Checking is O(S) kilobyte-scale objects per
block instead of O(S^2) pair relations or O(volume) receipts.

Refinements implemented here (Sections 8 and 14 of the foundations doc):

- Pending-set digests: P_ij = O_ij - I_ij is itself the ledger digest of
  the in-flight transactions on corridor (i, j); its norm meters in-flight
  volume continuously, and the invariant is enforced as "P_ij decays to
  zero within W blocks" (credits necessarily lag debits), with per-corridor
  aging alarms rather than per-block equality.
- Group-testing localization: on alarm, the beacon bisects the corridor
  set with aggregated digests, localizing f faulty corridors in
  O(f log C) probes; bisection over a linear measurement is exact.
- Corridor syndromes: each corridor also carries an IBLT of its in-flight
  set (both objects are linear), so an alarmed corridor decodes exactly
  the dangling transactions without a further exchange.
- Global ledger digest G and verifiable prune accounting come from
  dpmh.Accumulator / dpmh.prove_prune.

Semantics: this is detection and settlement verification, not locking.
Pair with optimistic execution + revert window, or credit-on-match.
"""

import math
from collections import defaultdict

import numpy as np

import dpmh
import reconcile

SIG_BYTES = 96          # BLS aggregate per shard attestation
LEDGER_WIRE = dpmh.wire_bytes("ledger")   # 2048 B per accumulator
RECEIPT_BYTES = 32      # per-tx receipt hash in the batched-receipt baseline
ROOT_BYTES = 32         # per-corridor batch commitment in baselines


class ShardSystem:
    """S shards, corridor accumulators, beacon-side conservation checks."""

    def __init__(self, n_shards: int, window: int = 2, n: int = dpmh.N_LEDGER):
        self.S = n_shards
        self.W = window
        self.n = n
        z = lambda: np.zeros(n, dtype=np.int64)
        self.O = defaultdict(z)     # (i, j) -> outbound accumulator
        self.I = defaultdict(z)     # (i, j) -> inbound accumulator
        self.debited = defaultdict(set)     # (i, j) -> txs the source applied
        self.credited = defaultdict(set)    # (i, j) -> txs the dest applied
        self.pending_since = {}     # (i, j) -> block when pending went nonzero
        self.block = 0

    # -- shard-side operations ------------------------------------------

    def debit(self, i: int, j: int, tx: str) -> None:
        self.O[(i, j)] += dpmh.tx_vector(tx, 0, dpmh.LEDGER_DOMAIN, self.n)
        self.debited[(i, j)].add(tx)

    def credit(self, i: int, j: int, tx: str) -> None:
        self.I[(i, j)] += dpmh.tx_vector(tx, 0, dpmh.LEDGER_DOMAIN, self.n)
        self.credited[(i, j)].add(tx)

    def pending(self, i: int, j: int) -> np.ndarray:
        """P_ij = O_ij - I_ij: the ledger digest of the in-flight set."""
        return self.O[(i, j)] - self.I[(i, j)]

    def corridor_syndrome(self, i: int, j: int, d_hint: float) -> reconcile.IBLT:
        """IBLT of the applied-debit view, sized by the pending norm."""
        m = reconcile.size_for(d_hint)
        return reconcile.IBLT.of(self.debited[(i, j)], m)

    # -- beacon-side checks ----------------------------------------------

    def advance_block(self) -> None:
        """End-of-block bookkeeping: age nonzero pending corridors."""
        self.block += 1
        for key in set(list(self.O.keys()) + list(self.I.keys())):
            p = self.O[key] - self.I[key]
            if np.any(p != 0):
                self.pending_since.setdefault(key, self.block)
            else:
                self.pending_since.pop(key, None)

    def global_invariant(self) -> dict:
        """Check sum_i O_i == sum_j I_j; violation norm estimates danglers.

        Verification cost per block: each shard publishes its aggregated
        outbound and inbound accumulators, quorum-signed.
        """
        total_o = np.zeros(self.n, dtype=np.int64)
        total_i = np.zeros(self.n, dtype=np.int64)
        for key, acc in self.O.items():
            total_o += acc
        for key, acc in self.I.items():
            total_i += acc
        delta = total_o - total_i
        holds = bool(np.all(delta == 0))
        bytes_per_block = self.S * 2 * (LEDGER_WIRE + SIG_BYTES)
        return {
            "holds": holds,
            "dangling_estimate": dpmh.est_symdiff(total_o, total_i),
            "verification_bytes": bytes_per_block,
        }

    def aged_alarms(self) -> list:
        """Corridors whose pending digest has persisted beyond the window.

        Ordinary settlement lag (credit lands within W blocks of the debit)
        never alarms; only stuck or faulty corridors do.
        """
        return [key for key, since in self.pending_since.items()
                if self.block - since >= self.W]

    def localize(self, expected_inflight: dict = None) -> dict:
        """Group-testing localization of faulty corridors.

        The beacon holds only per-shard aggregates; on a global-invariant
        failure it requests digests aggregated over halves of the corridor
        list, recursing into imbalanced halves. Bisection over a linear
        measurement is exact. Each probe costs one signed accumulator.

        expected_inflight: corridor -> set of txs legitimately in flight
        (from the current window); a corridor is faulty if its pending
        digest differs from the digest of its legitimate in-flight set.
        """
        expected_inflight = expected_inflight or {}
        corridors = sorted(set(list(self.O.keys()) + list(self.I.keys())))

        def corridor_fault(key) -> bool:
            expect = dpmh.ledger_digest(
                sorted(expected_inflight.get(key, set())), self.n)
            return not dpmh.wire_equal(self.pending(*key), expect, "ledger")

        def agg_fault(keys) -> bool:
            actual = np.zeros(self.n, dtype=np.int64)
            expect = np.zeros(self.n, dtype=np.int64)
            for key in keys:
                actual += self.pending(*key)
                expect += dpmh.ledger_digest(
                    sorted(expected_inflight.get(key, set())), self.n)
            return not dpmh.wire_equal(actual, expect, "ledger")

        probes = 0
        faulty = []

        def bisect(keys):
            nonlocal probes
            probes += 1
            if not agg_fault(keys):
                return
            if len(keys) == 1:
                faulty.append(keys[0])
                return
            mid = len(keys) // 2
            bisect(keys[:mid])
            bisect(keys[mid:])

        if corridors:
            bisect(corridors)
        return {
            "faulty": faulty,
            "probes": probes,
            "probe_bytes": probes * (LEDGER_WIRE + SIG_BYTES),
            "bound": (len(faulty) or 1) * max(
                1, math.ceil(math.log2(max(len(corridors), 2)))) * 2,
        }

    def decode_corridor(self, key, expected_inflight: set = None) -> dict:
        """Identify the exact dangling txs on an alarmed corridor.

        Two-syndrome step: the pending digest norm sizes an IBLT exchange
        between the source's applied-debit set and the destination's
        applied-credit set. Their symmetric difference is exactly the
        dropped credits (debited, never credited) plus the minted credits
        (credited, never debited); peeling recovers both.
        """
        d_hat = max(1.0, dpmh.est_symdiff(self.O[key], self.I[key]))
        res = reconcile.reconcile(self.debited[key], self.credited[key],
                                  d_hint=d_hat)
        return {"dangling": res["only_a"] | res["only_b"],
                "dropped": res["only_a"], "minted": res["only_b"],
                "ok": res["ok"], "bytes": res["bytes"], "d_hat": d_hat}


# ---------------------------------------------------------------------------
# Baseline verification-metadata models (per block)
# ---------------------------------------------------------------------------

def receipts_verification_bytes(txs_per_block: int, corridors: int) -> int:
    """Batched receipts: per-corridor batch root + per-tx receipt data."""
    return corridors * (ROOT_BYTES + SIG_BYTES) + txs_per_block * RECEIPT_BYTES


def two_pc_verification_bytes(txs_per_block: int, corridors: int) -> int:
    """Batched 2PC: prepare/commit batch certificates + per-tx votes."""
    return corridors * 2 * (ROOT_BYTES + SIG_BYTES) + txs_per_block * 2 * RECEIPT_BYTES


def conservation_verification_bytes(n_shards: int) -> int:
    """Conservation: 2 signed accumulators per shard, volume-independent."""
    return n_shards * 2 * (LEDGER_WIRE + SIG_BYTES)


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def simulate(n_shards: int = 16, blocks: int = 6, txs_per_corridor: int = 150,
             n_faulty_corridors: int = 2, faults_per_corridor: int = 4,
             seed: int = 42, window: int = 2) -> dict:
    """Ring-of-shards run with faults injected into a few chosen corridors.

    Faults: dropped credits (destination never applies) and minted credits
    (destination applies a tx no source debited). The run extends past the
    settlement window, so at check time every legitimate transaction has
    settled and the expected in-flight set is empty: any nonzero pending
    corridor is faulty. Measures detection, O(f log C) localization,
    exact decode of the dangling transactions, and verification bytes per
    block against the batched baselines.
    """
    rng = np.random.default_rng(seed)
    sys = ShardSystem(n_shards, window=window)
    faulty_set = set()
    while len(faulty_set) < n_faulty_corridors:
        i = int(rng.integers(0, n_shards))
        faulty_set.add((i, (i + 1) % n_shards))
    injected = defaultdict(set)     # corridor -> dangling txs

    total_txs = 0
    for b in range(blocks):
        for i in range(n_shards):
            j = (i + 1) % n_shards
            for k in range(txs_per_corridor):
                tx = f"blk{b}-c{i}-{k}"
                total_txs += 1
                sys.debit(i, j, tx)
                drop = ((i, j) in faulty_set and b == 1
                        and k < faults_per_corridor)
                if drop:
                    injected[(i, j)].add(tx)   # credit never lands
                else:
                    sys.credit(i, j, tx)
        # a minted credit on one faulty corridor (no matching debit)
        if b == 2 and faulty_set:
            i, j = sorted(faulty_set)[0]
            tx = f"MINT-{b}-{i}"
            sys.credit(i, j, tx)
            injected[(i, j)].add(tx)
        sys.advance_block()

    check = sys.global_invariant()
    faults_injected = sum(len(s) for s in injected.values())
    alarms = sys.aged_alarms()

    # Post-window: nothing should still be in flight.
    expected_empty = {}
    loc = sys.localize(expected_empty) if not check["holds"] else {
        "faulty": [], "probes": 0, "probe_bytes": 0, "bound": 0}

    decode_ok = True
    decoded_danglers = set()
    for key in loc["faulty"]:
        d = sys.decode_corridor(key, set())
        decode_ok = decode_ok and d["ok"]
        decoded_danglers |= d["dangling"]

    truly_dangling = set()
    for s in injected.values():
        truly_dangling |= s

    corridors = n_shards  # ring
    txs_per_block = total_txs // blocks
    return {
        "shards": n_shards,
        "blocks": blocks,
        "total_txs": total_txs,
        "faults_injected": faults_injected,
        "invariant_violated": not check["holds"],
        "detected": (not check["holds"]) == (faults_injected > 0),
        "dangling_estimate": check["dangling_estimate"],
        "aged_alarms": sorted(alarms),
        "localized": sorted(loc["faulty"]),
        "truly_faulty": sorted(injected.keys()),
        "localization_exact": set(loc["faulty"]) == set(injected.keys()),
        "probes": loc["probes"],
        "probe_bound": loc["bound"],
        "decode_ok": decode_ok,
        "decode_exact": decoded_danglers == truly_dangling,
        "verify_bytes": {
            "conservation": conservation_verification_bytes(n_shards),
            "receipts": receipts_verification_bytes(txs_per_block, corridors),
            "2pc": two_pc_verification_bytes(txs_per_block, corridors),
        },
        "receipts_crossover_txs": (
            conservation_verification_bytes(n_shards)
            - corridors * (ROOT_BYTES + SIG_BYTES)) // RECEIPT_BYTES,
    }


if __name__ == "__main__":
    r = simulate()
    print(f"shards={r['shards']} blocks={r['blocks']} txs={r['total_txs']} "
          f"faults={r['faults_injected']} on corridors {r['truly_faulty']}")
    print(f"invariant violated: {r['invariant_violated']} "
          f"(dangling estimate {r['dangling_estimate']:.1f}, "
          f"true {r['faults_injected']})")
    print(f"aged alarms: {r['aged_alarms']}")
    print(f"localized {r['localized']} in {r['probes']} probes "
          f"(bound ~{r['probe_bound']}); exact={r['localization_exact']}")
    print(f"decode ok={r['decode_ok']} exact={r['decode_exact']}")
    vb = r["verify_bytes"]
    print(f"verification bytes/block: conservation={vb['conservation']:,} "
          f"(volume-independent), receipts={vb['receipts']:,}, "
          f"2pc={vb['2pc']:,}")
    print(f"receipts overtake conservation above "
          f"~{r['receipts_crossover_txs']:,} cross-shard txs/block")
