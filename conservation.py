"""
conservation.py -- Cross-shard conservation (double-entry bookkeeping with
homomorphic digests).

Every cross-shard transaction t from shard i to shard j, debited at block
h, contributes the ledger digest of the corridor-keyed item "i>j|t" to
i's outbound accumulator O[i,j][h] and, when the destination applies the
credit (whose message carries h), to j's inbound accumulator I[i,j][h].
Digests are dpmh.LedgerDigest: LtHash for binding plus a sketch for size
estimates.

Invariant, checked only over SETTLED cohorts (debit height <= now - W):

    sum_i O_i[<= now-W]  ==  sum_j I_j[<= now-W]

Indexing by debit height means honest settlement lag never alarms: a
cohort is examined only after its window has passed, whatever the traffic.

Threat model. Each shard's quorum attests honestly to what that shard
applied (shard-internal execution is secured by its own BFT). Faults
caught here are in the cross-shard path: dropped, misrouted, replayed, or
minted credits (relayer bugs, Byzantine relayers, bridge faults). A
Byzantine shard quorum could sign accumulators that misreport what it
applied; that requires state-validity or fraud proofs and is out of scope.

What the corridor keying buys. Without it a credit applied on the wrong
corridor (debit on i->j, credit on k->l) cancels globally and evades both
the invariant and bisection. With it, the misroute appears as a fault on
both corridors, and any cancellation inside a bisection group requires an
LtHash collision, so group tests are exact.

Cost. Each shard publishes two signed ledger digests per block: O(S)
objects, independent of transaction volume and of the corridor count C.
A per-corridor batch root is cheaper per object (32 B) but costs O(C),
which is O(S^2) on a dense corridor graph; the crossover is computed
below rather than asserted.
"""

import math
from collections import Counter, defaultdict

import dpmh
import reconcile

SIG_BYTES = 96
LEDGER_WIRE = dpmh.ledger_wire_bytes()    # 2048 B LtHash + 2048 B sketch
RECEIPT_BYTES = 32
ROOT_BYTES = 32


def corridor_item(i: int, j: int, tx: str) -> str:
    return f"{i}>{j}|{tx}"


class ShardSystem:
    def __init__(self, n_shards: int, window: int = 2):
        self.S = n_shards
        self.W = window
        self.now = 0
        self.O = defaultdict(lambda: defaultdict(dpmh.LedgerDigest))
        self.I = defaultdict(lambda: defaultdict(dpmh.LedgerDigest))
        self.debited = defaultdict(Counter)    # (i, j) -> multiset of (tx, h)
        self.credited = defaultdict(Counter)

    # -- shard-side operations ------------------------------------------

    def debit(self, i: int, j: int, tx: str) -> int:
        """Source applies the debit at the current block; returns h."""
        h = self.now
        self.O[(i, j)][h] = self.O[(i, j)][h] + dpmh.LedgerDigest.item(
            corridor_item(i, j, tx))
        self.debited[(i, j)][(tx, h)] += 1
        return h

    def credit(self, i: int, j: int, tx: str, h: int) -> None:
        """Destination applies a credit claiming corridor (i, j), height h."""
        self.I[(i, j)][h] = self.I[(i, j)][h] + dpmh.LedgerDigest.item(
            corridor_item(i, j, tx))
        self.credited[(i, j)][(tx, h)] += 1

    def advance_block(self) -> None:
        self.now += 1

    # -- digests ----------------------------------------------------------

    def _settled(self, acc: dict) -> dpmh.LedgerDigest:
        out = dpmh.LedgerDigest()
        for h, d in acc.items():
            if h <= self.now - self.W:
                out = out + d
        return out

    def pending(self, key) -> dpmh.LedgerDigest:
        """Settled O - I on a corridor: the digest of its dangling items."""
        return self._settled(self.O[key]) - self._settled(self.I[key])

    def corridors(self) -> list:
        return sorted(set(self.O) | set(self.I))

    # -- beacon-side checks ----------------------------------------------

    def global_invariant(self) -> dict:
        """Compare the sum of all settled outbound vs inbound digests.

        The beacon receives per-shard aggregates (2 signed objects per
        shard); summing per corridor here is the same arithmetic.
        """
        total = dpmh.LedgerDigest()
        for key in self.corridors():
            total = total + self.pending(key)
        return {"holds": total.is_zero(),
                "dangling_estimate": total.est(),
                "verification_bytes": conservation_verification_bytes(self.S)}

    def aged_alarms(self) -> list:
        """Corridors with a nonzero settled pending digest."""
        return [k for k in self.corridors() if not self.pending(k).is_zero()]

    def localize(self) -> dict:
        """Adaptive group testing over corridors (bisection).

        Each probe asks the involved shards for one aggregated settled
        pending digest over a group of corridors. With corridor-keyed
        items, a group sums to zero iff every corridor in it is clean
        (else an LtHash collision was found), so recursion is exact. Cost
        is at most 2 f ceil(log2(C / f)) + 1 probes for f faulty corridors.
        When f is a large fraction of C, testing each corridor directly
        (C probes) is cheaper; the beacon cannot know f in advance, so both
        counts are reported rather than picking the winner in hindsight.
        """
        keys = self.corridors()
        probes = 0
        faulty = []

        def zero(group):
            nonlocal probes
            probes += 1
            tot = dpmh.LedgerDigest()
            for k in group:
                tot = tot + self.pending(k)
            return tot.is_zero()

        def bisect(group):
            if zero(group):
                return
            if len(group) == 1:
                faulty.append(group[0])
                return
            mid = len(group) // 2
            bisect(group[:mid])
            bisect(group[mid:])

        if keys:
            bisect(keys)
        return {"faulty": faulty, "probes": probes,
                "individual_probes": len(keys),
                "probe_bytes": probes * (LEDGER_WIRE + SIG_BYTES)}

    def decode_corridor(self, key, session: int = 0) -> dict:
        """Exact dangling items on a corridor, multiset-aware.

        IBLT over (tx, h) short ids of settled debits vs settled credits,
        with multiplicities: a replayed credit (multiplicity 2) leaves a
        count of -1 and decodes as a minted extra. Sized by the sketch
        estimate of the pending digest.
        """
        settled = lambda c: Counter({x: n for x, n in c.items()
                                     if x[1] <= self.now - self.W})
        deb, cred = settled(self.debited[key]), settled(self.credited[key])
        d_hat = max(1.0, self.pending(key).est())
        m = reconcile.size_for(d_hat)
        for attempt in range(8):
            sess = session * 1009 + attempt
            sid = lambda x: reconcile._key64(f"{x[0]}@{x[1]}", sess)
            t_d, t_c = reconcile.IBLT(m, attempt), reconcile.IBLT(m, attempt)
            for x, mult in deb.items():
                for _ in range(mult):
                    t_d.insert_key(sid(x))
            for x, mult in cred.items():
                for _ in range(mult):
                    t_c.insert_key(sid(x))
            a, b, ok = t_d.subtract(t_c).peel()
            if ok:
                ids = {sid(x): x for x in set(deb) | set(cred)}
                return {"dropped": {ids[k] for k in a},
                        "minted": {ids[k] for k in b},
                        "ok": True, "d_hat": d_hat,
                        "bytes": (attempt + 1) * m * reconcile.CELL_BYTES}
            m *= 2
        return {"dropped": set(), "minted": set(), "ok": False,
                "d_hat": d_hat, "bytes": 0}


# ---------------------------------------------------------------------------
# Verification-metadata models (bytes per block at the verifier)
# ---------------------------------------------------------------------------

def conservation_verification_bytes(n_shards: int) -> int:
    """2 signed ledger digests per shard; independent of volume and C."""
    return n_shards * 2 * (LEDGER_WIRE + SIG_BYTES)


def corridor_root_verification_bytes(n_corridors: int) -> int:
    """Each corridor's source and destination publish a signed batch root;
    the verifier compares them pairwise. Volume-independent, O(C)."""
    return n_corridors * 2 * (ROOT_BYTES + SIG_BYTES)


def receipts_verification_bytes(txs_per_block: int, n_corridors: int) -> int:
    """Batched receipts: per-corridor signed root plus a 32 B receipt per
    transaction checked at the destination."""
    return n_corridors * (ROOT_BYTES + SIG_BYTES) + txs_per_block * RECEIPT_BYTES


def crossover_shards_full_mesh() -> int:
    """Smallest S where conservation beats per-corridor roots on a full
    mesh (C = S(S-1))."""
    s = 2
    while conservation_verification_bytes(s) >= corridor_root_verification_bytes(
            s * (s - 1)):
        s += 1
    return s


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def simulate(n_shards: int = 16, blocks: int = 8, txs_per_corridor: int = 40,
             topology: str = "ring", lag: int = 1, window: int = 2,
             faults: dict = None, seed: int = 42) -> dict:
    """Run S shards with credits landing `lag` blocks after debits.

    faults: corridor -> list of (kind, k) with kind in
      "drop"     credit for the k-th tx of block 1 never lands
      "replay"   credit for the k-th tx of block 1 lands twice
      "mint"     a credit with no debit lands (claims height 1)
      "misroute" the k-th tx of block 1 is credited on another corridor
    Faults are injected in block 1, so they are settled by the end of the
    run while the last `lag` blocks remain legitimately in flight.
    """
    import random
    rnd = random.Random(seed)
    faults = faults or {}
    sysm = ShardSystem(n_shards, window=window)
    if topology == "ring":
        corr = [(i, (i + 1) % n_shards) for i in range(n_shards)]
    else:
        corr = [(i, j) for i in range(n_shards) for j in range(n_shards) if i != j]
    other = {c: corr[(corr.index(c) + 1) % len(corr)] for c in corr}
    inflight = []                       # (land_at, i, j, tx, h, times)
    truth = defaultdict(lambda: {"dropped": set(), "minted": set()})
    total = 0
    for b in range(blocks):
        for (i, j) in corr:
            kinds = dict((k, kind) for kind, k in faults.get((i, j), []))
            for k in range(txs_per_corridor):
                tx = f"b{b}-{i}>{j}-{k}"
                total += 1
                h = sysm.debit(i, j, tx)
                kind = kinds.get(k) if b == 1 else None
                if kind == "drop":
                    truth[(i, j)]["dropped"].add((tx, h))
                    continue
                if kind == "misroute":
                    oi, oj = other[(i, j)]
                    inflight.append((b + lag, oi, oj, tx, h, 1))
                    truth[(i, j)]["dropped"].add((tx, h))
                    truth[(oi, oj)]["minted"].add((tx, h))
                    continue
                times = 2 if kind == "replay" else 1
                if kind == "replay":
                    truth[(i, j)]["minted"].add((tx, h))
                inflight.append((b + lag, i, j, tx, h, times))
            if b == 1:
                for kind, k in faults.get((i, j), []):
                    if kind == "mint":
                        tx = f"MINT-{i}>{j}-{k}"
                        inflight.append((b + lag, i, j, tx, 1, 1))
                        truth[(i, j)]["minted"].add((tx, 1))
        rnd.shuffle(inflight)
        still = []
        for item in inflight:
            land, i, j, tx, h, times = item
            if land <= b:
                for _ in range(times):
                    sysm.credit(i, j, tx, h)
            else:
                still.append(item)
        inflight = still
        sysm.advance_block()

    check = sysm.global_invariant()
    alarms = sysm.aged_alarms()
    loc = sysm.localize()
    decoded = {k: sysm.decode_corridor(k) for k in loc["faulty"]}
    truly_faulty = sorted(k for k, t in truth.items() if t["dropped"] or t["minted"])
    decode_exact = all(decoded[k]["ok"]
                       and decoded[k]["dropped"] == truth[k]["dropped"]
                       and decoded[k]["minted"] == truth[k]["minted"]
                       for k in truly_faulty if k in decoded)
    n_faults = sum(len(t["dropped"]) + len(t["minted"]) for t in truth.values())
    return {
        "shards": n_shards, "corridors": len(corr), "blocks": blocks,
        "total_txs": total, "in_flight_at_check": len(inflight),
        "faults_injected": n_faults,
        "invariant_violated": not check["holds"],
        "dangling_estimate": check["dangling_estimate"],
        "aged_alarms": sorted(alarms),
        "truly_faulty": truly_faulty,
        "alarms_exact": sorted(alarms) == truly_faulty,
        "localized": sorted(loc["faulty"]),
        "localization_exact": sorted(loc["faulty"]) == truly_faulty,
        "probes": loc["probes"],
        "decode_exact": decode_exact and set(decoded) == set(truly_faulty),
        "verify_bytes": {
            "conservation": conservation_verification_bytes(n_shards),
            "corridor_roots": corridor_root_verification_bytes(len(corr)),
            "receipts": receipts_verification_bytes(
                total // blocks, len(corr)),
        },
    }


if __name__ == "__main__":
    f = {(0, 1): [("drop", 0), ("drop", 1)], (3, 4): [("replay", 2)],
         (7, 8): [("mint", 0)], (10, 11): [("misroute", 5)]}
    r = simulate(faults=f)
    for k in ("total_txs", "in_flight_at_check", "faults_injected",
              "invariant_violated", "dangling_estimate", "truly_faulty",
              "alarms_exact", "localization_exact", "probes", "decode_exact",
              "verify_bytes"):
        print(f"{k}: {r[k]}")
    print("full-mesh crossover vs per-corridor roots: S >=",
          crossover_shards_full_mesh())
