"""
bft.py -- The base protocol Pi and the Proxima fast path, with view change.

Single-slot Byzantine agreement among n replicas tolerating f Byzantine
(n >= 3f + 1), in the two-phase locking style of PBFT / Tendermint /
HotStuff-2:

  view v:  NEW-VIEW   every replica sends the leader its lock (highest
                      prepare QC it holds) and its latest prepare vote
           PROPOSE    leader proposes select(NV) if that is determined,
                      else any value; the proposal carries the NV set
           PREPARE    replicas vote iff the proposal equals select(NV)
                      computed from the attached set (first view: any)
           PREPARE-QC n - f prepare votes; replicas lock on it
           COMMIT     replicas vote commit; n - f commit votes = decision

Proxima's speculative fast path: if a single view gathers
n_fast = ceil((n + 3f + 1) / 2) prepare votes for one value, that set is
itself a decision certificate (one voting round). This is the FaB / SBFT
fast quorum; with n = 3f + 1 it is all n replicas. To stay safe, view
change must recover a value that may have been fast-decided, so select()
also accepts "at least n_fast - 2f prepare votes for x in the highest
view that has any evidence". Proof sketch: a fast decision on x in view w
puts >= n_fast - f honest votes on x; any n - f NEW-VIEW set keeps
>= n_fast - 2f of them, while y can collect at most f (Byzantine claims)
+ (n - n_fast) (honest votes elsewhere) < n_fast - 2f claims. By
induction no honest replica votes for y != x in a later view, so later
views carry no evidence for y. A prepare QC and a fast certificate in the
same view intersect in >= n_fast - f > f honest replicas, so they agree.

Setting n_fast = n - f (a single 2f + 1 round, the v1 design) breaks this:
fuzz() finds conflicting decisions within a few hundred runs, and
unsafe_fast_path_counterexample() constructs one directly.

The simulator is deliberately adversarial: Byzantine leaders equivocate,
Byzantine replicas double-vote and lie about their votes (they cannot
forge honest votes or QCs), and a scheduler decides which messages each
party receives in each view. Safety must hold under every schedule;
liveness needs an honest leader after GST, modeled by a final synchronous
view.
"""

import math
import random
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Tuple


def max_faults(n: int) -> int:
    return (n - 1) // 3


def quorum(n: int, f: int) -> int:
    return n - f


def fast_quorum(n: int, f: int) -> int:
    return math.ceil((n + 3 * f + 1) / 2)


@dataclass(frozen=True)
class QC:
    kind: str                 # "prepare" or "commit"
    view: int
    value: str
    signers: FrozenSet[int]


@dataclass
class NewView:
    sender: int
    lock: Optional[QC]                       # genuine QC (unforgeable)
    last_vote: Optional[Tuple[int, str]]     # (view, value) claimed


@dataclass
class Replica:
    rid: int
    byzantine: bool = False
    lock: Optional[QC] = None
    last_vote: Optional[Tuple[int, str]] = None
    voted: Dict[Tuple[str, int], str] = field(default_factory=dict)
    decided: Optional[str] = None


def select_ev(nvs: List[NewView], n: int, f: int,
              n_fast: int) -> Tuple[Optional[str], int]:
    """(value, evidence view) a leader must propose, or (None, -1) if free.

    Scans views from highest to lowest; in the first view with evidence it
    returns the QC's value if a prepare QC from that view is present,
    else a value with >= n_fast - 2f claimed prepare votes in that view.
    Views with claims below the threshold carry no evidence.
    """
    qcs = [nv.lock for nv in nvs if nv.lock is not None]
    votes = [nv.last_vote for nv in nvs if nv.last_vote is not None]
    views = sorted({q.view for q in qcs} | {w for w, _ in votes}, reverse=True)
    need = n_fast - 2 * f
    for w in views:
        in_view = [q for q in qcs if q.view == w]
        if in_view:
            return in_view[0].value, w
        counts: Dict[str, int] = {}
        for vw, val in votes:
            if vw == w:
                counts[val] = counts.get(val, 0) + 1
        hits = [val for val, c in counts.items() if c >= need]
        if hits:
            return sorted(hits)[0], w
    return None, -1


def select(nvs: List[NewView], n: int, f: int, n_fast: int) -> Optional[str]:
    return select_ev(nvs, n, f, n_fast)[0]


def valid_nv_set(nvs: List[NewView], n: int, f: int, genuine: set) -> bool:
    senders = [nv.sender for nv in nvs]
    return (len(senders) >= quorum(n, f) and len(set(senders)) == len(senders)
            and all(nv.lock is None or nv.lock in genuine for nv in nvs))


class Run:
    """One adversarial execution of single-slot consensus."""

    def __init__(self, n: int, byz: set, f: int = None, n_fast: int = None,
                 seed: int = 0, values=("x", "y")):
        self.n = n
        self.f = max_faults(n) if f is None else f
        self.n_fast = fast_quorum(n, self.f) if n_fast is None else n_fast
        self.rng = random.Random(seed)
        self.reps = [Replica(i, i in byz) for i in range(n)]
        self.byz = set(byz)
        self.honest = [r for r in self.reps if not r.byzantine]
        self.genuine: set = set()
        self.values = values
        self.msgs = 0
        self.fast_decisions = 0

    # -- helpers --------------------------------------------------------

    def leader(self, v: int) -> Replica:
        return self.reps[v % self.n]

    def subset(self, pool, k_min: int, sync: bool):
        pool = list(pool)
        if sync:
            return pool
        k = self.rng.randint(min(k_min, len(pool)), len(pool))
        return self.rng.sample(pool, k)

    def _decide(self, r: Replica, value: str) -> None:
        if not r.byzantine and r.decided is None:
            r.decided = value

    # -- one view ---------------------------------------------------------

    def view(self, v: int, sync: bool = False) -> None:
        n, f, q = self.n, self.f, quorum(self.n, self.f)
        L = self.leader(v)

        # NEW-VIEW (skipped in view 0).
        nvs = []
        if v > 0:
            for r in self.reps:
                if r.byzantine:
                    # Lies about its vote; presents any genuine QC or none.
                    lock = self.rng.choice([None] + list(self.genuine))
                    claim = (self.rng.randint(0, v - 1), self.rng.choice(self.values))
                    nvs.append(NewView(r.rid, lock, claim))
                else:
                    nvs.append(NewView(r.rid, r.lock, r.last_vote))
            self.msgs += n

        # PROPOSE. Honest leader: picks a valid q-subset as the scheduler
        # delivers it. Byzantine leader: may equivocate, attaching any
        # valid NV subset it likes to each proposal.
        proposals: Dict[int, Tuple[str, List[NewView]]] = {}
        if not L.byzantine:
            got = nvs if sync or v == 0 else self.subset(nvs, q, False)
            if v > 0 and len(got) < q:
                return
            val = (select(got, n, f, self.n_fast) if v > 0 else None) \
                or self.rng.choice(self.values)
            for r in self.subset(self.reps, 0, sync):
                proposals[r.rid] = (val, got)
        else:
            for r in self.reps:
                if self.rng.random() < 0.8:
                    got = self.rng.sample(nvs, min(len(nvs), q)) if v > 0 else []
                    forced = select(got, n, f, self.n_fast) if v > 0 else None
                    proposals[r.rid] = (forced or self.rng.choice(self.values), got)
        self.msgs += len(proposals)

        # PREPARE votes.
        prep: Dict[str, set] = {}
        for r in self.reps:
            if r.byzantine:
                for val in self.values:              # double-votes
                    prep.setdefault(val, set()).add(r.rid)
                continue
            if r.rid not in proposals or ("prepare", v) in r.voted:
                continue
            val, just = proposals[r.rid]
            if v > 0:
                if not valid_nv_set(just, n, f, self.genuine):
                    continue
                sel, w = select_ev(just, n, f, self.n_fast)
                if sel not in (None, val):
                    continue
                if (r.lock is not None and val != r.lock.value
                        and w < r.lock.view):
                    continue                          # HotStuff locking rule
            r.voted[("prepare", v)] = val
            r.last_vote = (v, val)
            prep.setdefault(val, set()).add(r.rid)
            self.msgs += 1

        # Collection: whoever aggregates sees a scheduler-chosen subset.
        # The leader (honest or not) may assemble any certificate whose
        # votes it actually received.
        for val, voters in prep.items():
            seen = set(self.subset(sorted(voters), 0, sync))
            if len(seen) >= self.n_fast:
                self.fast_decisions += 1
                for r in self.subset(self.reps, 0, sync):
                    self._decide(r, val)
                self.msgs += n
            if len(seen) >= q:
                qc = QC("prepare", v, val, frozenset(seen))
                self.genuine.add(qc)
                commit = set(self.byz)
                for r in self.subset(self.reps, 0, sync):
                    self.msgs += 1
                    if r.byzantine:
                        continue
                    if r.lock is None or qc.view >= r.lock.view:
                        r.lock = qc
                    if ("commit", v) not in r.voted:
                        r.voted[("commit", v)] = val
                        commit.add(r.rid)
                        self.msgs += 1
                cseen = set(self.subset(sorted(commit), 0, sync))
                if len(cseen) >= q:
                    for r in self.subset(self.reps, 0, sync):
                        self._decide(r, val)
                    self.msgs += n

    # -- execution ----------------------------------------------------------

    def execute(self, async_views: int = 4) -> dict:
        v = 0
        for _ in range(async_views):
            self.view(v, sync=False)
            v += 1
        # GST: synchronous views until an honest leader completes.
        for _ in range(self.n + 1):
            leader_honest = not self.leader(v).byzantine
            self.view(v, sync=True)
            v += 1
            if leader_honest and all(r.decided for r in self.honest):
                break
        decided = {r.decided for r in self.honest if r.decided is not None}
        return {"safe": len(decided) <= 1,
                "live": all(r.decided for r in self.honest),
                "decided": decided, "views": v, "msgs": self.msgs,
                "fast_decisions": self.fast_decisions}


def fuzz(n: int, trials: int, n_fast: int = None, n_byz: int = None,
         seed: int = 0) -> dict:
    """Random adversarial schedules; counts safety and liveness failures."""
    f = max_faults(n)
    n_byz = f if n_byz is None else n_byz
    rng = random.Random(seed)
    unsafe = not_live = 0
    example = None
    for t in range(trials):
        byz = set(rng.sample(range(n), n_byz))
        r = Run(n, byz, f=f, n_fast=n_fast, seed=seed * 100003 + t).execute()
        if not r["safe"]:
            unsafe += 1
            example = example or {"trial": t, "byz": sorted(byz), **r}
        if not r["live"]:
            not_live += 1
    return {"n": n, "f": f, "n_fast": n_fast or fast_quorum(n, f),
            "trials": trials, "unsafe": unsafe, "not_live": not_live,
            "example": example}


def unsafe_fast_path_counterexample(f: int = 1) -> dict:
    """Deterministic schedule showing a 2f+1 one-round fast path is unsafe.

    n = 3f + 1, view 0 led by a Byzantine replica that equivocates. The
    f Byzantine and f + 1 honest replicas vote y: 2f + 1 = n - f votes, so
    under the unsafe rule one honest replica receives a fast certificate
    and decides y. The other f honest replicas voted x. In view 1 the
    leader's NEW-VIEW set (n - f messages) holds the f Byzantine (now
    claiming x), the f x-voters, and a single y-voter; every other
    y-voter is slow. x now has 2f claims against one for y, so the view
    must propose x, and the slow path decides x: two honest replicas
    decide differently. Under the safe quorum (n), 2f + 1 votes are not a
    certificate, so nothing was decided in view 0.
    """
    n = 3 * f + 1
    byz = list(range(f))
    honest = list(range(f, n))
    y_honest, x_honest = honest[:f + 1], honest[f + 1:]
    unsafe_fast = n - f
    y_votes = len(byz) + len(y_honest)
    nvs = ([NewView(i, None, (0, "x")) for i in byz]
           + [NewView(i, None, (0, "x")) for i in x_honest]
           + [NewView(y_honest[0], None, (0, "y"))])
    assert len(nvs) == n - f
    counts = {"x": 2 * f, "y": 1}
    # Any deterministic rule must follow the majority of claims here.
    forced = max(counts, key=counts.get)
    return {"n": n, "f": f, "unsafe_quorum": unsafe_fast,
            "view0_votes_for_y": y_votes,
            "view0_fast_decided": "y" if y_votes >= unsafe_fast else None,
            "view1_forced_value": forced,
            "conflict": y_votes >= unsafe_fast and forced == "x",
            "safe_quorum": fast_quorum(n, f),
            "view0_fast_decided_under_safe_quorum":
                "y" if y_votes >= fast_quorum(n, f) else None,
            "unsafe_select_returns": select(nvs, n, f, unsafe_fast)}


if __name__ == "__main__":
    for n in (4, 7, 10):
        print("safe fast quorum:", fuzz(n, 400, seed=1))
    for n in (4, 7):
        r = fuzz(n, 400, n_fast=n - max_faults(n), seed=2)
        print(f"unsafe n_fast = n - f (n={n}): unsafe runs {r['unsafe']}/400")
    print(unsafe_fast_path_counterexample(1))
