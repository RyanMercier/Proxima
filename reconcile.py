"""
reconcile.py -- Exact set reconciliation for Proxima v2 (two-syndrome sync).

Replaces the v1 bloom filter throughout the protocol. Architecture
(docs/v2_formal_foundations.md, Section 9): estimate with the hard code,
decode with the easy code. The DPMH digest distance sizes the divergence
(d_hat); an Invertible Bloom Lookup Table (IBLT, Goodrich-Mitzenmacher)
sized by d_hat recovers exactly which elements differ, with no false
positives and communication proportional to the true difference. If the
decode fails (d_hat unreliable), the sketch doubles, rateless-style.

Cells hold (count, key_sum XOR, check_sum XOR); subtracting two tables
built over A and B leaves a table of the symmetric difference, peeled by
repeatedly extracting cells with count = +-1 whose checksum matches.

Wire cost model used by the simulator: CELL_BYTES per cell per attempt.
"""

import hashlib

CELL_BYTES = 16          # count (4) + key_sum (8) + check_sum (4)
HASH_COUNT = 3           # independent cell indices per key
ALPHA = 1.5              # cells per expected difference (peeling threshold ~1.22
                         # asymptotically; small tables need more headroom)
MIN_CELLS = 16


def _key64(item: str) -> int:
    """8-byte key for an item (txid prefix)."""
    return int.from_bytes(hashlib.sha256(item.encode()).digest()[:8], "big")


def _check(key: int) -> int:
    """4-byte checksum guarding pure-cell detection."""
    return int.from_bytes(
        hashlib.sha256(b"chk" + key.to_bytes(8, "big")).digest()[:4], "big")


def _cells_for(key: int, m: int, salt: int) -> list:
    """HASH_COUNT distinct cell indices for a key."""
    idxs = []
    i = 0
    while len(idxs) < HASH_COUNT:
        h = hashlib.sha256(
            salt.to_bytes(4, "big") + i.to_bytes(2, "big")
            + key.to_bytes(8, "big")).digest()
        idx = int.from_bytes(h[:8], "big") % m
        if idx not in idxs:
            idxs.append(idx)
        i += 1
    return idxs


class IBLT:
    def __init__(self, m: int, salt: int = 0):
        self.m = m
        self.salt = salt
        self.count = [0] * m
        self.key_sum = [0] * m
        self.chk_sum = [0] * m

    def insert(self, item: str, sign: int = 1) -> None:
        key = _key64(item)
        chk = _check(key)
        for idx in _cells_for(key, self.m, self.salt):
            self.count[idx] += sign
            self.key_sum[idx] ^= key
            self.chk_sum[idx] ^= chk

    @classmethod
    def of(cls, items, m: int, salt: int = 0) -> "IBLT":
        t = cls(m, salt)
        for it in items:
            t.insert(it)
        return t

    def subtract(self, other: "IBLT") -> "IBLT":
        assert self.m == other.m and self.salt == other.salt
        out = IBLT(self.m, self.salt)
        out.count = [a - b for a, b in zip(self.count, other.count)]
        out.key_sum = [a ^ b for a, b in zip(self.key_sum, other.key_sum)]
        out.chk_sum = [a ^ b for a, b in zip(self.chk_sum, other.chk_sum)]
        return out

    def peel(self):
        """Decode the difference table.

        Returns (only_in_A_keys, only_in_B_keys, ok) where A is the table
        subtract() was called on. ok = False means the table was too small
        for the true difference (caller should retry larger).
        """
        count = list(self.count)
        key_sum = list(self.key_sum)
        chk_sum = list(self.chk_sum)
        a_keys, b_keys = set(), set()
        progress = True
        while progress:
            progress = False
            for idx in range(self.m):
                c = count[idx]
                if c in (1, -1):
                    key = key_sum[idx]
                    if key == 0 or _check(key) != chk_sum[idx]:
                        continue
                    (a_keys if c == 1 else b_keys).add(key)
                    for j in _cells_for(key, self.m, self.salt):
                        count[j] -= c
                        key_sum[j] ^= key
                        chk_sum[j] ^= _check(key)
                    progress = True
        ok = all(c == 0 for c in count) and all(k == 0 for k in key_sum)
        return a_keys, b_keys, ok


def size_for(d_hint: float) -> int:
    """Cell count for an expected difference of d_hint elements."""
    return max(MIN_CELLS, int(ALPHA * (d_hint + 6)) + 1)


def reconcile(set_a, set_b, d_hint: float = None, max_rounds: int = 8) -> dict:
    """Recover the exact symmetric difference between two sets.

    Models the wire exchange: the responder sends an IBLT of its set sized
    by d_hint; the requester subtracts its own table and peels. On decode
    failure the table doubles (one extra round trip each time).

    Returns the difference (as items), bytes and rounds spent.
    """
    set_a, set_b = list(set_a), list(set_b)
    true_d = len(set(set_a) ^ set(set_b))
    if d_hint is None:
        d_hint = max(true_d, 1)          # caller had no estimate
    m = size_for(d_hint)
    bytes_used = 0
    rounds = 0
    for attempt in range(max_rounds):
        rounds += 1
        salt = attempt                    # fresh hash functions per attempt
        t_a = IBLT.of(set_a, m, salt)
        t_b = IBLT.of(set_b, m, salt)
        bytes_used += m * CELL_BYTES      # one table crosses the wire
        diff = t_a.subtract(t_b)
        a_keys, b_keys, ok = diff.peel()
        if ok:
            key_map = {_key64(x): x for x in set(set_a) | set(set_b)}
            only_a = {key_map[k] for k in a_keys if k in key_map}
            only_b = {key_map[k] for k in b_keys if k in key_map}
            return {"only_a": only_a, "only_b": only_b, "ok": True,
                    "bytes": bytes_used, "rounds": rounds, "cells": m,
                    "true_d": true_d}
        m *= 2
    return {"only_a": set(), "only_b": set(), "ok": False,
            "bytes": bytes_used, "rounds": rounds, "cells": m,
            "true_d": true_d}


def sync_cost(d_hint: float) -> int:
    """Happy-path wire bytes for a one-shot sketch sized by d_hint."""
    return size_for(d_hint) * CELL_BYTES
